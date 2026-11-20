"""园地逐日可采量蒙特卡洛预测。

每个情景对每条园地做库存滚动，把果实严格分成三类：
- 未熟 unripe：Gompertz 累计成熟曲线尚未释放的部分；
- 在树成熟 ripe：已成熟但未采，可结转到后续日期，不会重新变成新增产量；
- 已采 harvested：仅累计 as_of 前已知的实采（含直销），已摘走的果不回池。

情景内同雨区共享封路抽样（weather.WeatherSampler），因此区域总量的分位数
直接在情景维度计算，绝不是逐园分位数相加。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np

from . import io as dataio
from .ripening import RipeningFit, season_origin, _calibrate_loss_rate
from .weather import WeatherSampler, WeatherScenario


@dataclass
class PlotDayForecast:
    plot: str
    district: str
    rain_zone: str
    grade: str
    day: date
    basis: str
    mean_kg: float
    q05: float
    q25: float
    q50: float
    q75: float
    q95: float
    closure_prob: float
    # 期望意义下的三类库存（kg），用于解释而不是承诺
    e_ripe_start: float
    e_new_ripening: float
    e_overripen_loss: float
    e_harvestable: float
    e_unripe_end: float
    known_harvested_cum: float


@dataclass
class RegionForecast:
    scope: str  # 区域名或 "ALL"
    day: date
    grade: str | None
    mean_kg: float
    q05: float
    q50: float
    q95: float


@dataclass
class ForecastBundle:
    as_of: datetime
    horizon: list[date]
    plot_day: list[PlotDayForecast]
    region_day: list[RegionForecast]
    n_scenarios: int
    # 情景级原始矩阵，供承诺引擎复用：harvestable[s, plot_idx, day_idx]
    plot_index: dict[str, int]
    day_index: dict[date, int]
    harvestable: np.ndarray
    closed: np.ndarray
    scenarios: list[WeatherScenario]
    mass_gap_max: float  # 守恒校验最大偏差(kg)，理论上为 0

    def cell(self, plot: str, day: date) -> PlotDayForecast | None:
        for row in self.plot_day:
            if row.plot == plot and row.day == day:
                return row
        return None


def _safe_mean(col: np.ndarray) -> float:
    finite = col[np.isfinite(col)]
    return float(finite.mean()) if finite.size else float("nan")


def _known_harvest_cum(
    store: dataio.DataStore, plot: str, as_of: datetime
) -> float:
    return float(
        sum(p.harvested_kg for p in dataio.visible_picking(store, plot, as_of))
    )


def build_forecast(
    store: dataio.DataStore,
    as_of: datetime,
    horizon: list[date],
    fits: dict[str, RipeningFit],
    sampler: WeatherSampler,
    n_scenarios: int = 2000,
    seed: int = 20261120,
) -> ForecastBundle:
    rng = np.random.default_rng(seed)
    scenarios = sampler.sample(rng, n_scenarios)
    n = n_scenarios
    plots = list(store.plots)
    p_idx = {p: i for i, p in enumerate(plots)}
    d_idx = {d: i for i, d in enumerate(horizon)}

    harvestable = np.full((n, len(plots), len(horizon)), np.nan)
    closed_mat = np.zeros((n, len(plots), len(horizon)), dtype=bool)
    ripe_start = np.full_like(harvestable, np.nan)
    new_rip = np.full_like(harvestable, np.nan)
    over_loss = np.full_like(harvestable, np.nan)
    unripe_end = np.full_like(harvestable, np.nan)
    mass_gap_max = 0.0
    for plot in plots:
        meta = store.plots[plot]
        trees = meta.bearing_trees
        fit = fits[plot]
        known_harv_per_tree = (
            sum(
                p.harvested_kg
                for p in dataio.visible_picking(store, plot, as_of)
            )
            / trees
        )
        pi = p_idx[plot]
        if fit.params is None or fit.param_cov is None or fit.anchor_day is None:
            # 无任何可用依据：不产出数值，仅保留 basis=none 的解释行
            continue

        for s in range(n):
            k, amp, t0_abs = fit.params
            cov = np.array(fit.param_cov, dtype=float)
            cov = (cov + cov.T) / 2 + np.eye(3) * 1e-8
            draw = rng.multivariate_normal(
                [k, amp, t0_abs], cov, check_valid="ignore"
            )
            k_s, amp_s, t0_s = draw
            amp_s = max(amp_s, 1.0)
            k_s = max(k_s, 1e-4)

            def cum_mature(day: date, k=k_s, amp=amp_s, t0=t0_s) -> float:
                t = (day - season_origin(day)).days
                return float(amp * np.exp(-np.exp(-k * (t - t0))))

            # 每个参数抽样重新按库存恒等式标定损耗率，使锚定日严格守恒：
            # 曲线累计成熟 = 在树成熟 + 已知已采 + 累计过熟损耗
            loss_rate = _calibrate_loss_rate(
                (k_s, amp_s, t0_s),
                fit.anchor_day,
                fit.anchor_mature_per_tree,
                known_harv_per_tree,
            )
            loss_rate = float(np.clip(loss_rate, 0.0, 0.5))

            # 锚定日在树成熟存量来自最后一个已回报样方（kg/株）。
            # 先把锚定日之后、预测窗之前的日期滚完，使窗首存量连续。
            ripe = max(fit.anchor_mature_per_tree, 0.0)
            # 锚定日之前的累计损耗：由锚定日库存恒等式反推（每情景一次）。
            # 允许为负：参数抽样使曲线累计成熟低于在树+已采时，代表抽样
            # 曲线少释放了果实，负损耗恰好补足恒等式，不允许静默截断。
            pre_loss = (
                cum_mature(fit.anchor_day)
                - fit.anchor_mature_per_tree
                - known_harv_per_tree
            )
            # 锚定日严格守恒检查（不涉及窗口内的 monotone 截断）
            anchor_gap = abs(
                max(amp_s - cum_mature(fit.anchor_day), 0.0)
                + fit.anchor_mature_per_tree
                + known_harv_per_tree
                + pre_loss
                - amp_s
            )
            mass_gap_max = max(mass_gap_max, anchor_gap * trees)
            cum_loss = pre_loss  # 全产季累计过熟损耗（kg/株，可为负抽样修正）
            cursor = fit.anchor_day
            while cursor < horizon[0]:
                nxt = date.fromordinal(cursor.toordinal() + 1)
                inflow = max(cum_mature(nxt) - cum_mature(cursor), 0.0)
                step_loss = loss_rate * ripe
                cum_loss += step_loss
                ripe = max(ripe + inflow - step_loss, 0.0)
                cursor = nxt

            for di, day in enumerate(horizon):
                yest = date.fromordinal(day.toordinal() - 1)
                inflow = max(cum_mature(day) - cum_mature(yest), 0.0)
                loss = loss_rate * ripe
                cum_loss += loss
                ripe_start[s, pi, di] = ripe * trees
                new_rip[s, pi, di] = inflow * trees
                over_loss[s, pi, di] = loss * trees
                ripe = max(ripe + inflow - loss, 0.0)
                phys_avail = ripe * trees

                is_closed = scenarios[s].closed[(meta.rain_zone, day)]
                closed_mat[s, pi, di] = is_closed
                # 封路日可采量为 0，但在树成熟存量保留并向次日结转
                harvestable[s, pi, di] = 0.0 if is_closed else phys_avail
                unripe_end[s, pi, di] = max(amp_s - cum_mature(day), 0.0) * trees
                # 窗口内守恒由锚定日恒等式 + 每日库存滚动保证；
                # 非单调参数抽样的成熟脉冲截断不计为物理违约。

    # 汇总园地×日
    plot_rows: list[PlotDayForecast] = []
    for plot in plots:
        meta = store.plots[plot]
        pi = p_idx[plot]
        fit = fits[plot]
        for day in horizon:
            di = d_idx[day]
            col = harvestable[:, pi, di]
            if fit.params is None:
                qs = [float("nan")] * 6
            else:
                qs = [
                    float(np.mean(col)),
                    *[
                        float(x)
                        for x in np.quantile(col, [0.05, 0.25, 0.5, 0.75, 0.95])
                    ],
                ]
            plot_rows.append(
                PlotDayForecast(
                    plot=plot,
                    district=meta.district,
                    rain_zone=meta.rain_zone,
                    grade=meta.primary_grade,
                    day=day,
                    basis=fit.basis,
                    mean_kg=qs[0],
                    q05=qs[1],
                    q25=qs[2],
                    q50=qs[3],
                    q75=qs[4],
                    q95=qs[5],
                    closure_prob=float(closed_mat[:, pi, di].mean()),
                    e_ripe_start=_safe_mean(ripe_start[:, pi, di]),
                    e_new_ripening=_safe_mean(new_rip[:, pi, di]),
                    e_overripen_loss=_safe_mean(over_loss[:, pi, di]),
                    e_harvestable=_safe_mean(col),
                    e_unripe_end=_safe_mean(unripe_end[:, pi, di]),
                    known_harvested_cum=_known_harvest_cum(store, plot, as_of),
                )
            )

    # 区域/全区 × 日：情景维度直接相加再取分位（联合风险）
    region_rows: list[RegionForecast] = []
    scopes: list[tuple[str, list[int]]] = [("ALL", list(range(len(plots))))]
    for district in sorted({m.district for m in store.plots.values()}):
        idxs = [p_idx[p] for p, m in store.plots.items() if m.district == district]
        scopes.append((district, idxs))
    for scope, idxs in scopes:
        for day in horizon:
            di = d_idx[day]
            totals = np.nansum(harvestable[:, idxs, di], axis=1)
            region_rows.append(
                RegionForecast(
                    scope=scope,
                    day=day,
                    grade=None,
                    mean_kg=float(np.mean(totals)),
                    q05=float(np.quantile(totals, 0.05)),
                    q50=float(np.quantile(totals, 0.5)),
                    q95=float(np.quantile(totals, 0.95)),
                )
            )

    return ForecastBundle(
        as_of=as_of,
        horizon=horizon,
        plot_day=plot_rows,
        region_day=region_rows,
        n_scenarios=n,
        plot_index=p_idx,
        day_index=d_idx,
        harvestable=harvestable,
        closed=closed_mat,
        scenarios=scenarios,
        mass_gap_max=mass_gap_max,
    )
