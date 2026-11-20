"""分层回测误差。

在历史 as_of 时点只用当时可见的记录重建预测，再与后续实际观测比较。
实际“可采量”不等于送站重量：封路日为 0；开放日按库存恒等式从样方与
实采记录还原当日真实可采（在树成熟存量 + 当日新成熟 - 过熟损耗），
车满/直销压低的送站数不扣减可采能力。

分层维度各自独立可查：产季、可见记录量（薄样/正常）、道路状态、区域。
薄样地块的误差单独成组，绝不并入区级平均冒充本园保证。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np

from . import io as dataio
from .forecast import ForecastBundle, build_forecast
from .ripening import MIN_PLOT_POINTS, fit_all
from .weather import (
    WeatherSampler,
    calibrate_closure,
    calibrate_rain_errors,
)


@dataclass
class BacktestRow:
    as_of: datetime
    plot: str
    district: str
    season: int
    horizon_days: int
    basis: str
    record_band: str  # thin / normal / none
    road_state: str  # closed / open / unknown（事后实况分层）
    road_was_known: bool  # 预测时点该日道路状态是否已可见
    predicted_mean: float
    predicted_q05: float
    predicted_q95: float
    actual_harvestable: float
    abs_error: float
    pct_error: float
    covered_90: bool


@dataclass
class ErrorSummary:
    layer: str
    value: str
    n: int
    mae: float
    mape: float
    bias: float
    coverage_90: float
    thin_sample: bool

    def as_dict(self) -> dict:
        return {
            "layer": self.layer,
            "value": self.value,
            "n": self.n,
            "mae_kg": round(self.mae, 2),
            "mape": round(self.mape, 4),
            "bias_kg": round(self.bias, 2),
            "coverage_90": round(self.coverage_90, 4),
            "thin_sample": self.thin_sample,
        }


def actual_harvestable(
    store: dataio.DataStore, plot: str, day: date
) -> float:
    """用完整历史还原某园地某日的真实可采量（事后真相，仅供回测）。"""
    samples = [s for s in store.samples_by_plot.get(plot, []) if s.day <= day]
    if not samples:
        return float("nan")
    trees = store.plots[plot].bearing_trees
    # 找 day 当天或之前最近样方
    s0 = max(samples, key=lambda s: s.day)
    ripe = s0.mature_kg
    if s0.day == day:
        # 样方在晨时盘点，当日新成熟用下一个样方推断不到时按 0 处理
        pass
    # 若最近样方早于 day，用后续样方之间的净变化+采收线性滚动
    later = [s for s in store.samples_by_plot[plot] if s.day > day]
    cursor = s0.day
    while cursor < day:
        nxt_candidates = [s for s in store.samples_by_plot[plot] if s.day > cursor]
        nxt = min(nxt_candidates, key=lambda s: s.day) if nxt_candidates else None
        step_end = min(nxt.day if nxt else day, day)
        if nxt is not None:
            days_gap = (nxt.day - cursor).days
            harv_gap = sum(
                p.harvested_kg
                for p in store.picking_by_plot[plot]
                if cursor < p.day <= nxt.day
            ) / trees
            # 区间总成熟净流入（含损耗后的净值），按天均摊
            net_per_day = ((nxt.mature_kg - ripe) + harv_gap) / days_gap
        else:
            net_per_day = 0.0
        for d_offset in range(1, (step_end - cursor).days + 1):
            ripe = max(ripe + net_per_day, 0.0)
        cursor = step_end
        if nxt is not None and cursor == nxt.day:
            ripe = nxt.mature_kg
    # 当日道路状态决定可采
    zone = store.plots[plot].rain_zone
    road = next(
        (r for r in store.road_log if r.rain_zone == zone and r.service_day == day),
        None,
    )
    if road is not None and not road.passable:
        return 0.0
    return float(max(ripe * trees, 0.0))


def run_backtest(
    store: dataio.DataStore,
    as_of_points: list[datetime],
    horizon_days: tuple[int, ...] = (0, 1, 3, 5),
    n_scenarios: int = 800,
) -> list[BacktestRow]:
    rows: list[BacktestRow] = []
    for as_of in as_of_points:
        horizon = [
            (as_of.date() + timedelta(days=h))
            for h in horizon_days
        ]
        # 只回测有完整事后观测的目标日
        horizon = [d for d in horizon if d <= max(p.day for p in store.picking)]
        if not horizon:
            continue
        fits = fit_all(store, as_of)
        rain_model = calibrate_rain_errors(store, as_of)
        closure = calibrate_closure(store, as_of)
        sampler = WeatherSampler(
            store, as_of, horizon, rain_model, closure
        )
        bundle = build_forecast(
            store, as_of, horizon, fits, sampler,
            n_scenarios=n_scenarios,
            seed=20260101 + (as_of.date() - date(as_of.year, 1, 1)).days,
        )
        visible_road = dataio.visible_road(store, as_of)
        actual_road = {
            (r.rain_zone, r.service_day): r.passable for r in store.road_log
        }
        for plot, meta in store.plots.items():
            n_visible = len(
                [
                    s
                    for s in dataio.visible_samples(store, plot, as_of)
                    if s.day.year == as_of.year
                ]
            )
            band = (
                "none"
                if n_visible == 0
                else "thin"
                if n_visible < MIN_PLOT_POINTS
                else "normal"
            )
            for day in horizon:
                cell = bundle.cell(plot, day)
                actual = actual_harvestable(store, plot, day)
                if cell is None or np.isnan(actual):
                    continue
                known = visible_road.get((meta.rain_zone, day))
                # 道路状态按事后实况分层，同时保留“预测时未知”的信息
                was_known = (meta.rain_zone, day) in visible_road
                realized_closed = actual_road.get((meta.rain_zone, day))
                if realized_closed is False:
                    road_state = "closed"
                elif realized_closed is True:
                    road_state = "open"
                else:
                    road_state = "unknown"
                pred_mean = cell.mean_kg
                if np.isnan(pred_mean):
                    rows.append(
                        BacktestRow(
                            as_of, plot, meta.district, day.year,
                            (day - as_of.date()).days,
                            fits[plot].basis, band, road_state, was_known,
                            float("nan"), float("nan"), float("nan"),
                            actual, float("nan"), float("nan"), False,
                        )
                    )
                    continue
                rows.append(
                    BacktestRow(
                        as_of=as_of,
                        plot=plot,
                        district=meta.district,
                        season=day.year,
                        horizon_days=(day - as_of.date()).days,
                        basis=fits[plot].basis,
                        record_band=band,
                        road_state=road_state,
                        road_was_known=was_known,
                        predicted_mean=pred_mean,
                        predicted_q05=cell.q05,
                        predicted_q95=cell.q95,
                        actual_harvestable=actual,
                        abs_error=abs(pred_mean - actual),
                        pct_error=(
                            abs(pred_mean - actual) / actual
                            if actual > 1e-9
                            else float("nan")
                        ),
                        covered_90=bool(cell.q05 <= actual <= cell.q95),
                    )
                )
    return rows


def summarize(rows: list[BacktestRow]) -> list[ErrorSummary]:
    def group(predicate, layer, value, thin=False):
        sub = [r for r in rows if predicate(r) and not np.isnan(r.predicted_mean)]
        if not sub:
            return ErrorSummary(layer, value, 0, float("nan"), float("nan"),
                                float("nan"), float("nan"), thin)
        pct = np.array([r.pct_error for r in sub])
        pct = pct[np.isfinite(pct)]
        return ErrorSummary(
            layer=layer,
            value=value,
            n=len(sub),
            mae=float(np.mean([r.abs_error for r in sub])),
            mape=float(pct.mean()) if pct.size else float("nan"),
            bias=float(np.mean([r.predicted_mean - r.actual_harvestable for r in sub])),
            coverage_90=float(np.mean([r.covered_90 for r in sub])),
            thin_sample=thin,
        )

    out: list[ErrorSummary] = []
    seasons = sorted({r.season for r in rows})
    for season in seasons:
        out.append(group(lambda r, s=season: r.season == s, "season", str(season)))
    for band in ("normal", "thin", "none"):
        out.append(
            group(lambda r, b=band: r.record_band == b, "record_band", band,
                  thin=(band == "thin"))
        )
    for basis in ("plot", "pooled", "none"):
        out.append(
            group(lambda r, b=basis: r.basis == b, "fit_basis", basis,
                  thin=(basis == "pooled"))
        )
    for state in ("open", "closed", "unknown"):
        out.append(
            group(lambda r, st=state: r.road_state == st, "road_state", state)
        )
    for known in (True, False):
        out.append(
            group(
                lambda r, k=known: r.road_was_known == k,
                "road_known_at_forecast",
                "known" if known else "forecast_only",
            )
        )
    for district in sorted({r.district for r in rows}):
        out.append(
            group(lambda r, d=district: r.district == d, "district", district)
        )
    for h in sorted({r.horizon_days for r in rows}):
        out.append(
            group(lambda r, hh=h: r.horizon_days == hh, "horizon_days", str(h))
        )
    out.append(group(lambda r: True, "overall", "ALL"))
    return out
