"""样方驱动的成熟模型。

核心对象是每条园地的“每样株累计成熟重量” Gompertz 曲线（日轴统一为
距产季起点 11 月 1 日的天数，避免绝对日序数与有界优化冲突）：
    C(t) = K * exp(-exp(-k * (t - t0)))
样方观测到的 mature_kg 是当时仍在树上的成熟存量（此前已采部分不包括在内），
因此用库存恒等式把观测还原成累计成熟量再拟合：
    C(t_i) ≈ mature_obs(t_i) + 截至 t_i 的累计采收(kg/株) + 过熟损耗
在预测窗内逐日做库存滚动，保证同一片果只能在成熟脉冲与过熟损耗之间移动，
不会被重复计为新增产量。

资料不足时按同区同等级的园地合并拟合，并显式标注 basis：
- plot:   本园样方足够
- pooled: 本园样方过薄，借用同区同等级曲线（误差单独分层，不冒充本园保证）
- none:   无任何可用依据，不出具点预测
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
from scipy.optimize import curve_fit

from . import io as dataio

MIN_PLOT_POINTS = 6  # 本园独立拟合所需最少样方数
SEASON_MONTH = 11
SEASON_DAY = 1


def season_origin(day: date) -> date:
    return date(day.year, SEASON_MONTH, SEASON_DAY)


def _gompertz(t: np.ndarray, k: float, amp: float, t0: float) -> np.ndarray:
    return amp * np.exp(-np.exp(-k * (t - t0)))


@dataclass
class RipeningFit:
    plot: str
    basis: str  # plot / pooled / none
    pooled_from: list[str]
    pooled_cross_district: bool
    points: int
    # (k, K, t0)，t0 为距产季起点的天数
    params: tuple[float, float, float] | None
    param_cov: np.ndarray | None
    residual_sd: float
    # 由库存恒等式反推的过熟日损耗率（使锚定日曲线值=在树+已采+累计损耗）
    loss_rate: float
    anchor_day: date | None
    anchor_mature_per_tree: float
    anchor_harvest_per_tree: float

    def cumulative_mature(self, day_index: np.ndarray) -> np.ndarray:
        if self.params is None:
            return np.full_like(day_index, np.nan, dtype=float)
        k, amp, t0 = self.params
        return _gompertz(day_index, k, amp, t0)


def _harvest_per_tree_up_to(
    store: dataio.DataStore, plot: str, day: date, as_of: datetime
) -> float:
    trees = store.plots[plot].bearing_trees
    total = 0.0
    for p in dataio.visible_picking(store, plot, as_of):
        if p.day <= day:
            total += p.harvested_kg
    return total / trees


def _reconstruct_cumulative(
    store: dataio.DataStore,
    plot: str,
    as_of: datetime,
    season_year: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """把在树成熟存量观测还原成累计成熟（kg/株），并估计过熟损耗率。

    样方按产季（年）分开；默认取 as_of 所在产季。
    """
    samples = [
        s
        for s in dataio.visible_samples(store, plot, as_of)
        if season_year is None or s.day.year == season_year
    ]
    if not samples:
        return np.empty(0), np.empty(0), 0.0
    trees = store.plots[plot].bearing_trees
    origin = season_origin(samples[0].day)
    x = np.array([(s.day - origin).days for s in samples], dtype=float)
    mature = np.array([s.mature_kg for s in samples], dtype=float)
    harv = np.array(
        [
            sum(
                p.harvested_kg
                for p in dataio.visible_picking(store, plot, as_of)
                if p.day <= s.day and p.day.year == samples[0].day.year
            )
            / trees
            for s in samples
        ]
    )
    # 过熟/自然落果：晚期相邻观测成熟量下降且超过区间采收时，把净下降计为损耗率
    declines = []
    for i in range(1, len(samples)):
        gap = (samples[i].day - samples[i - 1].day).days
        net = mature[i - 1] - mature[i]
        interval_harv = harv[i] - harv[i - 1]
        if net > interval_harv and mature[i - 1] > 0:
            declines.append((net - interval_harv) / gap / mature[i - 1])
    loss_rate = float(np.median(declines)) if declines else 0.0
    # 逐日累计损耗（按前一日成熟存量近似）
    cum_loss = np.zeros_like(mature)
    stock = mature[0]
    for i in range(1, len(samples)):
        gap = (samples[i].day - samples[i - 1].day).days
        loss = 0.0
        cur = stock
        for _ in range(gap):
            step = loss_rate * cur
            loss += step
            cur -= step
        cum_loss[i] = cum_loss[i - 1] + loss
        stock = mature[i]
    cumulative = mature + harv + cum_loss
    return x, cumulative, loss_rate


def _fit_gompertz(
    x: np.ndarray, y: np.ndarray
) -> tuple[tuple[float, float, float], np.ndarray, float]:
    amp0 = max(float(y[-1]) * 1.15, 1.0)
    # t0 初值：累计成熟达到约 0.37K 的位置
    target = 0.37 * amp0
    idx = int(np.searchsorted(y, target))
    t0_guess = float(x[min(idx, len(x) - 1)])
    k0 = 0.15
    try:
        popt, pcov = curve_fit(
            _gompertz,
            x,
            y,
            p0=[k0, amp0, t0_guess],
            bounds=([1e-4, 1e-6, -10.0], [2.0, max(amp0 * 5, 10.0), 60.0]),
            maxfev=40000,
        )
    except (RuntimeError, ValueError):
        popt = np.array([k0, amp0, t0_guess])
        pcov = np.diag([0.05, (0.35 * amp0) ** 2, 4.0])
    resid = float(np.sqrt(np.mean((_gompertz(x, *popt) - y) ** 2)))
    return (float(popt[0]), float(popt[1]), float(popt[2])), pcov, resid


def _calibrate_loss_rate(
    params: tuple[float, float, float],
    anchor_day: date,
    ripe_stock: float,
    harvested_per_tree: float,
) -> float:
    """从锚定日库存恒等式反推恒定日损耗率 r：

    C(T) = ripe(T) + harv(T) + Σ_{t<=T} r·ripe(t-1)
    用二分法求解，使锚定日累计损耗恰好等于曲线累计成熟与在树+已采之差。
    """
    k, amp, t0 = params
    origin = season_origin(anchor_day)
    t_anchor = (anchor_day - origin).days

    def cum_mature(t: float) -> float:
        return float(amp * np.exp(-np.exp(-k * (t - t0))))

    target_loss = max(
        cum_mature(t_anchor) - ripe_stock - harvested_per_tree, 0.0
    )
    if target_loss <= 1e-9:
        return 0.0

    def cum_loss(rate: float) -> float:
        # 从产季首日滚动到锚定日；初始在树取曲线首日累计成熟
        ripe = max(cum_mature(0.0), 0.0)
        total = 0.0
        for t in range(1, t_anchor + 1):
            step = rate * ripe
            total += step
            inflow = max(cum_mature(float(t)) - cum_mature(float(t - 1)), 0.0)
            ripe = max(ripe + inflow - step, 0.0)
        return total

    lo, hi = 0.0, 0.5
    for _ in range(60):
        mid = (lo + hi) / 2
        if cum_loss(mid) < target_loss:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2)


def fit_plot(
    store: dataio.DataStore, plot: str, as_of: datetime
) -> RipeningFit:
    season_year = as_of.year
    samples = [
        s
        for s in dataio.visible_samples(store, plot, as_of)
        if s.day.year == season_year
    ]
    anchor_day = samples[-1].day if samples else None
    anchor_mature = samples[-1].mature_kg if samples else 0.0
    anchor_harv = (
        _harvest_per_tree_up_to(store, plot, anchor_day, as_of)
        if anchor_day
        else 0.0
    )
    if len(samples) >= MIN_PLOT_POINTS:
        x, y, _ = _reconstruct_cumulative(
            store, plot, as_of, season_year=season_year
        )
        params, pcov, resid = _fit_gompertz(x, y)
        loss_rate = _calibrate_loss_rate(
            params, anchor_day, anchor_mature, anchor_harv
        )
        return RipeningFit(
            plot=plot,
            basis="plot",
            pooled_from=[],
            pooled_cross_district=False,
            points=len(samples),
            params=params,
            param_cov=pcov,
            residual_sd=max(resid, 1.0),
            loss_rate=loss_rate,
            anchor_day=anchor_day,
            anchor_mature_per_tree=anchor_mature,
            anchor_harvest_per_tree=anchor_harv,
        )

    # 薄样方：优先合并同区同等级且样方充足的园地；同区无捐赠者时放宽到
    # 同等级跨区（仍显式标注 pooled，误差单独分层，不冒充本园保证）
    target_meta = store.plots[plot]
    donors = [
        name
        for name, meta in store.plots.items()
        if name != plot
        and meta.district == target_meta.district
        and meta.primary_grade == target_meta.primary_grade
        and len(
            [
                s
                for s in dataio.visible_samples(store, name, as_of)
                if s.day.year == season_year
            ]
        )
        >= MIN_PLOT_POINTS
    ]
    cross_district = False
    if not donors:
        donors = [
            name
            for name, meta in store.plots.items()
            if name != plot
            and meta.primary_grade == target_meta.primary_grade
            and len(
                [
                    s
                    for s in dataio.visible_samples(store, name, as_of)
                    if s.day.year == season_year
                ]
            )
            >= MIN_PLOT_POINTS
        ]
        cross_district = True
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for donor in donors:
        x, y, _ = _reconstruct_cumulative(
            store, donor, as_of, season_year=season_year
        )
        if len(x):
            xs.append(x)
            ys.append(y)
    if xs:
        x_all = np.concatenate(xs)
        y_all = np.concatenate(ys)
        params, pcov, resid = _fit_gompertz(x_all, y_all)
        k, amp, t0 = params
        # 若本园有少量样方，用其均值水平修正幅度
        if samples:
            origin = season_origin(anchor_day)
            donor_anchor = float(
                np.mean(
                    [
                        _gompertz(
                            np.array([(s.day - origin).days], dtype=float),
                            k,
                            amp,
                            t0,
                        )[0]
                        for s in samples
                    ]
                )
            )
            obs_anchor = float(np.mean([s.mature_kg for s in samples]))
            if donor_anchor > 0:
                amp *= float(np.clip(obs_anchor / donor_anchor, 0.5, 2.0))
            params = (k, amp, t0)
        return RipeningFit(
            plot=plot,
            basis="pooled",
            pooled_from=donors,
            pooled_cross_district=cross_district,
            points=len(samples),
            params=params,
            param_cov=pcov,
            residual_sd=max(resid * 1.5, 3.0),  # 池化借用，主动放大不确定度
            loss_rate=_calibrate_loss_rate(
                params, anchor_day, anchor_mature, anchor_harv
            )
            if anchor_day
            else float(np.clip(resid / max(params[1], 1.0) / 12.0, 0.002, 0.03)),
            anchor_day=anchor_day,
            anchor_mature_per_tree=anchor_mature,
            anchor_harvest_per_tree=anchor_harv,
        )

    return RipeningFit(
        plot=plot,
        basis="none",
        pooled_from=[],
        pooled_cross_district=False,
        points=len(samples),
        params=None,
        param_cov=None,
        residual_sd=float("inf"),
        loss_rate=float("nan"),
        anchor_day=anchor_day,
        anchor_mature_per_tree=anchor_mature,
        anchor_harvest_per_tree=anchor_harv,
    )


def fit_all(
    store: dataio.DataStore, as_of: datetime
) -> dict[str, RipeningFit]:
    return {plot: fit_plot(store, plot, as_of) for plot in store.plots}
