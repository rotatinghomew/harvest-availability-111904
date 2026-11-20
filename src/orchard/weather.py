"""多版本雨量预报、封路概率与同雨区联合情景。

历史（as_of 之前已发生的目标日）用同日 05:15 发布版本当作天气实况的代理，
标定两件事：
1. 不同预见期（lead = 目标日 - 发布日）雨量预报的偏差与离散度；
2. 封路阈值的逻辑回归概率（合成资料中封路与 >=40mm 雨完全对应，
   但预报有误差，所以承诺侧拿到的仍是概率而非确定开关）。

同一雨区下的若干园地共享同一条封路情景，这是“一场雨几个园一起落空”
风险的来源；不同雨区按历史同日封路联合频率生成相关情景。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np

from . import io as dataio

CLOSURE_RAIN = 40.0  # 历史实况中的硬阈值，仅用于标定，不直接用于未来承诺


def _lead_days(f: dataio.Forecast) -> int:
    return (f.target_day - f.issued_at.date()).days


@dataclass
class RainErrorModel:
    """每个预见期的乘法误差 log(预报/实况代理) 的均值与标准差。"""
    lead_stats: dict[int, tuple[float, float]]
    # 雨区间同日误差的相关系数（用残差估计）
    zone_correlation: float

    def realized_draw(
        self, forecast_rain: float, lead: int, rng: np.random.Generator
    ) -> float:
        mu, sigma = self.lead_stats.get(
            lead, self.lead_stats[max(self.lead_stats)]
        )
        err = np.exp(rng.normal(mu, sigma))
        return max(0.0, forecast_rain * err)


def calibrate_rain_errors(
    store: dataio.DataStore, as_of: datetime
) -> RainErrorModel:
    # 实况代理：目标日当天发布的版本（lead=0），且该日 road_log 已可见
    actuals: dict[tuple[str, date], float] = {}
    visible_road_days = {
        (r.rain_zone, r.service_day)
        for r in store.road_log
        if r.known_at <= as_of
    }
    for f in store.forecasts:
        if f.issued_at > as_of:
            continue
        if _lead_days(f) == 0 and (f.rain_zone, f.target_day) in visible_road_days:
            actuals[(f.rain_zone, f.target_day)] = f.rain_mm

    by_lead: dict[int, list[float]] = {}
    residual_by_zone: dict[str, list[tuple[date, float]]] = {}
    for f in store.forecasts:
        if f.issued_at > as_of:
            continue
        actual = actuals.get((f.rain_zone, f.target_day))
        if actual is None or actual <= 0 or f.rain_mm <= 0:
            continue
        lead = _lead_days(f)
        log_err = float(np.log(f.rain_mm / actual))
        by_lead.setdefault(lead, []).append(log_err)
        residual_by_zone.setdefault(f.rain_zone, []).append(
            (f.target_day, log_err)
        )

    lead_stats: dict[int, tuple[float, float]] = {}
    for lead, vals in by_lead.items():
        arr = np.array(vals)
        lead_stats[lead] = (float(arr.mean()), float(arr.std(ddof=1) or 0.05))
    if not lead_stats:
        lead_stats = {0: (0.0, 0.15), 1: (0.0, 0.2), 3: (0.0, 0.3)}

    # 同日跨雨区残差相关
    zone_series: dict[str, dict[date, float]] = {}
    for zone, pairs in residual_by_zone.items():
        zone_series[zone] = {d: v for d, v in pairs}
    corrs: list[float] = []
    zones = list(zone_series)
    for i in range(len(zones)):
        for j in range(i + 1, len(zones)):
            common = sorted(set(zone_series[zones[i]]) & set(zone_series[zones[j]]))
            if len(common) >= 5:
                a = np.array([zone_series[zones[i]][d] for d in common])
                b = np.array([zone_series[zones[j]][d] for d in common])
                if a.std() > 0 and b.std() > 0:
                    corrs.append(float(np.corrcoef(a, b)[0, 1]))
    correlation = float(np.clip(np.mean(corrs), 0.0, 0.95)) if corrs else 0.0
    return RainErrorModel(lead_stats=lead_stats, zone_correlation=correlation)


@dataclass
class ClosureModel:
    """逻辑回归：logit P(封路) = a + b * 雨量；历史完全可分时做硬间隔收缩。"""
    intercept: float
    coef: float

    def closure_prob(self, rain_mm: float) -> float:
        if rain_mm <= 0:
            return 0.0
        z = self.intercept + self.coef * rain_mm
        return float(1.0 / (1.0 + np.exp(-np.clip(z, -60, 60))))


def calibrate_closure(
    store: dataio.DataStore, as_of: datetime
) -> ClosureModel:
    rows: list[tuple[float, bool]] = []
    actual_rain = {
        (f.rain_zone, f.target_day): f.rain_mm
        for f in store.forecasts
        if f.issued_at <= as_of and _lead_days(f) == 0
    }
    for r in store.road_log:
        if r.known_at > as_of:
            continue
        rain = actual_rain.get((r.rain_zone, r.service_day))
        if rain is not None:
            rows.append((rain, not r.passable))
    if not rows:
        return ClosureModel(intercept=-40.0, coef=1.0)
    x = np.array([r[0] for r in rows])
    y = np.array([1.0 if r[1] else 0.0 for r in rows])
    # 牛顿法拟合带 L2 的逻辑回归，避免完全可分时系数发散
    a, b = -20.0, 0.5
    lr = 0.01
    for _ in range(4000):
        z = a + b * x
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
        grad_a = np.sum(p - y) + 0.1 * a
        grad_b = np.sum((p - y) * x) + 0.1 * b
        a -= lr * grad_a / len(x)
        b -= lr * grad_b / len(x)
    return ClosureModel(intercept=float(a), coef=float(b))


@dataclass
class WeatherScenario:
    """一次蒙特卡洛抽样的全部天气结果。"""
    rain: dict[tuple[str, date], float]
    closed: dict[tuple[str, date], bool]
    closure_prob: dict[tuple[str, date], float]


class WeatherSampler:
    def __init__(
        self,
        store: dataio.DataStore,
        as_of: datetime,
        horizon: list[date],
        rain_model: RainErrorModel,
        closure_model: ClosureModel,
    ):
        self.store = store
        self.as_of = as_of
        self.horizon = horizon
        self.rain_model = rain_model
        self.closure_model = closure_model
        self.zones = sorted({p.rain_zone for p in store.plots.values()})
        self.latest = dataio.latest_forecasts(store, as_of)
        self.known_road = dataio.visible_road(store, as_of)

    def _forecast_for(self, zone: str, day: date) -> tuple[float, int]:
        f = self.latest.get((zone, day))
        if f is None:
            return float("nan"), -1
        return f.rain_mm, (day - f.issued_at.date()).days

    def closure_probability(self, zone: str, day: date) -> float:
        """点估计封路概率：对预报误差做解析近似（对数正态积分 + 逻辑函数）。"""
        known = self.known_road.get((zone, day))
        if known is not None:
            return 0.0 if known else 1.0
        rain, lead = self._forecast_for(zone, day)
        if np.isnan(rain):
            return float("nan")
        mu, sigma = self.rain_model.lead_stats.get(
            lead, self.rain_model.lead_stats[max(self.rain_model.lead_stats)]
        )
        # 数值积分实际雨量分布
        grid = np.linspace(max(mu - 4 * sigma, -6), mu + 4 * sigma, 81)
        dens = np.exp(-0.5 * ((grid - mu) / sigma) ** 2) / (
            sigma * np.sqrt(2 * np.pi)
        )
        realized = rain * np.exp(grid)
        probs = np.array([self.closure_model.closure_prob(r) for r in realized])
        return float(np.sum(dens * probs) / np.sum(dens))

    def sample(self, rng: np.random.Generator, n: int) -> list[WeatherScenario]:
        rho = self.rain_model.zone_correlation
        z_idx = {z: i for i, z in enumerate(self.zones)}
        cov = np.full((len(self.zones), len(self.zones)), rho)
        np.fill_diagonal(cov, 1.0)
        chol = np.linalg.cholesky(cov)
        scenarios: list[WeatherScenario] = []
        for _ in range(n):
            rain: dict[tuple[str, date], float] = {}
            closed: dict[tuple[str, date], bool] = {}
            prob: dict[tuple[str, date], float] = {}
            for day in self.horizon:
                base = rng.standard_normal(len(self.zones))
                corr_norm = chol @ base
                for zone in self.zones:
                    key = (zone, day)
                    known = self.known_road.get(key)
                    if known is not None:
                        rain[key] = float("nan")  # 历史日不抽样
                        closed[key] = not known
                        prob[key] = 0.0 if known else 1.0
                        continue
                    forecast_rain, lead = self._forecast_for(zone, day)
                    if np.isnan(forecast_rain):
                        rain[key] = float("nan")
                        closed[key] = False
                        prob[key] = float("nan")
                        continue
                    mu, sigma = self.rain_model.lead_stats.get(
                        lead,
                        self.rain_model.lead_stats[
                            max(self.rain_model.lead_stats)
                        ],
                    )
                    log_err = mu + sigma * corr_norm[z_idx[zone]]
                    realized = max(0.0, forecast_rain * np.exp(log_err))
                    p = self.closure_model.closure_prob(realized)
                    rain[key] = realized
                    prob[key] = p
                    closed[key] = rng.random() < p
            scenarios.append(
                WeatherScenario(rain=rain, closed=closed, closure_prob=prob)
            )
        return scenarios
