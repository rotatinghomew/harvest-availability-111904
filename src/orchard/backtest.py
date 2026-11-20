"""按每次下单前可拿到的输入重放预测与承诺。"""
from __future__ import annotations

from datetime import datetime

from . import io as dataio
from .errors import BacktestRow, ErrorSummary, run_backtest, summarize
from .forecast import ForecastBundle, build_forecast
from .ledger import LedgerView, replay
from .planning import PlanningEngine, PlanOption
from .ripening import RipeningFit, fit_all
from .weather import (
    WeatherSampler,
    calibrate_closure,
    calibrate_rain_errors,
)


def forecast_at(
    store: dataio.DataStore,
    as_of: datetime,
    horizon: list,
    n_scenarios: int = 2000,
    seed: int = 20261120,
) -> tuple[dict[str, RipeningFit], ForecastBundle, WeatherSampler]:
    """在 as_of 时点只用可见记录构建完整预测。"""
    fits = fit_all(store, as_of)
    rain_model = calibrate_rain_errors(store, as_of)
    closure = calibrate_closure(store, as_of)
    sampler = WeatherSampler(store, as_of, horizon, rain_model, closure)
    bundle = build_forecast(
        store, as_of, horizon, fits, sampler,
        n_scenarios=n_scenarios, seed=seed,
    )
    return fits, bundle, sampler


def planning_at(
    store: dataio.DataStore,
    as_of: datetime,
    horizon: list,
    n_scenarios: int = 2000,
) -> tuple[ForecastBundle, LedgerView, PlanningEngine]:
    _fits, bundle, _sampler = forecast_at(
        store, as_of, horizon, n_scenarios=n_scenarios
    )
    view = replay(store, as_of)
    engine = PlanningEngine(store, bundle, view)
    return bundle, view, engine


def backtest_at_points(
    store: dataio.DataStore,
    as_of_points: list[datetime],
    horizon_days: tuple[int, ...] = (0, 1, 3, 5),
    n_scenarios: int = 600,
) -> tuple[list[BacktestRow], list[ErrorSummary]]:
    rows = run_backtest(
        store, as_of_points, horizon_days, n_scenarios=n_scenarios
    )
    return rows, summarize(rows)
