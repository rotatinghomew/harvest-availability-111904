"""原始记录载入与时点快照。

所有业务时间都带 UTC 偏移；“可见性”以记录自身的 reported_at / known_at /
issued_at / occurred_at 与给定 as_of 比较，而不是与业务日比较。
空值保持 None，绝不按零处理。
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Iterable


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _parse_day(value: str) -> date:
    return date.fromisoformat(value)


def _num(value: str) -> float:
    return float(value) if value not in ("", None) else float("nan")


@dataclass(frozen=True)
class Plot:
    plot: str
    district: str
    rain_zone: str
    bearing_trees: int
    primary_grade: str


@dataclass(frozen=True)
class Sample:
    plot: str
    day: date
    sampled_trees: int
    mass_tree_kg: float
    mature_kg: float  # 每样株在树成熟重量（kg/株）
    reported_at: datetime


@dataclass(frozen=True)
class Picking:
    plot: str
    day: date
    harvested_kg: float
    direct_kg: float
    received_kg: float
    limit_reason: str
    known_at: datetime


@dataclass(frozen=True)
class Forecast:
    rain_zone: str
    target_day: date
    issued_at: datetime
    rain_mm: float
    low_c: float


@dataclass(frozen=True)
class RoadLog:
    rain_zone: str
    service_day: date
    passable: bool
    known_at: datetime
    note: str


@dataclass(frozen=True)
class Route:
    plot: str
    hub: str
    route_class: str


@dataclass(frozen=True)
class HubDay:
    hub: str
    day: date
    receiving_kg: float


@dataclass(frozen=True)
class Request:
    request: str
    market: str
    first_day: date
    last_day: date
    grade: str
    kg: float
    shortfall_limit: float
    can_combine: bool
    can_switch_hub: bool


@dataclass(frozen=True)
class Commitment:
    reference: str
    request: str
    plot: str
    hub: str
    promised_day: date
    quantity_kg: float
    status: str  # 确认 / 交货 / 取消
    occurred_at: datetime


@dataclass
class DataStore:
    plots: dict[str, Plot]
    samples: list[Sample]
    picking: list[Picking]
    forecasts: list[Forecast]
    road_log: list[RoadLog]
    routes: list[Route]
    hub_days: list[HubDay]
    requests: list[Request]
    commitments: list[Commitment]
    # 便于查询的索引
    samples_by_plot: dict[str, list[Sample]] = field(default_factory=dict)
    picking_by_plot: dict[str, list[Picking]] = field(default_factory=dict)

    def season_days(self) -> list[date]:
        days = {p.day for p in self.picking} | {s.day for s in self.samples}
        return sorted(days)


def load_store(data_dir: str | Path) -> DataStore:
    data_dir = Path(data_dir)
    plots: dict[str, Plot] = {}
    with (data_dir / "plots.csv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            plots[row["plot"]] = Plot(
                plot=row["plot"],
                district=row["district"],
                rain_zone=row["rain_zone"],
                bearing_trees=int(row["bearing_trees"]),
                primary_grade=row["primary_grade"],
            )

    samples: list[Sample] = []
    with (data_dir / "samples.csv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            samples.append(
                Sample(
                    plot=row["plot"],
                    day=_parse_day(row["day"]),
                    sampled_trees=int(row["sampled_trees"]),
                    mass_tree_kg=_num(row["mass_tree_kg"]),
                    mature_kg=_num(row["mature_kg"]),
                    reported_at=_parse_dt(row["reported_at"]),
                )
            )

    picking: list[Picking] = []
    with (data_dir / "picking.csv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            picking.append(
                Picking(
                    plot=row["plot"],
                    day=_parse_day(row["day"]),
                    harvested_kg=_num(row["harvested_kg"]),
                    direct_kg=_num(row["direct_kg"]),
                    received_kg=_num(row["received_kg"]),
                    limit_reason=row["limit_reason"],
                    known_at=_parse_dt(row["known_at"]),
                )
            )

    forecasts: list[Forecast] = []
    with (data_dir / "forecasts.csv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            forecasts.append(
                Forecast(
                    rain_zone=row["rain_zone"],
                    target_day=_parse_day(row["target_day"]),
                    issued_at=_parse_dt(row["issued_at"]),
                    rain_mm=_num(row["rain_mm"]),
                    low_c=_num(row["low_c"]),
                )
            )

    road_log: list[RoadLog] = []
    with (data_dir / "road_log.csv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            road_log.append(
                RoadLog(
                    rain_zone=row["rain_zone"],
                    service_day=_parse_day(row["service_day"]),
                    passable=row["passable"] == "1",
                    known_at=_parse_dt(row["known_at"]),
                    note=row.get("note", "") or "",
                )
            )

    routes: list[Route] = []
    with (data_dir / "route_capacity.csv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            routes.append(Route(row["plot"], row["hub"], row["route_class"]))

    hub_days: list[HubDay] = []
    with (data_dir / "hub_days.csv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            hub_days.append(
                HubDay(row["hub"], _parse_day(row["day"]), _num(row["receiving_kg"]))
            )

    requests: list[Request] = []
    with (data_dir / "requests.csv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            requests.append(
                Request(
                    request=row["request"],
                    market=row["market"],
                    first_day=_parse_day(row["first_day"]),
                    last_day=_parse_day(row["last_day"]),
                    grade=row["grade"],
                    kg=_num(row["kg"]),
                    shortfall_limit=_num(row["shortfall_limit"]),
                    can_combine=row["can_combine"] == "1",
                    can_switch_hub=row["can_switch_hub"] == "1",
                )
            )

    commitments: list[Commitment] = []
    with (data_dir / "commitments.csv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            commitments.append(
                Commitment(
                    reference=row["reference"],
                    request=row["request"],
                    plot=row["plot"],
                    hub=row["hub"],
                    promised_day=_parse_day(row["promised_day"]),
                    quantity_kg=_num(row["quantity_kg"]),
                    status=row["status"],
                    occurred_at=_parse_dt(row["occurred_at"]),
                )
            )

    samples_by_plot: dict[str, list[Sample]] = {}
    for s in samples:
        samples_by_plot.setdefault(s.plot, []).append(s)
    for lst in samples_by_plot.values():
        lst.sort(key=lambda s: s.day)
    picking_by_plot: dict[str, list[Picking]] = {}
    for p in picking:
        picking_by_plot.setdefault(p.plot, []).append(p)
    for lst in picking_by_plot.values():
        lst.sort(key=lambda p: p.day)

    return DataStore(
        plots=plots,
        samples=samples,
        picking=picking,
        forecasts=forecasts,
        road_log=road_log,
        routes=routes,
        hub_days=hub_days,
        requests=requests,
        commitments=commitments,
        samples_by_plot=samples_by_plot,
        picking_by_plot=picking_by_plot,
    )


def visible_samples(store: DataStore, plot: str, as_of: datetime) -> list[Sample]:
    """as_of 时点已回报的样方（reported_at <= as_of）。"""
    return [s for s in store.samples_by_plot.get(plot, []) if s.reported_at <= as_of]


def visible_picking(store: DataStore, plot: str, as_of: datetime) -> list[Picking]:
    """as_of 时点已传回的采收记录（known_at <= as_of）。"""
    return [p for p in store.picking_by_plot.get(plot, []) if p.known_at <= as_of]


def visible_road(store: DataStore, as_of: datetime) -> dict[tuple[str, date], bool]:
    return {
        (r.rain_zone, r.service_day): r.passable
        for r in store.road_log
        if r.known_at <= as_of
    }


def latest_forecasts(
    store: DataStore, as_of: datetime
) -> dict[tuple[str, date], Forecast]:
    """每个 (雨区, 目标日) 取 as_of 前最新发布版本。"""
    latest: dict[tuple[str, date], Forecast] = {}
    for f in store.forecasts:
        if f.issued_at > as_of:
            continue
        key = (f.rain_zone, f.target_day)
        cur = latest.get(key)
        if cur is None or f.issued_at > cur.issued_at:
            latest[key] = f
    return latest


def forecast_versions(
    store: DataStore, rain_zone: str, target_day: date, as_of: datetime
) -> list[Forecast]:
    out = [
        f
        for f in store.forecasts
        if f.rain_zone == rain_zone
        and f.target_day == target_day
        and f.issued_at <= as_of
    ]
    out.sort(key=lambda f: f.issued_at)
    return out


def season_of(day: date) -> int:
    """产季按业务年份（资料均为 11 月采收季）。"""
    return day.year


def iter_dates(first: date, last: date) -> Iterable[date]:
    d = first
    while d <= last:
        yield d
        d = date.fromordinal(d.toordinal() + 1)
