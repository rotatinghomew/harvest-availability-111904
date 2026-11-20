"""命令行入口：预测、承诺、缺口、回测、报表。

用法示例：
  python -m orchard.cli forecast --as-of 2026-11-18T10:00:00+08:00 \
      --first 2026-11-18 --last 2026-11-21 --out reports/forecast.csv
  python -m orchard.cli plan --as-of 2026-11-18T10:00:00+08:00
  python -m orchard.cli gaps --as-of 2026-11-18T10:00:00+08:00
  python -m orchard.cli backtest
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import backtest as bt
from . import io as dataio
from .ledger import replay

_BJ = timezone(timedelta(hours=8))


def _store(args) -> dataio.DataStore:
    return dataio.load_store(args.data)


def _horizon(first: str | None, last: str | None, default_days: int = 4):
    if first is None:
        d0 = date.today()
    else:
        d0 = date.fromisoformat(first)
    d1 = date.fromisoformat(last) if last else d0 + timedelta(days=default_days - 1)
    return list(dataio.iter_dates(d0, d1))


def cmd_forecast(args) -> None:
    store = _store(args)
    as_of = datetime.fromisoformat(args.as_of)
    horizon = _horizon(args.first, args.last)
    fits, bundle, _ = bt.forecast_at(
        store, as_of, horizon, n_scenarios=args.scenarios
    )
    out = Path(args.out) if args.out else None
    fh = out.open("w", encoding="utf-8", newline="") if out else sys.stdout
    writer = csv.writer(fh)
    writer.writerow(
        [
            "plot", "district", "rain_zone", "grade", "day", "basis",
            "mean_kg", "q05", "q25", "q50", "q75", "q95", "closure_prob",
            "e_ripe_start_kg", "e_new_ripening_kg", "e_overripen_loss_kg",
            "e_unripe_end_kg", "known_harvested_cum_kg",
        ]
    )
    for r in bundle.plot_day:
        writer.writerow(
            [
                r.plot, r.district, r.rain_zone, r.grade, r.day.isoformat(),
                r.basis,
                _f(r.mean_kg), _f(r.q05), _f(r.q25), _f(r.q50), _f(r.q75),
                _f(r.q95), round(r.closure_prob, 4),
                _f(r.e_ripe_start), _f(r.e_new_ripening),
                _f(r.e_overripen_loss), _f(r.e_unripe_end),
                round(r.known_harvested_cum, 2),
            ]
        )
    writer.writerow([])
    writer.writerow(["scope", "day", "mean_kg", "q05", "q50", "q95"])
    for r in bundle.region_day:
        writer.writerow(
            [r.scope, r.day.isoformat(), _f(r.mean_kg), _f(r.q05),
             _f(r.q50), _f(r.q95)]
        )
    if out:
        fh.close()
        print(f"预测报表写入 {out}；质量守恒最大偏差 {bundle.mass_gap_max:.4f} kg")
    print_basis_warnings(fits)


def cmd_plan(args) -> None:
    store = _store(args)
    as_of = datetime.fromisoformat(args.as_of)
    horizon = _horizon(args.first, args.last)
    _bundle, view, engine = bt.planning_at(
        store, as_of, horizon, n_scenarios=args.scenarios
    )
    print(f"截至 {as_of.isoformat()} 的台账占量：")
    for ref, item in sorted(view.commitments.items()):
        print(
            f"  {ref} {item.request} {item.plot}->{item.hub} "
            f"{item.promised_day} 确认{item.confirmed_kg:.0f} "
            f"已交{item.delivered_kg:.0f} 取消{item.cancelled_kg:.0f} "
            f"在占{item.open_kg:.0f}"
        )
    print()
    options = engine.plan_all(store.requests)
    for req in store.requests:
        option = options[req.request]
        print(
            f"订单 {req.request}（{req.market} {req.grade} {req.kg:.0f}kg "
            f"窗 {req.first_day}~{req.last_day} 容忍 {req.shortfall_limit:.0%}）"
        )
        print(f"  结论: {option.classification}")
        if option.existing_lines:
            print(f"  已持有承诺 {option.existing_kg:.0f}kg（不可改写）:")
            for ln in option.existing_lines:
                print(
                    f"    - {ln.plot}({ln.basis}) -> {ln.hub} {ln.day} "
                    f"{ln.kg:.0f}kg {ln.route_class} {ln.note}"
                )
        if option.lines:
            print(
                f"  本次新增候选 {sum(ln.kg for ln in option.lines):.0f}kg:"
            )
            for ln in option.lines:
                flag = " [需买方同意]" if ln.needs_buyer_consent else ""
                print(
                    f"    - {ln.plot}({ln.basis}) -> {ln.hub} {ln.day} "
                    f"{ln.kg:.0f}kg {ln.route_class}{flag} {ln.note}"
                )
        print(
            f"  合计覆盖 {option.total_kg:.0f}/{req.kg:.0f}kg，"
            f"联合缺货概率 {option.shortfall_prob:.2%}，"
            f"期望缺货 {option.expected_shortfall_kg:.1f}kg"
        )
        for reason in option.reasons:
            print(f"  说明: {reason}")
        if option.impact_on_others:
            print("  对其他订单可承诺余量影响:")
            for k, v in option.impact_on_others.items():
                print(f"    {k}: 剩余可承诺期望 {v:.0f}kg")
        print()


def cmd_gaps(args) -> None:
    store = _store(args)
    as_of = datetime.fromisoformat(args.as_of)
    horizon = _horizon(args.first, args.last)
    _bundle, _view, engine = bt.planning_at(
        store, as_of, horizon, n_scenarios=args.scenarios
    )
    gaps = engine.gap_analysis()
    if not gaps:
        print("已承诺订单在当前预测下全部可覆盖，无缺口。")
        return
    for g in gaps:
        print(
            f"缺口 {g.reference}（{g.request} {g.plot}->{g.hub} "
            f"{g.promised_day} 在占 {g.open_kg:.0f}kg）"
        )
        print(
            f"  覆盖概率 {g.cover_prob:.2%}，期望缺口 {g.expected_gap_kg:.1f}kg"
        )
        for remedy in g.remedies:
            print(f"  补救: {remedy}")
        for c in g.buyer_consent_required:
            print(f"  需买方同意: {c}")


def cmd_compare(args) -> None:
    store = _store(args)
    as_of = datetime.fromisoformat(args.as_of)
    horizon = _horizon(args.first, args.last)
    _bundle, _view, engine = bt.planning_at(
        store, as_of, horizon, n_scenarios=args.scenarios
    )
    req = next(r for r in store.requests if r.request == args.request)
    others = [r for r in store.requests if r.request != req.request]
    adjusted = engine.plan_request(req, others)
    result = engine.compare_plans(req, adjusted)
    print(f"订单 {args.request} 保持原计划 vs 调整计划：")
    for label in ("keep", "adjust"):
        part = result[label]
        print(f"  [{label}] 缺货概率 {part['shortfall_prob']:.2%}，"
              f"期望缺货 {part['expected_shortfall_kg']:.1f}kg")
        for line in part["lines"]:
            print(f"      {line}")
    for reason in result["adjust"].get("reasons", []):
        print(f"  调整说明: {reason}")


def cmd_backtest(args) -> None:
    store = _store(args)
    # 每个产季内、样方回报后的若干下单前时点（10:00，当日 07:00 路况已可见）
    points: list[datetime] = []
    season_years = sorted({s.day.year for s in store.samples})
    for year in season_years:
        for offset in (6, 9, 12, 15, 18):
            points.append(
                datetime(year, 11, 1, 10, 0, tzinfo=_BJ)
                + timedelta(days=offset)
            )
    points = [p for p in points if p.date() <= max(r.day for r in store.picking)]
    rows, summaries = bt.backtest_at_points(
        store, points, horizon_days=(0, 1, 3, 5), n_scenarios=args.scenarios
    )
    out = Path(args.out) if args.out else None
    fh = out.open("w", encoding="utf-8", newline="") if out else sys.stdout
    writer = csv.writer(fh)
    writer.writerow(["layer", "value", "n", "mae_kg", "mape", "bias_kg",
                     "coverage_90", "thin_sample"])
    for s in summaries:
        writer.writerow(
            [s.layer, s.value, s.n, _f(s.mae), _f(s.mape), _f(s.bias),
             _f(s.coverage_90), s.thin_sample]
        )
    if out:
        fh.close()
        print(f"分层误差报表写入 {out}（{len(rows)} 条回测单元）")
    else:
        print(f"\n共 {len(rows)} 条回测单元")


def cmd_ledger(args) -> None:
    store = _store(args)
    as_of = datetime.fromisoformat(args.as_of)
    view = replay(store, as_of)
    print(f"台账重放 @ {as_of.isoformat()}（事件不可变，仅过滤 occurred_at）")
    for ref, item in sorted(view.commitments.items()):
        print(
            f"  {ref} {item.request} {item.plot}->{item.hub} {item.promised_day} "
            f"确认{item.confirmed_kg:.0f} 已交{item.delivered_kg:.0f} "
            f"取消{item.cancelled_kg:.0f} 在占{item.open_kg:.0f} "
            f"{'[已结清]' if item.is_closed else ''}"
        )


def print_basis_warnings(fits) -> None:
    for plot, fit in fits.items():
        if fit.basis == "pooled":
            print(
                f"[依据警告] {plot} 仅 {fit.points} 个样方，借用 "
                f"{','.join(fit.pooled_from)} 的池化曲线，误差单独分层。",
                file=sys.stderr,
            )
        elif fit.basis == "none":
            print(
                f"[依据缺失] {plot} 无任何可见样方，不出具数值预测。",
                file=sys.stderr,
            )


def _f(x) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return ""
    return f"{x:.2f}"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="orchard", description="可采量预测与收货承诺")
    p.add_argument("--data", default=str(Path(__file__).resolve().parents[2] / "data"))
    p.add_argument("--scenarios", type=int, default=2000)
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("forecast")
    f.add_argument("--as-of", required=True)
    f.add_argument("--first")
    f.add_argument("--last")
    f.add_argument("--out")
    f.set_defaults(func=cmd_forecast)

    pl = sub.add_parser("plan")
    pl.add_argument("--as-of", required=True)
    pl.add_argument("--first")
    pl.add_argument("--last")
    pl.set_defaults(func=cmd_plan)

    g = sub.add_parser("gaps")
    g.add_argument("--as-of", required=True)
    g.add_argument("--first")
    g.add_argument("--last")
    g.set_defaults(func=cmd_gaps)

    c = sub.add_parser("compare")
    c.add_argument("--as-of", required=True)
    c.add_argument("--request", required=True)
    c.add_argument("--first")
    c.add_argument("--last")
    c.set_defaults(func=cmd_compare)

    b = sub.add_parser("backtest")
    b.add_argument("--out")
    b.set_defaults(func=cmd_backtest)

    l = sub.add_parser("ledger")
    l.add_argument("--as-of", required=True)
    l.set_defaults(func=cmd_ledger)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
