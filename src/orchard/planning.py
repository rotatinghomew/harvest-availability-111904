"""排车前的收货承诺引擎。

输入：市场订单（到货窗、等级、缺货容忍度、拼车/换站许可）、园地逐日可采量
情景预测、站点日能力、园地-站点线路，以及已确认承诺台账。

输出区分：
- firm（可承诺）：在扣除既有占量后，订单整体缺货概率 <= 容忍度的方案；
- conditional（有条件候选）：期望上够量但概率约束不过，或依赖封路解除/
  换站/买方同意的方案；
- 同一订单允许拼车时，按“同一天若干园的联合到货”评估，而不是把各园
  置信上限相加。

已批准订单在新预测下出现缺口时，保留原承诺量与发货窗，只给补救建议。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np

from . import io as dataio
from .forecast import ForecastBundle
from .ledger import LedgerView


@dataclass
class AllocationLine:
    plot: str
    hub: str
    day: date
    kg: float
    route_class: str
    basis: str  # plot / pooled / none
    needs_buyer_consent: bool = False
    note: str = ""
    is_existing: bool = False  # 来自不可变台账的既有承诺


@dataclass
class PlanOption:
    request: str
    lines: list[AllocationLine]
    total_kg: float
    shortfall_prob: float
    expected_shortfall_kg: float
    classification: str  # firm / conditional / infeasible
    reasons: list[str] = field(default_factory=list)
    # 该方案落地后对其他订单的挤压：受影响订单与剩余可承诺量
    impact_on_others: dict[str, float] = field(default_factory=dict)
    # 该订单在不可变台账中已经持有的承诺（不重算、不改写）
    existing_lines: list[AllocationLine] = field(default_factory=list)
    existing_kg: float = 0.0
    residual_need_kg: float = 0.0

    def consented_lines(self) -> list[AllocationLine]:
        return [ln for ln in self.lines if ln.needs_buyer_consent]


@dataclass
class GapItem:
    reference: str
    request: str
    plot: str
    hub: str
    promised_day: date
    open_kg: float
    cover_prob: float          # 现有占量被新预测覆盖的概率
    expected_gap_kg: float
    remedies: list[str]
    buyer_consent_required: list[str]
    impact_on_others: dict[str, float] = field(default_factory=dict)


class PlanningEngine:
    def __init__(
        self,
        store: dataio.DataStore,
        bundle: ForecastBundle,
        ledger_view: LedgerView,
    ):
        self.store = store
        self.bundle = bundle
        self.ledger = ledger_view
        self.routes: dict[str, list[dataio.Route]] = {}
        for r in store.routes:
            self.routes.setdefault(r.plot, []).append(r)
        self.hub_cap: dict[tuple[str, date], float] = {
            (h.hub, h.day): h.receiving_kg for h in store.hub_days
        }
        self.basis = {
            row.plot: row.basis for row in bundle.plot_day
        }
        # 已占用果量（同园同日）与站能力（同站同日），来自不可变台账
        self.used_fruit = dict(ledger_view.reserved_fruit)
        self.used_hub = dict(ledger_view.reserved_hub)

    # ---------- 基础可用性 ----------
    def _cell_scenarios(self, plot: str, day: date) -> np.ndarray:
        pi = self.bundle.plot_index[plot]
        di = self.bundle.day_index.get(day)
        if di is None:
            # 预测窗未覆盖的日期没有可用依据，按全零处理并由 basis 解释
            return np.zeros(self.bundle.n_scenarios)
        return self.bundle.harvestable[:, pi, di]

    def free_fruit_scenarios(self, plot: str, day: date) -> np.ndarray:
        """扣除台账已占用后的园地×日自由可采情景。"""
        reserved = self.used_fruit.get((plot, day), 0.0)
        return np.maximum(self._cell_scenarios(plot, day) - reserved, 0.0)

    def free_hub(self, hub: str, day: date) -> float:
        cap = self.hub_cap.get((hub, day))
        if cap is None:
            return 0.0
        return max(cap - self.used_hub.get((hub, day), 0.0), 0.0)

    def eligible_plots(self, req: dataio.Request) -> list[str]:
        return [
            plot
            for plot, meta in self.store.plots.items()
            if meta.primary_grade == req.grade
        ]

    def candidate_lines(
        self, req: dataio.Request
    ) -> list[tuple[str, str, date, str]]:
        """所有 (园, 站, 日, 线路等级) 候选，含可达性与站能力过滤。"""
        out = []
        for day in dataio.iter_dates(req.first_day, req.last_day):
            if day not in self.bundle.day_index:
                continue  # 预测窗未覆盖的日期不产生候选
            for plot in self.eligible_plots(req):
                for route in self.routes.get(plot, []):
                    if self.free_hub(route.hub, day) <= 0:
                        continue
                    out.append((plot, route.hub, day, route.route_class))
        return out

    # ---------- 情景级联合缺货评估 ----------
    def evaluate(
        self,
        req: dataio.Request,
        lines: list[AllocationLine],
        include_existing: bool = False,
    ) -> tuple[float, float]:
        """返回 (缺货概率, 期望缺货量)。同园同日多线合并后按情景联合计算。

        台账既有占量已经计入 used_fruit/used_hub 与自由余额；
        include_existing=True 时把只读的既有承诺也纳入本订单覆盖评估，
        此时站能力只校验新增部分，避免对既有占量重复扣减。
        """
        if not lines:
            return 1.0, req.kg
        existing_refs = set()
        if include_existing:
            existing_refs = {
                (ln.plot, ln.day) for ln in lines if ln.is_existing
            }
        # 按 (园,日) 聚合；新增线的站能力为确定性硬约束，超限直接判失败
        need: dict[tuple[str, date], float] = {}
        hub_new: dict[tuple[str, date], float] = {}
        for ln in lines:
            need[(ln.plot, ln.day)] = need.get((ln.plot, ln.day), 0.0) + ln.kg
            if not ln.is_existing:
                hub_new[(ln.hub, ln.day)] = (
                    hub_new.get((ln.hub, ln.day), 0.0) + ln.kg
                )
        for (hub, day), q in hub_new.items():
            if q > self.free_hub(hub, day) + 1e-9:
                return 1.0, req.kg  # 站能力硬约束不过
        total = np.zeros(self.bundle.n_scenarios)
        for (plot, day), q in need.items():
            free = self.free_fruit_scenarios(plot, day)
            if (plot, day) in existing_refs:
                # 既有承诺占用的果在自由余额之外，加回其在占量
                free = free + self.used_fruit.get((plot, day), 0.0)
            total += np.minimum(free, q)
        shortfall = np.maximum(req.kg - total, 0.0)
        return (
            float(np.mean(shortfall > 1e-9)),
            float(np.mean(shortfall)),
        )

    # ---------- 方案构造 ----------
    def _greedy_lines(
        self, req: dataio.Request, candidates: list[tuple[str, str, date, str]]
    ) -> list[AllocationLine]:
        """按自由可采期望从大到小贪心填充，受站能力与园日余量约束。"""
        ranked = []
        for plot, hub, day, route_class in candidates:
            exp_free = float(np.nanmean(self.free_fruit_scenarios(plot, day)))
            ranked.append((exp_free, plot, hub, day, route_class))
        ranked.sort(reverse=True)
        lines: list[AllocationLine] = []
        remain = req.kg
        hub_left = {
            (h, d): self.free_hub(h, d) for (h, d) in {
                (hub, day) for _, _, hub, day, _ in ranked
            }
        }
        for _exp, plot, hub, day, route_class in ranked:
            if remain <= 1e-9:
                break
            exp_free = float(np.nanmean(self.free_fruit_scenarios(plot, day)))
            if not np.isfinite(exp_free) or exp_free <= 0:
                continue
            q = min(remain, exp_free, hub_left[(hub, day)])
            if q <= 0:
                continue
            lines.append(
                AllocationLine(
                    plot=plot, hub=hub, day=day, kg=round(q, 3),
                    route_class=route_class, basis=self.basis.get(plot, "none"),
                )
            )
            hub_left[(hub, day)] -= q
            remain -= q
        return lines

    def plan_request(
        self, req: dataio.Request, others: list[dataio.Request]
    ) -> PlanOption:
        # 台账中该订单已持有的承诺：只读，不参与重新分配
        existing = self.ledger.open_for_request(req.request)
        existing_lines = [
            AllocationLine(
                plot=item.plot, hub=item.hub, day=item.promised_day,
                kg=item.open_kg,
                route_class=next(
                    (rt.route_class for rt in self.routes.get(item.plot, [])
                     if rt.hub == item.hub),
                    "",
                ),
                basis=self.basis.get(item.plot, "none"),
                note=f"已确认承诺 {item.reference}",
                is_existing=True,
            )
            for item in existing
        ]
        existing_kg = sum(ln.kg for ln in existing_lines)

        candidates = self.candidate_lines(req)
        if not candidates and not existing_lines:
            return PlanOption(
                req.request, [], 0.0, 1.0, req.kg, "infeasible",
                ["窗内无可达园地或站点无剩余能力"],
                existing_lines=existing_lines, existing_kg=existing_kg,
                residual_need_kg=req.kg,
            )
        # 不允许拼车：只保留单园单线候选，逐园评估取最优
        if not req.can_combine:
            best: PlanOption | None = None
            for plot, hub, day, route_class in candidates:
                exp_free = float(np.nanmean(self.free_fruit_scenarios(plot, day)))
                q = min(req.kg - existing_kg, exp_free, self.free_hub(hub, day))
                if q <= 0 or not np.isfinite(q):
                    continue
                line = AllocationLine(
                    plot, hub, day, round(q, 3), route_class,
                    self.basis.get(plot, "none"),
                )
                p, es = self.evaluate(
                    req, existing_lines + [line], include_existing=True
                )
                opt = PlanOption(
                    req.request, [line], existing_kg + round(q, 3), p, es,
                    "firm" if p <= req.shortfall_limit + 1e-12 else "conditional",
                    existing_lines=existing_lines, existing_kg=existing_kg,
                    residual_need_kg=max(req.kg - existing_kg, 0.0),
                )
                if best is None or (es, p) < (best.expected_shortfall_kg,
                                              best.shortfall_prob):
                    best = opt
            if best is None:
                p, es = self.evaluate(
                    req, existing_lines, include_existing=True
                )
                return PlanOption(
                    req.request, existing_lines, existing_kg, p, es,
                    "conditional" if existing_lines else "infeasible",
                    ["合格等级园地在窗内没有新增可采依据或余量"],
                    existing_lines=existing_lines, existing_kg=existing_kg,
                    residual_need_kg=max(req.kg - existing_kg, 0.0),
                )
            option = best
        else:
            residual_req = dataio.Request(
                request=req.request, market=req.market,
                first_day=req.first_day, last_day=req.last_day,
                grade=req.grade, kg=max(req.kg - existing_kg, 0.0),
                shortfall_limit=req.shortfall_limit,
                can_combine=req.can_combine,
                can_switch_hub=req.can_switch_hub,
            )
            lines = (
                self._greedy_lines(residual_req, candidates)
                if residual_req.kg > 0
                else []
            )
            new_kg = sum(ln.kg for ln in lines)
            p, es = self.evaluate(
                req, existing_lines + lines, include_existing=True
            )
            reasons: list[str] = []
            if existing_kg + new_kg < req.kg - 1e-9:
                classification = "conditional"
                reasons.append(
                    f"已承诺 {existing_kg:.0f}kg、新增可组织 {new_kg:.0f}kg，"
                    f"距订单 {req.kg:.0f}kg 的差额需新货源"
                )
            else:
                classification = (
                    "firm"
                    if p <= req.shortfall_limit + 1e-12
                    else "conditional"
                )
                if classification == "conditional":
                    reasons.append(
                        f"联合缺货概率 {p:.1%} 高于容忍度 "
                        f"{req.shortfall_limit:.1%}（一场雨同区园可能同时落空）"
                    )
            option = PlanOption(
                req.request, lines, existing_kg + new_kg, p, es,
                classification, reasons,
                existing_lines=existing_lines, existing_kg=existing_kg,
                residual_need_kg=max(req.kg - existing_kg, 0.0),
            )

        # 依据标注：薄样/无依据的线不能算 firm
        weak = [ln for ln in (option.lines + existing_lines)
                if ln.basis != "plot"]
        if weak and option.classification == "firm":
            option.classification = "conditional"
            option.reasons.append(
                "含薄样/池化依据园地 "
                + ",".join(sorted({ln.plot for ln in weak}))
                + "，其误差单独分层，不提供本园级保证"
            )
        # 无依据园地的新增线需要买方同意
        for ln in option.lines:
            if ln.basis == "none":
                ln.needs_buyer_consent = True
                ln.note = "园地无可用样方依据"

        option.impact_on_others = self._impact(req, option.lines, others)
        return option

    def plan_all(
        self, requests: list[dataio.Request]
    ) -> dict[str, PlanOption]:
        """订单组合顺序排产：每确认一个方案即模拟占用，再排下一个。

        只在引擎内存中推进，不写入不可变台账；真实确认仍需外部事件。
        """
        snapshot_fruit = dict(self.used_fruit)
        snapshot_hub = dict(self.used_hub)
        results: dict[str, PlanOption] = {}
        try:
            for req in requests:
                others = [r for r in requests if r.request != req.request]
                option = self.plan_request(req, others)
                results[req.request] = option
                for ln in option.lines:
                    self.used_fruit[(ln.plot, ln.day)] = (
                        self.used_fruit.get((ln.plot, ln.day), 0.0) + ln.kg
                    )
                    self.used_hub[(ln.hub, ln.day)] = (
                        self.used_hub.get((ln.hub, ln.day), 0.0) + ln.kg
                    )
        finally:
            # 恢复初始占量，保证引擎可重复调用
            self.used_fruit = snapshot_fruit
            self.used_hub = snapshot_hub
        return results

    def _impact(
        self,
        req: dataio.Request,
        lines: list[AllocationLine],
        others: list[dataio.Request],
    ) -> dict[str, float]:
        """模拟占用后，其他订单在其窗内的期望可承诺余量。"""
        out: dict[str, float] = {}
        add_fruit: dict[tuple[str, date], float] = {}
        add_hub: dict[tuple[str, date], float] = {}
        for ln in lines:
            add_fruit[(ln.plot, ln.day)] = (
                add_fruit.get((ln.plot, ln.day), 0.0) + ln.kg
            )
            add_hub[(ln.hub, ln.day)] = (
                add_hub.get((ln.hub, ln.day), 0.0) + ln.kg
            )
        for other in others:
            if other.request == req.request:
                continue
            exp_total = 0.0
            for day in dataio.iter_dates(other.first_day, other.last_day):
                for plot in self.eligible_plots(other):
                    col = self._cell_scenarios(plot, day)
                    reserved = (
                        self.used_fruit.get((plot, day), 0.0)
                        + add_fruit.get((plot, day), 0.0)
                    )
                    exp_total += float(
                        np.nanmean(np.maximum(col - reserved, 0.0))
                    )
            out[other.request] = round(min(exp_total, other.kg), 3)
        return out

    # ---------- 已承诺订单的缺口与补救 ----------
    def gap_analysis(self) -> list[GapItem]:
        gaps: list[GapItem] = []
        for ref, item in self.ledger.commitments.items():
            if item.is_closed:
                continue
            day = item.promised_day
            if day not in self.bundle.day_index:
                continue
            free = self.free_fruit_scenarios(item.plot, day)
            cover = np.minimum(free, item.open_kg)
            gap = np.maximum(item.open_kg - cover, 0.0)
            cover_prob = float(np.mean(gap <= 1e-9))
            expected_gap = float(np.mean(gap))
            if cover_prob >= 1.0 - 1e-9 and expected_gap <= 1e-6:
                continue
            remedies, consents = self._remedies(item, expected_gap)
            gaps.append(
                GapItem(
                    reference=ref,
                    request=item.request,
                    plot=item.plot,
                    hub=item.hub,
                    promised_day=day,
                    open_kg=item.open_kg,
                    cover_prob=cover_prob,
                    expected_gap_kg=round(expected_gap, 3),
                    remedies=remedies,
                    buyer_consent_required=consents,
                )
            )
        return gaps

    def _remedies(
        self, item, gap_kg: float
    ) -> tuple[list[str], list[str]]:
        req = next(
            r for r in self.store.requests if r.request == item.request
        )
        remedies: list[str] = []
        consents: list[str] = []
        need = max(gap_kg, 0.0)
        # 1) 同等级其他园、同日、同站可达
        alt_plots = [
            p
            for p in self.eligible_plots(req)
            if p != item.plot
            and any(rt.hub == item.hub for rt in self.routes.get(p, []))
        ]
        found = 0.0
        for plot in alt_plots:
            free = float(np.nanmean(self.free_fruit_scenarios(plot, item.promised_day)))
            take = min(free, need - found)
            if take > 0:
                found += take
                remedies.append(
                    f"同日由 {plot} 补 {take:.0f}kg 至 {item.hub}（同等级）"
                )
        # 2) 换站（需订单允许且买方同意）
        if found < need - 1e-9 and req.can_switch_hub:
            for route in self.routes.get(item.plot, []):
                if route.hub == item.hub:
                    continue
                cap = self.free_hub(route.hub, item.promised_day)
                if cap > 0:
                    take = min(cap, need - found)
                    remedies.append(
                        f"原园 {item.plot} 改送 {route.hub} {take:.0f}kg"
                    )
                    consents.append(
                        f"换站 {item.hub}->{route.hub} 需买方同意"
                    )
                    found += take
        # 3) 窗内顺延一天（保留发货窗精神，需买方同意）
        if found < need - 1e-9:
            nxt = date.fromordinal(item.promised_day.toordinal() + 1)
            if nxt <= req.last_day:
                free = float(np.nanmean(self.free_fruit_scenarios(item.plot, nxt)))
                take = min(free, need - found)
                if take > 0:
                    remedies.append(
                        f"{item.plot} 顺延至 {nxt} 补 {take:.0f}kg"
                    )
                    consents.append("调整发货日需买方同意")
                    found += take
        if found < need - 1e-9:
            remedies.append(
                f"仍有 {need - found:.0f}kg 无可用补救来源，需主动协商缺货"
            )
        return remedies, consents

    # ---------- 调整 vs 保持原计划的风险对比 ----------
    def compare_plans(
        self,
        req: dataio.Request,
        adjusted: PlanOption,
        baseline_lines: list[AllocationLine] | None = None,
    ) -> dict:
        if baseline_lines is None:
            baseline_lines = [
                AllocationLine(
                    plot=item.plot, hub=item.hub, day=item.promised_day,
                    kg=item.open_kg, route_class="",
                    basis=self.basis.get(item.plot, "none"),
                    is_existing=True,
                )
                for item in self.ledger.open_for_request(req.request)
            ]
        bp, be = self.evaluate(
            req, baseline_lines, include_existing=True
        )
        adjusted_all = adjusted.existing_lines + adjusted.lines
        ap, ae = self.evaluate(
            req, adjusted_all, include_existing=True
        )
        return {
            "request": req.request,
            "keep": {
                "shortfall_prob": round(bp, 4),
                "expected_shortfall_kg": round(be, 3),
                "lines": [
                    f"{ln.plot}@{ln.hub} {ln.day} {ln.kg:.0f}kg"
                    for ln in baseline_lines
                ],
            },
            "adjust": {
                "shortfall_prob": round(ap, 4),
                "expected_shortfall_kg": round(ae, 3),
                "classification": adjusted.classification,
                "lines": [
                    f"{ln.plot}@{ln.hub} {ln.day} {ln.kg:.0f}kg"
                    + ("（既有承诺）" if ln.is_existing else "")
                    for ln in adjusted_all
                ],
                "reasons": adjusted.reasons,
            },
        }
