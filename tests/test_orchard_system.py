"""可采量预测与收货承诺的不变量测试。

覆盖用户需求中的硬约束：
1. 送站重量不等于产量，封路/车满压低送站不进入可采量；
2. 未熟/可采未采/已采三分量与质量守恒，同果不多日累计；
3. 历史重放只看当时可见的样方与预报版本；
4. 分层误差可按产季/记录量/道路状态/区域查，薄样不冒充区级保证；
5. 区域区间来自情景联合分布，不是逐园分位数相加；
6. 承诺区分 firm/conditional，拼车按联合风险，站能力与占量硬约束；
7. 确认占用、交货/取消释放，已摘果不回池；新预测不改写司机单子；
8. 无依据园地显式标注，不出数值承诺。
"""
import math
import unittest
from datetime import date, datetime, timedelta, timezone

import numpy as np

from orchard import backtest as bt
from orchard import io as dataio
from orchard import ripening as rip
from orchard.forecast import build_forecast
from orchard.ledger import OpenCommitment, LedgerView, replay
from orchard.planning import AllocationLine, PlanningEngine
from orchard.weather import (
    WeatherSampler,
    calibrate_closure,
    calibrate_rain_errors,
)

BJ = timezone(timedelta(hours=8))
DATA = "/workspace/data"


def dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.store = dataio.load_store(DATA)

    def test_loads_all_tables(self):
        self.assertEqual(len(self.store.plots), 7)
        self.assertTrue(self.store.samples)
        self.assertTrue(self.store.commitments)

    def test_visibility_is_by_report_time_not_business_day(self):
        # 2026-11-06 的样方 11-07 才回报：11-06 12:00 不可见，11-07 12:00 可见
        as_of_early = dt("2026-11-06T12:00:00+08:00")
        as_of_late = dt("2026-11-07T12:00:00+08:00")
        early = dataio.visible_samples(self.store, "E2", as_of_early)
        late = dataio.visible_samples(self.store, "E2", as_of_late)
        self.assertNotIn(date(2026, 11, 6), {s.day for s in early})
        self.assertIn(date(2026, 11, 6), {s.day for s in late})

    def test_latest_forecast_uses_only_issued_versions(self):
        as_of = dt("2026-11-17T12:00:00+08:00")
        latest = dataio.latest_forecasts(self.store, as_of)
        self.assertTrue(all(f.issued_at <= as_of for f in latest.values()))
        key = ("沙坪", date(2026, 11, 18))
        self.assertEqual(latest[key].issued_at.date(), date(2026, 11, 17))
        # 11-18 05:15 的版本在该时点尚不存在
        later = [
            f
            for f in self.store.forecasts
            if (f.rain_zone, f.target_day) == key and f.issued_at > as_of
        ]
        self.assertTrue(later)

    def test_picking_known_at_next_day(self):
        as_of = dt("2026-11-17T20:00:00+08:00")
        visible = dataio.visible_picking(self.store, "E1", as_of)
        self.assertNotIn(date(2026, 11, 17), {p.day for p in visible})


class RipeningTests(unittest.TestCase):
    def setUp(self):
        self.store = dataio.load_store(DATA)
        self.as_of = dt("2026-11-18T10:00:00+08:00")

    def test_season_scoped_fit_sigmoid_shape(self):
        fit = rip.fit_plot(self.store, "E1", self.as_of)
        self.assertEqual(fit.basis, "plot")
        k, amp, t0 = fit.params
        # t0 必须在产季日轴（11 月内约 0..30），不能是绝对日序数
        self.assertLess(t0, 60)
        curve = fit.cumulative_mature(np.array([0.0, 10.0, 20.0, 40.0]))
        self.assertTrue(np.all(np.diff(curve) >= -1e-9))

    def test_thin_sample_pooled_and_flagged(self):
        as_of = dt("2025-11-16T10:00:00+08:00")
        fit = rip.fit_plot(self.store, "W4", as_of)
        self.assertEqual(fit.basis, "pooled")
        self.assertIn("W1", fit.pooled_from)
        # 池化残差主动放大
        donor = rip.fit_plot(self.store, "W1", as_of)
        self.assertGreaterEqual(fit.residual_sd, donor.residual_sd)

    def test_no_basis_when_no_samples(self):
        as_of = dt("2024-11-10T10:00:00+08:00")
        fit = rip.fit_plot(self.store, "W4", as_of)
        self.assertEqual(fit.basis, "none")
        self.assertIsNone(fit.params)

    def test_cross_district_pooling_for_orphan_grade(self):
        # E3 是河东唯一 B 级园，薄样时只能跨区借 W3
        as_of = dt("2024-11-07T10:00:00+08:00")
        fit = rip.fit_plot(self.store, "E3", as_of)
        if fit.basis == "pooled":
            self.assertTrue(fit.pooled_cross_district)
            self.assertIn("W3", fit.pooled_from)


class ForecastInvariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = dataio.load_store(DATA)
        cls.as_of = dt("2026-11-18T10:00:00+08:00")
        cls.horizon = [
            date(2026, 11, 18) + timedelta(days=i) for i in range(4)
        ]
        cls.fits, cls.bundle, _ = bt.forecast_at(
            cls.store, cls.as_of, cls.horizon, n_scenarios=600
        )

    def test_mass_conservation_at_anchor(self):
        # 锚定日恒等式偏差应只有数值误差
        self.assertLess(self.bundle.mass_gap_max, 1e-4)

    def test_three_buckets_present(self):
        cell = self.bundle.cell("E1", date(2026, 11, 19))
        self.assertGreater(cell.e_ripe_start, 0)
        self.assertGreater(cell.e_new_ripening, 0)
        self.assertGreaterEqual(cell.e_unripe_end, 0)
        self.assertGreater(cell.known_harvested_cum, 0)

    def test_closure_zeroes_harvest_but_carries_stock(self):
        # 双湾 11-18 已知封闭
        w3_18 = self.bundle.cell("W3", date(2026, 11, 18))
        self.assertEqual(w3_18.closure_prob, 1.0)
        self.assertEqual(w3_18.mean_kg, 0.0)
        # 11-19 的期初在树量仍接近封闭前存量（结转，未消失）
        w3_19 = self.bundle.cell("W3", date(2026, 11, 19))
        self.assertGreater(w3_19.e_ripe_start, w3_18.e_ripe_start * 0.9)

    def test_no_double_count_across_days(self):
        # 每日新成熟量之和不超过曲线总果实量
        for plot in self.store.plots:
            pi = self.bundle.plot_index[plot]
            trees = self.store.plots[plot].bearing_trees
            total_amp = self.fits[plot].params[1] * trees
            new_sum = sum(
                self.bundle.cell(plot, d).e_new_ripening for d in self.horizon
            )
            self.assertLessEqual(new_sum, total_amp + 1.0)

    def test_region_quantiles_are_joint_not_sum_of_marginals(self):
        day = date(2026, 11, 21)
        di = self.bundle.day_index[day]
        marginal_q05 = sum(
            np.quantile(self.bundle.harvestable[:, i, di], 0.05)
            for i in range(len(self.store.plots))
        )
        joint_row = next(
            r for r in self.bundle.region_day
            if r.scope == "ALL" and r.day == day
        )
        # 联合分位与逐园分位之和必须不同（联合保留同雨共变）
        self.assertNotAlmostEqual(
            joint_row.q05, marginal_q05, delta=abs(marginal_q05) * 0.001 + 1
        )

    def test_same_zone_plots_close_together(self):
        di = self.bundle.day_index[date(2026, 11, 21)]
        e2 = self.bundle.closed[:, self.bundle.plot_index["E2"], di]
        e3 = self.bundle.closed[:, self.bundle.plot_index["E3"], di]
        # 石塘同区：联合封闭概率等于单园封闭概率（完全共变），
        # 而独立情形的联合概率只是边际乘积——两者必须显著不同
        self.assertAlmostEqual((e2 & e3).mean(), e2.mean(), places=6)
        self.assertGreater(
            e2.mean(), e2.mean() * e3.mean() + 0.05
        )

    def test_no_basis_cell_is_nan_not_zero(self):
        store = dataio.load_store(DATA)
        as_of = dt("2024-11-08T12:00:00+08:00")
        horizon = [date(2024, 11, 9)]
        _f, bundle, _ = bt.forecast_at(store, as_of, horizon, n_scenarios=200)
        cell = bundle.cell("W4", date(2024, 11, 9))
        self.assertEqual(cell.basis, "none")
        self.assertTrue(math.isnan(cell.mean_kg))

    def test_received_kg_not_used_as_capacity(self):
        # 车满日 received < harvested：预测可采量不得受送站缺口压低
        car_full = [
            p
            for p in self.store.picking
            if p.limit_reason == "车满" and p.received_kg < p.harvested_kg
        ]
        self.assertTrue(car_full)  # 资料中确实存在该现象
        # 模型输入只依赖 harvested 与样方，不依赖 received
        import inspect
        src = inspect.getsource(build_forecast)
        self.assertNotIn("received_kg", src)


class WeatherTests(unittest.TestCase):
    def test_closure_threshold_calibration(self):
        store = dataio.load_store(DATA)
        as_of = dt("2026-11-18T10:00:00+08:00")
        model = calibrate_closure(store, as_of)
        self.assertLess(model.closure_prob(5.0), 0.05)
        self.assertGreater(model.closure_prob(45.0), 0.95)

    def test_rain_error_by_lead_time(self):
        store = dataio.load_store(DATA)
        as_of = dt("2026-11-18T10:00:00+08:00")
        rain = calibrate_rain_errors(store, as_of)
        self.assertTrue(rain.lead_stats)
        # 更长预见期离散度不低于临近预报
        leads = sorted(rain.lead_stats)
        self.assertGreaterEqual(
            rain.lead_stats[leads[-1]][1], rain.lead_stats[leads[0]][1]
        )


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.store = dataio.load_store(DATA)

    def test_replay_before_events_shows_original_promises(self):
        view = replay(self.store, dt("2026-11-18T09:00:00+08:00"))
        self.assertEqual(view.commitments["C1"].open_kg, 330)
        self.assertEqual(view.commitments["C2"].open_kg, 330)

    def test_delivery_and_cancel_release_but_keep_records(self):
        view = replay(self.store, dt("2026-11-18T17:00:00+08:00"))
        c1 = view.commitments["C1"]
        c2 = view.commitments["C2"]
        self.assertEqual(c1.confirmed_kg, 330)  # 原始确认量不改写
        self.assertEqual(c1.delivered_kg, 260)
        self.assertEqual(c1.open_kg, 70)
        self.assertEqual(c2.cancelled_kg, 70)
        self.assertEqual(c2.open_kg, 260)
        # 已摘走的果单独记账
        self.assertEqual(view.delivered_fruit[("E1", date(2026, 11, 18))], 260)

    def test_over_delivery_rejected(self):
        from dataclasses import replace
        store = dataio.load_store(DATA)
        bad = replace(
            store.commitments[2],
            quantity_kg=9999.0,
            occurred_at=dt("2026-11-18T15:30:00+08:00"),
        )
        store.commitments.append(bad)
        with self.assertRaises(ValueError):
            replay(store, dt("2026-11-18T16:00:00+08:00"))


class PlanningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = dataio.load_store(DATA)
        cls.as_of = dt("2026-11-18T10:00:00+08:00")
        cls.horizon = [
            date(2026, 11, 18) + timedelta(days=i) for i in range(4)
        ]
        cls.bundle, cls.view, cls.engine = bt.planning_at(
            cls.store, cls.as_of, cls.horizon, n_scenarios=800
        )
        cls.requests = {r.request: r for r in cls.store.requests}

    def test_existing_commitments_not_rewritten(self):
        opt = self.engine.plan_request(
            self.requests["O1"],
            [r for r in self.store.requests if r.request != "O1"],
        )
        self.assertEqual(opt.existing_kg, 660)
        self.assertTrue(all(ln.is_existing for ln in opt.existing_lines))
        # 新增候选不得重复占用既有承诺的园日余量
        self.assertEqual(opt.residual_need_kg, 0.0)

    def test_hub_capacity_is_hard_constraint(self):
        req = self.requests["O2"]
        # 中心站单日能力 450，任何单线不得超过
        opt = self.engine.plan_request(
            req, [r for r in self.store.requests if r.request != "O2"]
        )
        for ln in opt.lines:
            cap = next(
                h.receiving_kg
                for h in self.store.hub_days
                if h.hub == ln.hub and h.day == ln.day
            )
            self.assertLessEqual(ln.kg, cap + 1e-6)

    def test_combine_uses_joint_shortfall_probability(self):
        req = self.requests["O2"]  # can_combine=1
        opt = self.engine.plan_request(
            req, [r for r in self.store.requests if r.request != "O2"]
        )
        self.assertGreaterEqual(len(opt.lines), 1)
        self.assertTrue(0.0 <= opt.shortfall_prob <= 1.0)
        # 手工复核：联合评估与选项给出的概率一致
        p, _ = self.engine.evaluate(
            req, opt.existing_lines + opt.lines, include_existing=True
        )
        self.assertAlmostEqual(p, opt.shortfall_prob, places=6)

    def test_no_combine_single_plot_only(self):
        req = self.requests["O3"]  # can_combine=0
        opt = self.engine.plan_request(
            req, [r for r in self.store.requests if r.request != "O3"]
        )
        plots = {ln.plot for ln in opt.lines}
        self.assertLessEqual(len(plots), 1)

    def test_firm_requires_tolerance_and_plot_basis(self):
        for req in self.store.requests:
            opt = self.engine.plan_request(
                req,
                [r for r in self.store.requests if r.request != req.request],
            )
            if opt.classification == "firm":
                self.assertLessEqual(
                    opt.shortfall_prob, req.shortfall_limit + 1e-9
                )
                self.assertTrue(
                    all(
                        ln.basis == "plot"
                        for ln in opt.lines + opt.existing_lines
                    )
                )

    def test_plan_all_respects_cumulative_occupancy(self):
        options = self.engine.plan_all(self.store.requests)
        # 全部新增线在同一 (hub,day) 的占用不得超过站能力
        used: dict[tuple[str, date], float] = {}
        for opt in options.values():
            for ln in opt.lines:
                used[(ln.hub, ln.day)] = used.get((ln.hub, ln.day), 0) + ln.kg
        for (hub, day), q in used.items():
            cap = next(
                h.receiving_kg
                for h in self.store.hub_days
                if h.hub == hub and h.day == day
            )
            reserved = self.view.reserved_hub.get((hub, day), 0.0)
            self.assertLessEqual(q + reserved, cap + 1e-6)

    def test_gap_analysis_preserves_promised_quantity(self):
        # 构造超承诺：封闭园 W3 11-18 承诺 5000kg
        view = replay(self.store, self.as_of)
        fake = OpenCommitment(
            "CX", "O3", "W3", "西站", date(2026, 11, 18),
            5000.0, 0.0, 0.0,
        )
        view.commitments["CX"] = fake
        view.reserved_fruit[("W3", date(2026, 11, 18))] = 5000.0
        view.reserved_hub[("西站", date(2026, 11, 18))] = 5000.0
        view.reservations_by_request.setdefault("O3", []).append("CX")
        eng = PlanningEngine(self.store, self.bundle, view)
        gaps = eng.gap_analysis()
        gap = next(g for g in gaps if g.reference == "CX")
        self.assertEqual(gap.open_kg, 5000)  # 承诺量保留
        self.assertEqual(gap.cover_prob, 0.0)
        self.assertTrue(gap.remedies)
        self.assertTrue(
            any("买方同意" in c for c in gap.buyer_consent_required)
        )

    def test_compare_keep_vs_adjust(self):
        req = self.requests["O1"]
        opt = self.engine.plan_request(
            req, [r for r in self.store.requests if r.request != "O1"]
        )
        result = self.engine.compare_plans(req, opt)
        self.assertIn("keep", result)
        self.assertIn("adjust", result)
        self.assertLessEqual(result["keep"]["shortfall_prob"], 1.0)


class BacktestTests(unittest.TestCase):
    def test_stratified_summaries_cover_all_layers(self):
        store = dataio.load_store(DATA)
        points = [
            datetime(2025, 11, d, 10, 0, tzinfo=BJ) for d in (7, 12, 17)
        ]
        rows, summaries = bt.backtest_at_points(
            store, points, horizon_days=(0, 1, 3), n_scenarios=200
        )
        layers = {(s.layer, s.value) for s in summaries}
        self.assertIn(("season", "2025"), layers)
        self.assertIn(("record_band", "thin"), layers)
        self.assertIn(("road_state", "closed"), layers)
        self.assertIn(("district", "河西"), layers)
        thin = next(
            s for s in summaries
            if s.layer == "record_band" and s.value == "thin"
        )
        self.assertTrue(thin.thin_sample)

    def test_actual_harvestable_zero_on_closed_day(self):
        from orchard.errors import actual_harvestable
        store = dataio.load_store(DATA)
        # 2024-11-01 沙坪封闭，E1 实际可采为 0
        self.assertEqual(
            actual_harvestable(store, "E1", date(2024, 11, 1)), 0.0
        )

    def test_actual_harvestable_ignores_car_full_delivery(self):
        from orchard.errors import actual_harvestable
        store = dataio.load_store(DATA)
        # 车满日园地仍有可采量（显著大于当日送站量）
        row = next(
            p for p in store.picking
            if p.limit_reason == "车满" and p.received_kg < p.harvested_kg
        )
        val = actual_harvestable(store, row.plot, row.day)
        self.assertGreater(val, row.received_kg)


if __name__ == "__main__":
    unittest.main()
