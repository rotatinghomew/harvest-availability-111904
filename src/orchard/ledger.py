"""司机承诺事件台账。

承诺是只增不改的事件流：确认占用果与站能力；交货/取消按此次实际变动
释放或结转占量。已摘走的果（交货部分）不回预测余额。

关键约束：台账记录与分析员的预测是两种记录。预测可以随新盘点/新预报
重算，但任何重算都不得改写司机已经拿走的单子——本模块不提供修改历史
事件的接口，只提供“截至某时点重放”的只读视图。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from . import io as dataio


@dataclass
class OpenCommitment:
    reference: str
    request: str
    plot: str
    hub: str
    promised_day: date
    confirmed_kg: float
    delivered_kg: float
    cancelled_kg: float

    @property
    def open_kg(self) -> float:
        """仍占用果与站能力的量。"""
        return max(
            self.confirmed_kg - self.delivered_kg - self.cancelled_kg, 0.0
        )

    @property
    def is_closed(self) -> bool:
        return self.open_kg <= 1e-9


@dataclass
class LedgerView:
    as_of: datetime
    commitments: dict[str, OpenCommitment]
    # 派生索引
    reserved_fruit: dict[tuple[str, date], float]      # (plot, day) -> 占用 kg
    reserved_hub: dict[tuple[str, date], float]        # (hub, day) -> 占用 kg
    delivered_fruit: dict[tuple[str, date], float]     # 已摘走，不回池
    reservations_by_request: dict[str, list[str]]

    def open_for_request(self, request: str) -> list[OpenCommitment]:
        return [
            self.commitments[ref]
            for ref in self.reservations_by_request.get(request, [])
            if not self.commitments[ref].is_closed
        ]


def replay(
    store: dataio.DataStore, as_of: datetime
) -> LedgerView:
    """只按 occurred_at <= as_of 重放事件，重现历史时点的真实占量。"""
    open_items: dict[str, OpenCommitment] = {}
    for ev in sorted(store.commitments, key=lambda c: c.occurred_at):
        if ev.occurred_at > as_of:
            continue
        item = open_items.get(ev.reference)
        if ev.status == "确认":
            if item is not None:
                raise ValueError(
                    f"承诺 {ev.reference} 重复确认；事件流不可变，拒绝重放"
                )
            open_items[ev.reference] = OpenCommitment(
                reference=ev.reference,
                request=ev.request,
                plot=ev.plot,
                hub=ev.hub,
                promised_day=ev.promised_day,
                confirmed_kg=ev.quantity_kg,
                delivered_kg=0.0,
                cancelled_kg=0.0,
            )
        elif ev.status == "交货":
            if item is None:
                raise ValueError(f"交货事件 {ev.reference} 缺少在先确认")
            if item.delivered_kg + ev.quantity_kg > item.confirmed_kg + 1e-9:
                raise ValueError(f"交货 {ev.reference} 超过确认量")
            item.delivered_kg += ev.quantity_kg
        elif ev.status == "取消":
            if item is None:
                raise ValueError(f"取消事件 {ev.reference} 缺少在先确认")
            if (
                item.cancelled_kg + ev.quantity_kg
                > item.confirmed_kg - item.delivered_kg + 1e-9
            ):
                raise ValueError(f"取消 {ev.reference} 超过未交量")
            item.cancelled_kg += ev.quantity_kg
        else:
            raise ValueError(f"未知承诺事件状态: {ev.status}")

    reserved_fruit: dict[tuple[str, date], float] = {}
    reserved_hub: dict[tuple[str, date], float] = {}
    delivered_fruit: dict[tuple[str, date], float] = {}
    by_request: dict[str, list[str]] = {}
    for ref, item in open_items.items():
        by_request.setdefault(item.request, []).append(ref)
        if not item.is_closed:
            k = (item.plot, item.promised_day)
            reserved_fruit[k] = reserved_fruit.get(k, 0.0) + item.open_kg
            h = (item.hub, item.promised_day)
            reserved_hub[h] = reserved_hub.get(h, 0.0) + item.open_kg
        if item.delivered_kg > 0:
            d = (item.plot, item.promised_day)
            delivered_fruit[d] = (
                delivered_fruit.get(d, 0.0) + item.delivered_kg
            )
    return LedgerView(
        as_of=as_of,
        commitments=open_items,
        reserved_fruit=reserved_fruit,
        reserved_hub=reserved_hub,
        delivered_fruit=delivered_fruit,
        reservations_by_request=by_request,
    )
