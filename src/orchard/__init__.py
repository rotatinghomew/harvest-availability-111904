"""园地逐日可采量预测与收货承诺调度。

模块划分：
- io:         原始 CSV 载入与“某时点可见”快照
- ripening:   样方驱动的成熟曲线（区分未熟/在树成熟/已采）
- weather:    多版本雨量预报、封路概率与同雨区联合情景
- forecast:   园地逐日可采量蒙特卡洛预测（质量守恒）
- errors:     分层回测误差（产季/记录量/道路状态/区域）
- ledger:     司机承诺事件台账（不可变记录）
- planning:   排车前承诺、联合风险、缺口与补救
- backtest:   历史时点重放
"""

__all__ = [
    "io",
    "ripening",
    "weather",
    "forecast",
    "errors",
    "ledger",
    "planning",
    "backtest",
]
