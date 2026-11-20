# 可采量与收货调度

三产季园地盘点、路况和预报均为合成值。样方重量和订单重量均以公斤表示。

`data` 保存原始记录，`DATA.md` 是资料字典。`src/orchard` 是预测与承诺应用，
`tests` 含原始资料契约与业务不变量测试。检查只读取本地数据，不访问业务系统。

## 安装与测试

环境使用 Python 3.11.2：

```sh
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
python3 -m unittest discover -s tests -v
```

所有运行脚本均以 `python3 -I` 启动，隔离本地目录注入。

## 业务口径（不可妥协的约束）

1. **送站重量不是产量**。封路、车满、直销压低的是送站数，预测可采量只使用
   样方在树存量与实采量（`harvested_kg`），不使用 `received_kg`。
2. **果实三分量且质量守恒**。每个情景把果分为未熟、在树成熟（可采未采）、
   已采（含直销）。封路日可采为 0 但在树存量结转到次日；同一批果在多日窗内
   只移动采收日，不会累加成新增产量。锚定日库存恒等式偏差为数值零。
3. **历史重放只看当时可见的记录**。样方按 `reported_at`、采收按 `known_at`、
   预报按 `issued_at` 取最新版本、路况按 `known_at`、承诺按 `occurred_at` 过滤。
4. **薄样地块不冒充保证**。样方不足 6 个的园地走同区同等级池化拟合
   （`basis=pooled`，无同区捐赠者时跨区并显式标注），无任何样方则
   `basis=none` 且预测单元格为 NaN，不出数值承诺。池化误差在回测中单独分层。
5. **联合风险不靠分位数相加**。同雨区园地共享封路情景，区域/全区区间在
   蒙特卡洛情景维度直接求和后取分位，保留“一场雨几个园一起落空”的相关性。
6. **承诺与预测是两种记录**。司机承诺是只增事件流（确认/交货/取消），
   新预测只能给出缺口与补救，不能改写已拿走的单子；已摘走的果不回预测余额。

## 命令

```sh
# 园地逐日可采量预测（含三分量、封路概率、区域联合区间）
python3 -I -m orchard.cli --scenarios 2000 forecast \
  --as-of 2026-11-18T10:00:00+08:00 \
  --first 2026-11-18 --last 2026-11-21 --out reports/forecast.csv

# 排车前承诺：firm / conditional / infeasible，含既有承诺、联合缺货概率
python3 -I -m orchard.cli --scenarios 2000 plan \
  --as-of 2026-11-18T10:00:00+08:00 --first 2026-11-18 --last 2026-11-21

# 已批准订单在新预测下的缺口与补救（保留承诺量与发货窗）
python3 -I -m orchard.cli --scenarios 2000 gaps \
  --as-of 2026-11-19T10:00:00+08:00 --first 2026-11-18 --last 2026-11-21

# 保持原计划 vs 调整计划的交付风险对比
python3 -I -m orchard.cli --scenarios 2000 compare --request O1 \
  --as-of 2026-11-18T10:00:00+08:00 --first 2026-11-18 --last 2026-11-21

# 分层回测误差（产季/记录量/拟合依据/道路状态/区域/预见期）
python3 -I -m orchard.cli --scenarios 600 backtest --out reports/errors.csv

# 任意时点的承诺台账重放
python3 -I -m orchard.cli ledger --as-of 2026-11-18T17:00:00+08:00
```

## 模块结构

| 模块 | 职责 |
| --- | --- |
| `orchard.io` | CSV 载入与按 `as_of` 的可见性快照 |
| `orchard.ripening` | 样方还原累计成熟、Gompertz 曲线、薄样池化、损耗率标定 |
| `orchard.weather` | 多版本预报误差、封路逻辑回归、同雨区联合抽样 |
| `orchard.forecast` | 三分量库存滚动蒙特卡洛、质量守恒校验、区域联合区间 |
| `orchard.errors` | 历史时点重放回测与分层误差汇总 |
| `orchard.ledger` | 不可变承诺事件台账与时点重放 |
| `orchard.planning` | firm/conditional 承诺、拼车联合风险、站能力、缺口补救 |
| `orchard.backtest` | 预测与排产的装配入口 |
