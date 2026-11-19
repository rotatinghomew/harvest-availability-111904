# 资料字典

CSV 为 UTF-8，第一行为列名。日期为北京时间，时间带偏移；资料可见时刻与业务发生时刻分别以字段含义为准。空值不能默认做零。

### `plots.csv`

雨区同时影响道路与成熟进度；树量是样方外推的经营台账。

字段：`plot`、`district`、`rain_zone`、`bearing_trees`、`primary_grade`。

### `samples.csv`

样方树上在库果量及其中符合主要等级的成熟重量，不包括此前已采；`reported_at` 可能晚于观察日。

字段：`plot`、`day`、`sampled_trees`、`mass_tree_kg`、`mature_kg`、`reported_at`。

### `picking.csv`

实采、其中直销、送站验收重量是不同观测。封路或满车的接站重量不是可采能力；`known_at` 是次日传回时刻。

字段：`plot`、`day`、`harvested_kg`、`direct_kg`、`received_kg`、`limit_reason`、`known_at`。

### `forecasts.csv`

同一目标日可有多个发布版本，预报不是最终天气观测。

字段：`rain_zone`、`target_day`、`issued_at`、`rain_mm`、`low_c`。

### `road_log.csv`

道路封控实况，仅在可见后才能作已知状态。

字段：`rain_zone`、`service_day`、`passable`、`known_at`、`note`。

### `route_capacity.csv`

可经过的站点，山区封闭禁止进出。

字段：`plot`、`hub`、`route_class`。

### `hub_days.csv`

站按日期的验收能力，多个订单共享。

字段：`hub`、`day`、`receiving_kg`。

### `requests.csv`

缺货容忍度为允许发生未足量到货的概率上限；不得用它当折扣重量。窗按日期、含两端。

字段：`request`、`market`、`first_day`、`last_day`、`grade`、`kg`、`shortfall_limit`、`can_combine`、`can_switch_hub`。

### `commitments.csv`

确认、交货、取消是事件。交货和取消记录的数量是此次实际变动；事件到达后相应减少原占量。尚不知后续交货的历史时点应重现原承诺。

字段：`reference`、`request`、`plot`、`hub`、`promised_day`、`quantity_kg`、`status`、`occurred_at`。
