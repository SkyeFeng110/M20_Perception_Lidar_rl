# M20 感知训练工程

本次更新发布 **`M20-Stair-V11.1-RouteHold-v0`** 代码，以及 RouteHold run `2026-11-25_11-30-18_303338_route_hold_latest` 的 `model_58098.pt`、真实配置和 TensorBoard 训练/评估事件。该 run 的其他 checkpoint 不上传；EntrySequenceRepair 和 FlatPosture 已发布 logs 原样保留，`interview_prep/` 继续完整上传。

克隆、MuJoCo 验证、训练及跨机器运行边界见 [仓库 README](../README.md)。本阶段设计和验证范围见 [RouteHold 说明](docs/m20_route_hold_56898_20261125.md)。训练源码中的基础模块保留为当前任务依赖。

当前训练入口：`scripts/train_m20_v111_route_hold_latest.sh`。新主机先用 `M20_PYTHON=python bash scripts/train_m20_v111_route_hold_latest.sh --check-only` 检查。MuJoCo 验证 `58098` 时显式指定 checkpoint，程序读取相邻的真实 `params/`。
