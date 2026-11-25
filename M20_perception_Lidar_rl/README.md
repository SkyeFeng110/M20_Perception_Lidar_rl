# M20 感知训练工程

本次更新包含 **`M20-Stair-V11.1-FlatPosture-v0`** 的平地姿态修复和关节目标限位。logs 新增指定 run 的 `model_56998.pt`、真实配置及训练日志；EntrySequenceRepair 之前发布的 `50498`、`50598` 和日志原样保留，`interview_prep/` 继续完整上传。

克隆、MuJoCo 验证、两阶段训练及跨机器恢复的边界见 [仓库 README](../README.md)。本次设计见 [FlatPosture 说明](docs/m20_flat_posture_latest_20261124.md)，上一阶段见 [EntrySequenceRepair 说明](docs/m20_entry_sequence_flat_from49998_20261124.md)。训练源码中的早期版本保留为当前任务的基础依赖。

当前训练入口：`scripts/train_m20_v111_flat_posture_latest.sh`。新主机先用 `M20_PYTHON=python bash scripts/train_m20_v111_flat_posture_latest.sh --check-only` 检查。验证 `56998` 时显式指定 checkpoint，程序读取该 run 的真实 `params/`。
