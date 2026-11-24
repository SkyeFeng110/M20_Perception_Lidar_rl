# M20 感知训练工程

当前发布任务为 **`M20-Stair-V11.1-EntrySequenceRepair-v0`**，仅保留 `logs/rsl_rl/M20-V11.1-EntrySequenceRepair/` 这次训练的权重和运行记录。

克隆、MuJoCo 验证、冻结教师依赖与继续训练方法见 [仓库 README](../README.md)；本次设计和评估依据见 [EntrySequenceRepair 说明](docs/m20_entry_sequence_flat_from49998_20261124.md)。训练源码中的早期版本保留为当前任务的基础依赖，训练日志之外另完整分发 `interview_prep/` 基线与教师。

训练入口：`scripts/train_m20_v111_sequence_from49998.sh`。验证入口会自动读取所选 checkpoint 同目录的 `params/`。49998 冻结教师位于随仓库分发的 `interview_prep/`，本次训练的 checkpoint 已包含在指定 logs 目录中。
