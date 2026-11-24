# 49998：B1 入阶与平地满杆修正

2026-11-24。本文主体记录训练启动前的设计与检查。发布快照时训练已经启动，实际已保存 checkpoint 和快照时间见仓库根目录 `PUBLISH_MANIFEST.json`。

新任务为 `M20-Stair-V11.1-EntrySequenceRepair-v0`，训练实验目录为 `logs/rsl_rl/M20-V11.1-EntrySequenceRepair/`。本次完成实现、CPU MuJoCo 基线采集、单元与回归测试、Isaac 环境/梯度检查；没有运行 PPO 更新，没有生成改善后的新权重。

## 初始化与接口

- 初始 Actor、Critic 和保护教师都使用 `interview_prep/model_49998.pt`，SHA256 为 `ef0fafd826687d8de283b4c56842c7d15f345bbd98e30120ed4af692a1cc6589`。
- 新目标使用全新 Adam 状态和 `1e-5` 初始学习率；Actor 全部参数可训练。
- Actor 63→16、Critic 568 不变；没有增加足端传感器或部署输入。
- nominal、动作缩放、PD、力矩限制、armature 已与新任务导出的运行配置逐项比较一致。观测项目、顺序、缩放与维度匹配；历史动作项由 `last_action` 改为 `executed_last_action`，使零命令停车时的轮速历史与现有真机/MuJoCo 控制器一致，仍为原来的 16 个历史动作值。
- 49998 原始训练 YAML 不在本仓库。`params_49998` 是此前 1515/67100 部署包继承的执行配置，关联 JSON 对 checkpoint 与 YAML 都校验哈希；不能把它称作 49998 原始训练配置。新训练会保存它实际使用的独立 `params/`。

## 基线的实际问题

新收集的固定工况全部来自 49998 的真实 CPU MuJoCo rollout：

| 工况 | 观测 |
|---|---|
| 平地站立起点、1.0 m/s | 前后轮伸展 p95 约 29.1 cm；偏航约 6.07°，横移约 39.7 cm |
| Z→C→直接 1.0 m/s、平地 | 横移约 29.7 cm，实速约 0.937 m/s；伸展 p95 约 8.5 cm |
| 固定平地满杆瞬态探针 | 俯仰峰值 3.17°、俯仰角速度 70.0°/s、右前轮向内约 10.0 cm、前轮间距最小 33.6 cm |
| 16 cm 高、26/30 cm 踏面楼梯 | 原生 8 个上楼工况均可到顶，但首个前轮支撑多为 B2，1.0 m/s 时另一个前轮还会首先支撑 B3/B4；安全入阶 0/8 |

不同起点/控制流程的数值不能混合解释为同一条轨迹。`baseline_only passed=true` 只表示证据结构有效；本 baseline 的严格 repair 判定仍为失败。

## 奖励冲突与处理

| 冲突/漏洞 | 新任务的处理 |
|---|---|
| 0.35 m/s 入阶成本与实际 0.7/1.0 m/s 命令相冲突 | 从 reward graph 删除 `v111_entry_overshoot_speed`，速度成本系数设为 0；恢复原生速度跟踪和前进缺额。没有固定入阶降速目标 |
| 第二腿早抬、稍晚落稳也领序列奖金 | 分开记录真实起抬与合法起抬；第二前腿只能在第一前腿已确认 B1 支撑后起抬；序列事件不再包含后腿保持信用 |
| 第一腿先离开 B1，历史双证书仍被当成正确交接 | 第一前轮必须持续在 B1 承重到第二前轮确认，失载锁存 `front_loss`；完整序列、clean 完成均拒绝该事件 |
| 前轮双证书过早解除后腿等待 | 后腿逐轮按到首阶踢面的实际距离释放，阈值 22 cm；双前轮完成后仍检查远处后腿早抬；后腿保持要求两轮都有载荷 |
| 早期错误耗尽所有风险预算 | 前腿交接 1.5、后腿 1.0、压缩/姿态 0.75、B1 越阶 1.0 四个独立账本；各项均有限，总原始预算 4.25 |
| 多个老探测器对 B1 重复付奖，鼓励先行腿追 B2 | 旧 B1 support/skip 奖励退出；补充的相邻阶、落脚和额外前进奖励在前轮 B1 阶段屏蔽，结束入阶后恢复。原生速度/前进奖励始终独立有效 |
| 对角腿 mirror 将前腿与远处后腿动作捆绑 | 后腿仍需要等待首阶期间屏蔽该 mirror 项，后腿接近踢面后恢复；不会给后续 B2+ 加入全局新顺序 |
| 平地收腿目标会压制入阶摆腿 | 新几何成本只在真实平地列且直行命令下生效；同时修正伸腿、路线偏航/横移、前倾和支撑宽度，成本最多 0.5/秒。正向质量奖金必须有实际前进，停车不能领取 |
| 保护教师把错误 B1 与平地偏置拉回来 | 所有前后腿 B1 动作和 B1 驱动均不模仿；平地只保留共同前向驱动，释放旧伸腿/转向目标；B2+ 保护只使用四轮都已建立 >=B2 支撑后再过 120 ms 的片段 |
| 每次失败不断压低全局 LR | 仅跌倒、绕行、停滞、失去启动/到顶等能力退化收紧；LR 最低为初始的 60%，保护倍数最多 1.5，无此类退化时恢复到初始值。B1 未学会、平地姿态/偏航未达标不会触发旧的单向学习率收紧 |

保留局部策略变化约束、镜像等变约束和已有低噪声训练；没有新增全局动作幅度成本。错误之后仍可继续前进、重同步后续台阶并获得后续奖励，行为失败不停止训练。

## 样本与验收

上楼直接练习 0.5/0.7/1.0 m/s；部分样本先空闲再直接前进，未加前向升速滤波。平地新增原始 0→1.0、0→0.3、0.3→1.0 样本，其他平地样本保留原生分布。键盘/生产满杆采用当前真机的 raw 1.0 m/s 路径；候选渐进升速工况另外记录。

重新构建的回放共 6730 帧，其中 2212 帧属于测量确认的四轮 post-B2 区间，来自 8 个完整上楼案例。旧 47900/66800 回放未作为本阶段教师。

每 100 轮保存分段 checkpoint，默认每 1000 轮做物理评估，末尾也评估。保留原生速度/全楼梯、直接 C→前进、平地满杆和生产 1.0 m/s 工况；entry profile 不再跑牵引/下楼附加套件。评估失败记录并继续，运行错误和实际资源不足仍会退出。

flat 探针现在同时检查 `yaw_peak_deg`、`lateral_peak_m`；它们进入相对退化判断、严格通过条件、候选排序和 TensorBoard。MuJoCo 序列诊断升级为 `entry_transfer_v3`，检查第一前轮支撑是否持续到交接结束。训练指标改为 episode 锁存/累计值：

- `v111_entry_transfer_first_lift/first_support/second_legal_lift`：前缀到达率；
- `v111_entry_transfer_sequence/clean_complete`：合法序列与完整成功率；
- `front_loss/rear_early/skipped`：未保持支撑、后腿早抬与越阶；
- 四个 `*_cost_used`：独立风险账本；
- `v111_sequence_flat_reach/yaw/lateral/speed`：真实平地伸展、航向、横移与实速；
- `FlatStart/*/yaw_peak_deg/lateral_peak_m` 与 `Production/current_z_c_8/*`：测量的 raw 满杆结果。

测量最优模型单独存入本次 session 的 `best_candidates/`，不会被后续最新模型覆盖。新 run 只清理自身较旧且未评估的中间 checkpoint；已评估 checkpoint、最优归档和历史 run 不删除。

## 检查结果与证据

- 72 项测试通过：含提前抬腿后错时落地、首腿中途失载、无真实抬腿却双轮 B1 支撑、双前轮完成后后腿远处早抬、风险预算互不挤占、伪造 post-B2 mask、真实 RSL 辅助梯度、反馈恢复、最优模型归档/续训、行为失败继续训练。
- 实际 Isaac 检查：64 环境、350 控制步，实际奖励/重置有限，平地成本在非平地列恒为零，实际采到 raw 1.0 m/s；全部 Actor 参数可训练、教师冻结、梯度有限、优化器状态为空且权重逐项未改变。
- 导出配置可由 MuJoCo SavedPolicy 读取，部署执行参数一致、63 维观测合同兼容；历史动作使用上述停车处理后的值。
- **这些检查证明接线与反例修复，不证明已经学会安全入阶或真机问题已消失。需要使用新训练权重与本次固定基线比较。**

证据目录：`artifacts/entry_49998_repair_20261124/`，包含 `reference/`、`tests.log`、`preflight.log`、`frozen_runtime.json`、`runtime_params/`、`deployment_contract.json`。`frozen_runtime.json` 内包含最终启用的逐项 reward audit。

## 手动训练

```bash
cd /home/arcus/deeprobotics_m20_rl/M20_Perception_Lidar_RL/M20_perception_Lidar_rl
M20_NUM_ENVS=2048 M20_TOTAL_UPDATES=50000 M20_EVAL_EVERY=1000 \
  bash scripts/train_m20_v111_sequence_from49998.sh
```

这是从 49998 额外学习 50000 轮，预期末尾编号为 99998。脚本未设置 `M20_NUM_ENVS` 时默认 2048，可自行覆盖。

只检查而不启动训练：

```bash
bash scripts/train_m20_v111_sequence_from49998.sh --check-only
```

中断后的同阶段续训，使用新训练目录里的 `resume_after_eval.pt`，或它之后同阶段的 `model_N.pt`；`M20_TOTAL_UPDATES` 表示本次再追加多少轮：

```bash
M20_RESUME_CHECKPOINT=/absolute/path/to/new_run/resume_after_eval.pt \
M20_NUM_ENVS=2048 M20_TOTAL_UPDATES=10000 \
  bash scripts/train_m20_v111_sequence_from49998.sh
```

若使用评估之后才产生的 `model_N.pt`，它携带上一轮评估时已保存的保护与最优模型状态；紧接本轮评估写出的 `resume_after_eval.pt` 还包含本轮反馈。不可用旧 EntryTransfer checkpoint 直接恢复优化器到这个新目标。
