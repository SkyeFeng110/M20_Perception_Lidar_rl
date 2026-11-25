# M20 Perception LiDAR RL

M20 轮足机器人感知运动策略的 Isaac Lab / RSL-RL 训练代码、MuJoCo 验证程序与真机部署工具。本次更新发布 **`M20-Stair-V11.1-RouteHold-v0`** 及 RouteHold `model_58098.pt`、真实配置和训练日志；此前 EntrySequenceRepair 与 FlatPosture logs 原样保留。策略接口为 **63 维观测 → 16 维动作**，Critic 为 568 维。

本次 logs 仅新增 `M20-V11.1-RouteHold/2026-11-25_11-30-18_303338_route_hold_latest/model_58098.pt`，连同该 run 的真实 `params/`、TensorBoard 训练与评估事件、观测清单和启动时 Git diff。其他 RouteHold 权重不上传。RouteHold 56898 冻结教师及其测量轨迹保存在 `artifacts/route_hold_20261125/`，58098 traction 基线结果保存在 `artifacts/route_traction_20261125/reference58098/traction_b1_b3/`。

原先已发布的 EntrySequenceRepair `50498`、`50598` 及其日志原样保留，没有追加该旧实验后来产生的 checkpoint。`interview_prep/` 继续完整分发，包含 `35300` 基线和 `49998` 冻结教师。准确上传时间、文件清单与 SHA-256 见 [PUBLISH_MANIFEST.json](PUBLISH_MANIFEST.json)。训练仍在运行，本次上传不会中断它。

```text
M20_perception_Lidar_rl/
  source/rl_training/       任务、观测、奖励、PPO 与其继承依赖
  scripts/                  当前训练入口、预检查、行为评估和测试
  logs/rsl_rl/M20-V11.1-EntrySequenceRepair/
                            原先发布的 50498、50598 和日志
  logs/rsl_rl/M20-V11.1-FlatPosture/
                            前次发布的 56998、真实配置与训练日志
  logs/rsl_rl/M20-V11.1-RouteHold/
                            本次指定的 58098、真实配置与训练日志
  interview_prep/           基线、49998 教师、关联配置与说明
  data/m20_recovery/         本阶段带标签回放及来源哈希
  artifacts/entry_49998_repair_20261124/
                            EntrySequenceRepair 基线证据
  artifacts/flat_posture_repair_20261124/
                            FlatPosture 教师行为轨迹与基线证据
  artifacts/flat_posture_20261124_181441_2759441/
                            指定 run 的状态、指标与评估日志
  artifacts/route_hold_20261125/
                            56898 冻结教师及本阶段物理评估数据
  artifacts/route_traction_20261125/reference58098/
                            58098 traction 基线和测量报告
  deep_robotics_model/      官方机器人资源子模块
  docs/                     当前设计与相关历史说明
  tools/                    可重复的快照导出脚本和过滤规则
mujoco_validation/
  deploy/mujoco_v135/       独立 MuJoCo 验证、M20 资源与地形
  deploy_real/             真机导出、感知接入和关节保护工具源码
```

早期训练 logs、上述目录之外的 checkpoint、旧启动脚本和版本审计工具、无关 AMP 动作数据、重复部署包、虚拟环境、缓存、录像和临时输出不包含在发布中。所有本地历史文件保留。为保持当前任务的继承和导入完整，训练源码中的基础版本模块保留；未随本次分发的旧任务及旧模型路径不能直接使用。部署工具需配合实际 SDK、ROS 和硬件环境，完整机器安装包未包含。

## 克隆与 MuJoCo 验证

```bash
git clone --recurse-submodules https://github.com/SkyeFeng110/M20_Perception_Lidar_rl.git
cd M20_Perception_Lidar_rl
python3 -m venv .venv-mujoco
source .venv-mujoco/bin/activate
python -m pip install -r mujoco_validation/deploy/mujoco_v135/requirements.txt
bash mujoco_validation/run_mujoco_v135.sh --viewer --keyboard \
  --checkpoint M20_perception_Lidar_rl/logs/rsl_rl/M20-V11.1-RouteHold/2026-11-25_11-30-18_303338_route_hold_latest/model_58098.pt \
  --stair-height 0.16 --tread-depth 0.26
```

上面的命令明确加载本次发布的 `58098`。未指定 `--checkpoint` 时仍沿用 EntrySequenceRepair 默认模型；切换模型时应显式指定路径。运行配置自动读取该模型旁的 `params/env.yaml`、`params/agent.yaml`；迁移模型时应一同复制实际配置。

按 Z 起立、C 进入运控；8/5 前后、4/6 左右、7/9 转向；R 复位站立、Backspace 复位趴卧、Esc 退出。行为评估是否改善须看真实测量，不能仅根据 checkpoint 编号或训练奖励判断。

## 训练与继续训练

训练环境需要另行安装 Isaac Sim、Isaac Lab 和 RSL-RL 5.x。本机记录的依赖版本见 [environment_versions_20261117.json](M20_perception_Lidar_rl/docs/environment_versions_20261117.json)。当前 RouteHold 入口默认 2048 环境、追加 50000 更新、每 1000 更新做物理评估。

新主机激活已安装好的 Isaac Lab 环境后，从仓库根目录检查：

```bash
python -m pip install -e M20_perception_Lidar_rl/source/rl_training --no-deps
cd M20_perception_Lidar_rl
M20_PYTHON=python bash scripts/train_m20_v111_route_hold_latest.sh --check-only
```

正式开始 RouteHold 训练：

```bash
M20_PYTHON=python M20_NUM_ENVS=2048 M20_TOTAL_UPDATES=50000 M20_EVAL_EVERY=1000 \
  bash scripts/train_m20_v111_route_hold_latest.sh
```

该命令自动选择已发布的 `58098` 初始化 Actor/Critic，并创建新的优化器；56898 教师和 traction 基线用于保护既有能力及比较早期履带接触滑移。当前日志仅包含权重和常规训练记录，适合初始化新训练；它不包含续训优化器状态。阶段目标、评估协议和未验证边界见 [RouteHold 说明](M20_perception_Lidar_rl/docs/m20_route_hold_56898_20261125.md)。FlatPosture 入口和设计记录仍保留在仓库中。

以下为保留的 EntrySequenceRepair 使用说明，其入口为 `scripts/train_m20_v111_sequence_from49998.sh`。

本阶段使用 `interview_prep/model_49998.pt` 作为冻结教师，已随完整 `interview_prep/` 一同上传，SHA-256 为 `ef0fafd826687d8de283b4c56842c7d15f345bbd98e30120ed4af692a1cc6589`。如需使用其他存放路径，可用 `M20_RECOVERY_TEACHER` 指向相同文件，并携带相邻的 `.config.json` 和其引用的执行配置。`interview_prep/params_49998/` 是经哈希关联的历史执行配置，不能称作原始训练配置。

激活目标主机的 Isaac Lab 环境后，仅检查依赖和基线：

```bash
cd M20_perception_Lidar_rl
M20_PYTHON=python3 bash scripts/train_m20_v111_sequence_from49998.sh --check-only
```

从本次已发布的 checkpoint 继续训练，入口会使用随仓库分发的冻结教师：

```bash
M20_PYTHON=python3 \
M20_RESUME_CHECKPOINT=/absolute/path/to/M20-V11.1-EntrySequenceRepair/RUN/model_N.pt \
M20_NUM_ENVS=2048 M20_TOTAL_UPDATES=10000 \
  bash scripts/train_m20_v111_sequence_from49998.sh
```

跨机器续训还需迁移 checkpoint 内 `m20_step_retention.best_candidate` 的归档路径及对应证据；它们保留了原机绝对路径，`--check-only` 不会检查这一运行期恢复环节。原机继续训练不受此路径迁移影响。

每次运行保存自己的真实 `params/`；不能用另一次训练的配置替代。具体奖励、样本、评估和未验证边界见 [本阶段说明](M20_perception_Lidar_rl/docs/m20_entry_sequence_flat_from49998_20261124.md)。

## 更新发布快照

在包含原始训练记录的工作仓库根目录执行：

```bash
python3 M20_perception_Lidar_rl/tools/export_entry_sequence_snapshot.py \
  --published-root /absolute/path/to/existing_published_checkout \
  --output /absolute/path/to/new_snapshot
```

导出脚本复制当前文件，限制权重目录并生成带哈希的清单；不会删除本地文件或停止训练。此前已发布的 EntrySequenceRepair 日志从 `--published-root` 原样保留。官方机器人资源以 `.gitmodules` 所列子模块及清单中的固定提交分发。上传更新继续接在既有 GitHub 提交之后，不携带原训练仓库的 Git 历史。
