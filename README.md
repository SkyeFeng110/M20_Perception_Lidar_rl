# M20 Perception LiDAR RL

M20 轮足机器人感知运动策略的 Isaac Lab / RSL-RL 训练代码、MuJoCo 验证程序与真机部署工具。当前工作任务是 **`M20-Stair-V11.1-EntrySequenceRepair-v0`**：修正第一阶 B1 支撑交接和平地满杆姿态，同时保护后续楼梯运动能力。策略接口为 **63 维观测 → 16 维动作**，Critic 为 568 维。

训练记录仅分发 `M20_perception_Lidar_rl/logs/rsl_rl/M20-V11.1-EntrySequenceRepair/` 这次训练的 checkpoint、实际 `params/`、TensorBoard 事件和运行差异；另完整保留用户指定的 `interview_prep/`，包含 `35300` 基线与 `49998` 冻结教师。快照时训练仍在继续，准确时间、文件清单和 SHA-256 见 [PUBLISH_MANIFEST.json](PUBLISH_MANIFEST.json)。训练之后产生的文件需再次同步。

```text
M20_perception_Lidar_rl/
  source/rl_training/       任务、观测、奖励、PPO 与其继承依赖
  scripts/                  当前训练入口、预检查、行为评估和测试
  logs/rsl_rl/M20-V11.1-EntrySequenceRepair/
                            本次训练权重、真实配置和 TensorBoard
  interview_prep/           基线、49998 教师、关联配置与说明
  data/m20_recovery/         本阶段带标签回放及来源哈希
  artifacts/entry_49998_repair_20261124/
                            本阶段基线轨迹、评估结果和配置证据
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
  --stair-height 0.16 --tread-depth 0.26
```

验证入口默认选择本次训练目录中编号最大的 `model_N.pt`。也可用 `--checkpoint PATH` 指定本次其他模型。运行配置自动读取该模型旁的 `params/env.yaml`、`params/agent.yaml`；迁移模型时应一同复制实际配置。

按 Z 起立、C 进入运控；8/5 前后、4/6 左右、7/9 转向；R 复位站立、Backspace 复位趴卧、Esc 退出。行为评估是否改善须看真实测量，不能仅根据 checkpoint 编号或训练奖励判断。

## 训练与继续训练

训练环境需要另行安装 Isaac Sim、Isaac Lab 和 RSL-RL 5.x。本机记录的依赖版本见 [environment_versions_20261117.json](M20_perception_Lidar_rl/docs/environment_versions_20261117.json)。当前入口为 `scripts/train_m20_v111_sequence_from49998.sh`，默认 2048 环境、追加 50000 更新、每 1000 更新做物理评估。

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
  --output /absolute/path/to/new_snapshot
```

导出脚本复制当前文件，限制权重目录并生成带哈希的清单；不会删除本地文件或停止训练。官方机器人资源以 `.gitmodules` 所列子模块及清单中的固定提交分发，初始化新快照 Git 仓库后需添加相应 gitlink。新快照不携带旧 Git 历史。
