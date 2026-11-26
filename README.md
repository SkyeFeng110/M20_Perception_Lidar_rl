# M20 Perception LiDAR RL

M20 轮足机器人感知运动策略的 Isaac Lab / RSL-RL 训练源码、MuJoCo 验证程序与真机部署工具。本次发布 2026-11-26 工作区中的 EarlyTraction / DescentTractionV2 源码，保留基础任务的继承依赖；策略接口为 63 维观测 → 16 维动作，Critic 为 568 维。

## 本次模型与训练记录

`logs/` 只保留以下 run：

```text
M20_perception_Lidar_rl/logs/rsl_rl/M20-V11.1-EarlyTraction58098/
  2026-11-25_17-53-45_450637_early_traction_from58098/
    model_65298.pt
    params/env.yaml
    params/agent.yaml
    events.out.tfevents.*
    mujoco_eval/events.out.tfevents.*
    actor_observation_manifest.txt
    git/M20_Perception_Lidar_rl.diff
```

模型 SHA-256：`059768c807e0d4d67aa3643496d8791a786f24e2d423ce5de3faf19c68317d3f`。完整文件大小和哈希见 [PUBLISH_MANIFEST.json](PUBLISH_MANIFEST.json)，本次发布检查见 [VALIDATION_REPORT.json](VALIDATION_REPORT.json)。

该训练从 58098 初始化，记录为追加完成 7200 次更新后由用户停止，最后 checkpoint 为 65298。对应的启动参数、源码快照、状态、TensorBoard 配套评估记录和失败证据保存在 `M20_perception_Lidar_rl/artifacts/early_traction_58098_20261125_175334_1006334/`。记录中的 `repair_passed=false`，65298 是用户指定发布的末次模型，不能据此认定为通过行为验收的最佳策略。

原先发布的 EntrySequenceRepair、FlatPosture 和 RouteHold logs 已从当前分支文件树移除。Git 提交历史保留，历史提交仍可访问旧文件；所有本地历史文件保留。

## 保留范围

- `M20_perception_Lidar_rl/source/`、`scripts/`：当前训练、课程、奖励、模型选择、评估与所需基础模块。训练启动脚本仅保留 EarlyTraction 与 DescentTractionV2 两个入口。
- `data/m20_recovery/` 和所需 `artifacts/`：教师回放、原始配置和行为基线；35300、49998、56898、58098 是依赖的冻结教师，位于 logs 之外。未新增其他实验的中间权重，也不重复存放 65298。
- `mujoco_validation/deploy/mujoco_v135/`、`deploy_real/`：仿真、导出、感知和关节保护源码。
- `deep_robotics_model/`：官方机器人资源子模块，固定提交 `18192847e16ab85c056440072fcc5844cef43856`。

重复的 `M20_perception_Lidar_rl (2)/`、压缩包、环境、缓存、录像、无关输出、旧训练启动脚本和未选中的模型不上传。本地源仓库不执行删除操作。

## 克隆与 MuJoCo

```bash
git clone --recurse-submodules https://github.com/SkyeFeng110/M20_Perception_Lidar_rl.git
cd M20_Perception_Lidar_rl
python3 -m venv .venv-mujoco
source .venv-mujoco/bin/activate
python -m pip install -r mujoco_validation/deploy/mujoco_v135/requirements.txt
bash mujoco_validation/run_mujoco_v135.sh --viewer --keyboard --checkpoint M20_perception_Lidar_rl/logs/rsl_rl/M20-V11.1-EarlyTraction58098/2026-11-25_17-53-45_450637_early_traction_from58098/model_65298.pt --stair-height 0.16 --tread-depth 0.26
```

默认入口也优先选择本次发布的 65298；实际执行配置从相邻 `params/` 读取。Z 起立，C 进入运控，8/5 前后，4/6 左右，7/9 转向，R 复位站立，Backspace 复位趴卧，Esc 退出。

## 训练源码与迁移

新阶段的下楼纠偏、低摩擦抗滑与前轮主导转向实现见 [DescentTractionV2 说明](M20_perception_Lidar_rl/docs/M20_DESCENT_TRACTION_TURN_V2_20261126.md)。发布时的源码比 65298 的原训练阶段更新；原训练源码和配置保存在上述 run 证据中，不能将新 V2 的训练目标当作 65298 已有的能力。

训练环境需另行安装 Isaac Sim、Isaac Lab 和 RSL-RL 5.x，依赖记录见 [environment_versions_20261117.json](M20_perception_Lidar_rl/docs/environment_versions_20261117.json)。入口为：

```text
M20_perception_Lidar_rl/scripts/train_m20_v111_early_traction_from58098.sh
M20_perception_Lidar_rl/scripts/train_m20_v111_descent_traction_v2.sh
```

V2 默认以本次 65298 初始化新优化器，58098 仍为能力保持教师。原始 `launch.json`、配置与评估清单保留来源机器的绝对路径，跨机器训练前须迁移引用并重新核验资产；缓存 `ready.json` 不能直接作为新机器验证结果。真机工具还需实际 SDK、ROS 与硬件环境。本次发布没有启动训练、MuJoCo 行走或真机运动。

## 再次导出

在此版本仓库根目录执行：

```bash
python3 M20_perception_Lidar_rl/tools/export_entry_sequence_snapshot.py --output /absolute/path/to/new_snapshot
```

导出规则见 `M20_perception_Lidar_rl/tools/entry_sequence_export_policy.json`，使用明确的 checkpoint 白名单，不保留旧 logs，也不删除源文件。导出后应重新执行发布检查并更新验证记录。
