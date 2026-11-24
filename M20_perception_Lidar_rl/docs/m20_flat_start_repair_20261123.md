# 平地加速与连续入阶修正阶段

本阶段按用户最新要求默认从 67700 初始化，使用新优化器；保留 66800 的成功楼梯回放及测量基准。下文已有的 67100 物理测量仍是历史诊断证据，不代表已验证 67700 的行为。没有启动新训练，没有停止旧训练，没有更新机器人上的程序。

## 速度范围核查

67100 所属训练目录原始 `params/env.yaml` 的 `commands.base_velocity.ranges.lin_vel_x` 为 **[-2, 2] m/s**，`curriculum.command_levels` 为 `null`。训练命令类包含显式 0→1 和 0.3→1 m/s 场景；重点楼梯采样档为 0.5、0.7、1.0 m/s，起步队列还覆盖 0.5～1.0 m/s。因此 0.7 是旧 MuJoCo 的 8 键档位，不能解释成旧训练的速度上限。

1.0 m/s 已经被训练覆盖，但覆盖速度值不等于学好了该速度的突然启动。旧约束对前腿横向内收、单步大目标跳变和快速俯仰不足。直接到 0.7 m/s 也能触发明显内收，说明变化率是关键原因之一。

新 FlatStart 阶段普通命令范围对齐部署限幅：前后 [-1, 1] m/s、横移 [-0.5, 0.5] m/s、偏航 [-0.6, 0.6] rad/s。实际手柄后退映射未像前向那样放大：原始 -0.49 输入仍为 -0.49；键盘 5 对应该输入。尚无本次实机后退端点的同步测量，部署代码允许的负向限幅与手柄实际端点应区分。

## 命令处理及部署状态

Python 的 `mujoco_validation/deploy/mujoco_v135/command_filter.py` 是训练及 MuJoCo 共用的普通前向升速实现；C++ `deploy_real/forward_command_ramp.hpp` 按相同公式实现并经过跨语言对拍。

- 普通升速 1.5 m/s²，0→1 约 0.67 秒；普通减速 3.333 m/s²，反向先过零。
- 侧移和偏航保持直接给定，与当前真机映射一致。
- 原始满杆终速仍为 1.0 m/s；急停、故障阻尼和策略退出路径绕过普通升速器。
- 策略重入清零升速历史；未提交动作回滚历史，避免失败推理不断推进命令。
- 新训练、验证及候选部署补丁对齐零命令停轮与处理后历史动作，维持 63→16 接口。

曾试验 1.0 m/s²：平地有效，但 26 cm 踏面、0.7 m/s 的冻结策略改变了后续四腿先后顺序并出现新提前抬后腿。因此采用 1.5。67100 在原始站立接管流程的 0.7/1.0 × 26/30 cm 四个楼梯例中保持逐阶完整、到顶与原有稳健先后顺序，未新增提前抬后腿。它是确定性小规模回归，不能代表任意初态均保持相同顺序。

候选部署补丁在：

```
mujoco_validation/deploy_real/flat_start_67100_aligned_20261123/
  forward_command.patch
  overlay/src/M20_sdk_deploy/run_policy/
  receipt.json
```

补丁尚未上机。旧的 `flat_start_67100_20261123`（1.0 m/s²）和 `flat_start_67100_a15_20261123`（未补充原始请求日志）是诊断中间产物，后续使用上面的 `aligned` 版本。上机前仍需要目标平台重新编译、完整离线执行验证及现场运动验证，不能复用旧 ELF 并认为改动已生效。

## 当前真机与候选补丁必须分别测试

MuJoCo 默认 `--control-profile current`：对齐当前未打补丁的真机，8 键**直接给 1.0 m/s**；不再用默认升速掩盖瞬态。

显式 `--control-profile flat-start`：对齐候选升速补丁。只有真机应用同一补丁后，才能把它视为实际部署路径。两种方式均记录模式、命令及动作历史到 manifest/trajectory。

当前真机模式：

```bash
cd /home/arcus/deeprobotics_m20_rl/M20_Perception_Lidar_RL
M20_PYTHON=/home/arcus/miniconda3/envs/isaaclab/bin/python \
bash mujoco_validation/run_mujoco_v135.sh \
  --checkpoint M20_perception_Lidar_rl/logs/rsl_rl/M20-V11.1-EntryTransfer/2026-11-23_13-52-59_392869_entry_transfer_resume67000_3072/model_67700.pt \
  --viewer --keyboard --control-profile current \
  --stair-height .16 --tread-depth .26
```

测试候选补丁时，仅把末尾控制版本改成 `--control-profile flat-start`。平地可增加 `--terrain-xml mujoco_validation/deploy/mujoco_v135/terrains/flat.xml`。

## 训练目标与能力保护

通过 `M20_FLAT_START_REPAIR=1` 显式启用新阶段；普通 EntryTransfer 的训练目标不变。

新代价只在真实平地列的直行中生效，使用仿真真值计算轮心在机身坐标下的横向包络、前轮支撑宽度、髋侧摆、俯仰及角速度、关节/轮速目标突变；不新增策略观测，不要求真机增加足端传感器。入阶和后续楼梯上的这项代价为零。

- 新增代价每秒上界 0.35，保留原速度跟踪、有效前进、入阶交替支撑及成功楼梯奖励，不乘低速/姿态抑制系数。
- 加速窗口覆盖从零起步、轻推到满杆以及加速后的短暂稳定阶段。
- 普通训练输入使用候选升速；flat/ascent 过渡队列中保留一组当前真机原始阶跃压力样本，防止只靠滤波隐藏策略缺陷。
- 通用教师动作模仿排除平地片段，保留 B2+ 成功支撑回放、局部更新幅度约束和所有 Actor 参数可训练。
- 只有平地退化时不加强有同样缺陷的教师模仿；真实楼梯能力退化仍保留原有有限反馈，继续学习。

当前冻结模型在候选升速下平地明显改善，但两种控制版本的严格入阶安全验收仍未通过。这一阶段用于继续修正，不能把“已经能够逐阶到顶”当作“入阶危险动作已解决”。

67100 在同一平地闭环中，0→1 m/s 的测量如下（不是实机测量）：

| 指标 | 当前直接满杆 | 候选 1.5 m/s² 升速 |
| --- | ---: | ---: |
| 俯仰峰值 | 4.38° | 0.72° |
| 右前髋侧摆峰值 | 19.44° | 3.17° |
| 右前轮向内位移 | 16.24 cm | 未出现向内位移 |
| 最终实际前进速度 | 1.095 m/s | 1.013 m/s |

## 验收、最佳模型和记录

默认每 100 轮保存，每 500 轮验收，行为失败记录并继续，不停止、不自动回滚当前 PPO。

除原历史能力保护证据外，新增两个实际物理闭环测试：

1. `flat_start/`：0→0.3、0→1、0.3→1，分别测原始阶跃与候选滤波，共 6 例；包括 pitch/rate、hipx、轮心内移、前轮宽度、目标跳变及实际前进速度。
2. `production/`：当前真机直接 1.0 m/s 的 Z→C→8，及候选补丁 0.7/1.0 m/s 的 Z→C→前进，测平地和 16 cm 高、26/30 cm 深楼梯，共 9 例。覆盖平地加速、到达 B1、前轮交替/后轮提前抬起、后续四腿顺序与逐阶支撑。

历史 0.7 m/s 证据只是低速能力保护对照，已固定原速度与原命令语义，不冒充当前满杆测试。新生产测试中的 1.0 m/s 是部署输入命令值，验收同时记录实际速度；不能仅因命令到 1.0 就声称机器人实际达到该速度。

新增测量也参与最佳模型选择，失去支撑、变慢/停滞、恶化横移偏航或改变已获得的四腿顺序不能靠平地姿态分数抵消。入阶相同时平地确有改善也可以保存为最佳候选。未执行的牵引测评不当作通过。

总 `repair_passed` 在写入 TensorBoard、`latest_metrics.json`、`metrics_history.jsonl` 和 `status.json` 前统一计算；历史能力验收通过，但平地、入阶或实际部署命令路径不通过时，总结果仍为未通过。历史能力的单独结果保留在 `native_repair_passed`。

TensorBoard：`FlatStart/<case>/...`、`Production/<control profile>/<terrain>/<tread>/<speed>/entry/...`；原有入阶、逐阶支撑和速度指标继续观察。每个验收目录保存各子目录 `manifest.json`、`results.json` 与对应日志，最佳候选归档包含相应测量证据。

## 新训练入口

默认 3072 环境、额外 50000 轮；初始化 67700，新优化器，楼梯教师/回放仍为匹配物理证据的 66800。

```bash
cd /home/arcus/deeprobotics_m20_rl/M20_Perception_Lidar_RL/M20_perception_Lidar_rl
bash scripts/train_m20_v111_flat_start_from67700.sh
```

当前旧训练仍在运行。切换由用户执行，不要同时启动第二个长期训练。

仅 CPU 预检：在上面命令末尾加 `--check-only`。冻结 Isaac 集成与梯度验证：加 `--probe-only`，不执行学习或 optimizer.step。修改轮数可用 `M20_TOTAL_UPDATES=30000 bash ...`；环境数可用 `M20_NUM_ENVS=4096 bash ...`。

新阶段续训用 `M20_RESUME_CHECKPOINT=/绝对路径/新阶段模型.pt bash scripts/train_m20_v111_flat_start_from67700.sh`，旧阶段权重应作为初始化，不继承旧优化器或过期的选择状态。原 `train_m20_v111_flat_start_from67100.sh` 保留为兼容入口，同样默认使用 67700。

## 真机日志校准

候选补丁沿用非阻塞 alignment journal，并补充 `requested_velocity3`；`obs63[6:9]` 是实际送入策略的命令。已有日志同时保存原始/处理后动作、16×5 关节目标、关节位置/速度、IMU 和同机时间。

完成候选补丁构建后，可通过已有 `M20_ALIGNMENT_TRACE_PATH` 机制采集，再离线运行：

```bash
python scripts/tools/analyze_m20_flat_trace.py \
  --trace /路径/aos_alignment.jsonl --output /路径/flat_trace_analysis.json
```

该工具拒绝不完整捕获，区分原始请求和策略命令，报告俯仰、髋侧摆、目标突变及实际策略间隔。关节目标与当前反馈之差不能直接当作电机延迟，IMU 俯仰也不是外部姿态真值。真实端到端时延、增益、摩擦和实际速度仍需运动日志校准，本次没有获得这些真机测量。

## 已完成验证

- 启动器 CPU 预检通过；冻结 Isaac 运行 40 环境 × 170 步通过，确认楼梯上的新增平地代价为零、梯度有限、Actor 全部参数可训练且权重未更新。
- C++、NumPy 和 Torch 命令升速实现对拍通过，候选部署 runner 语法编译通过，alignment journal 的 16 项既有检查通过。
- 最新平地修正及训练分段检查共 34 项通过，覆盖当前/候选部署路径均执行、行为失败继续训练和总验收指标一致性。测试时以 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 排除本机 ROS 的自动 pytest 插件依赖冲突。
- Shell/Python 语法与 `git diff --check` 通过。未启动本阶段长期训练，未在真机执行候选补丁。
