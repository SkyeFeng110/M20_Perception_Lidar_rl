# 本机部署迁移与 45200 检查

**2026-11-20 最新启动状态：** 新版 AOS 已启动，PID **20961**，root tmux `local45200_aos_recovery95`；启动后核对为 Idle、`safe_mode=0`、摇杆零位、无关节故障。机器人重启后的原始包转发及 GOS 三个感知会话已恢复，正式启动前 999/999 条有效地形观测通过。手柄使用 R2 → L1，站稳后 L2；本次未发送手柄运动键。见 `AOS_START_RECOVERY95_20261120.json`。

**2026-11-19 部署完成时状态：** HipY 95 Nm、R2 卸力/重新起立和满杆响应修改已部署，原生编译、完整无电机验证与实时预检查均通过。当前自定义 AOS **停止**。45200 入口指向 `model45200_recovery95_20261119`，详见 [本次恢复说明](RECOVERY_FEATURES_20261119.md) 与 `RECOVERY_FEATURES_DEPLOYMENT_20261119.json`。以下启动记录为历史。

**历史状态：** 四条腿 HipX 已统一为 **0.65 rad**，新版 AOS 已启动，PID **231831**，tmux `local45200_aos_allhipx065`。编译、完整无电机验证和预检查通过，当前 Idle、安全就绪、摇杆零位、无关节故障。感知 2 秒宽限保留，真机运动尚未复测。启动记录见 `AOS_START_ALL_HIPX065_20261119.json`。

**历史状态：** 限位修正版已按用户要求启动，PID 80931，tmux
`local45200_aos_ref_limits`；当前检查为 Idle、安全就绪、摇杆零位、无关节故障。
见 `AOS_START_REFERENCE_LIMITS_20261119.json`。修复后尚未做运动验收。
16:01 的前进操作曾触发右后 HipX `target_feedback_error`，事故已归档，
原 PID 21129 已停止。下方 PID 21129 的 Idle/启动描述保留为故障前记录。
修正细节见 [限位修正](REFERENCE_LIMITS_20261119.md)。

本次用户报告：GOS 独立雷达解码持续 `MSOPTIMEOUT`，导致 AOS 无法启动；要求把 mark 主机上的部署迁到本机并恢复启动。

## 已完成

- 来源：`mark@192.168.0.234:/media/mark/2T_SSD/M20_perception_Lidar_rl_2/M20_Perception_Lidar_RL/mujoco_validation/deploy_real/`。
- 工具、源码归档、原部署产物和事故证据均迁入本目录。传输 13096 个文件、938114101 字节，排除 Python/pytest 缓存；传输后 rsync 按内容校验，无差异。没有删除或覆盖远端文件。
- 导入源码清单：`artifacts/local_migration_20261119/imported_source_manifest.json`。原有部署记录是历史证据，不表示本次机器人实时状态。
- 本机 Python：仓库根目录 `.venv-m20-real/bin/python`，共享已有 `isaaclab` 环境依赖，单独安装 `onnxruntime==1.32.0` 和 `flatbuffers==25.12.19`。没有更改正在训练的环境。
- 9 项现有 Python 单元检查通过。
- 本次 checkpoint SHA-256 与远端 45200 记录一致；迁入 ONNX 的 8 个 Actor 张量与本地 checkpoint 逐张量完全一致。
- 远端保留的旧训练 YAML 与本次原始 YAML 字节不同，但实际观测配置、感知配置、默认角、动作尺度、PD、力矩上限、armature、控制周期全部相同。
- 用本次 checkpoint 相邻的原始训练配置重新导出，6724 条历史观测 ONNX/PyTorch 对照通过，最大绝对误差 `5.7220458984375e-06`；没有样本触发动作裁剪。接口为 63 维输入、16 维输出，感知契约 `v11_signed_edges_v2`。

本机完整导出位于 `artifacts/local_migration_20261119/policy_45200/`，包括模型、原始 YAML 副本、执行参数头和数值验证报告。该模型尚未替换到机器人；迁入的旧模型继续保留在 `artifacts/20261119/policy_45200/`。两者 SHA-256 不同，因为导出元数据包含来源路径等信息，Actor 张量相同。

## 本机命令

从仓库根目录执行已有检查：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv-m20-real/bin/python -m unittest discover \
  -s mujoco_validation/deploy_real -p 'test_*.py' -v
```

复现导出时，`--output` 必须是尚不存在的目录：

```bash
run=M20_perception_Lidar_rl/logs/rsl_rl/M20-V11.1-RecoveryRefine/2026-11-18_14-59-56_032979_refine_30000_from43800
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv-m20-real/bin/python mujoco_validation/deploy_real/export_native_actor.py \
  --baseline-onnx mujoco_validation/deploy_real/artifacts/20261118/1516_reference/aos/policy/model.onnx \
  --checkpoint "$run/model_45200.pt" --config-dir "$run/params" \
  --output mujoco_validation/deploy_real/artifacts/local_migration_20261119/policy_45200_recheck
```

导出器沿用的 `config_provenance` 提示是较早迁移的保守说明。本次已直接使用 checkpoint 相邻的原始 `params/`，其文件哈希见本次报告。

## 本机连接与断点

用户已将 USB 网卡插到本机并连接 `CA9B_NO.1515_24G`。本机经 `10.21.41.1` 连接 AOS，再由 AOS 跳转到 GOS；办公室网络仍由另一张网卡提供。SSH 主机公钥与 mark 主机保存的公钥核对。板端身份已实测：AOS 序列号 `4c7428ea7d51f4b`、eth2 MAC `00:e0:9a:4a:ff:8d`；GOS 序列号 `6a3df90cc64ca70b`、eth0 MAC `46:40:20:07:09:44`。

用户在检查期间重启了 1515，因此重启前后的结果分别保留。重启后原始包转发和 GOS 自定义感知会话均未运行。三张 AOS 有线网卡未收到已知雷达端口的数据，前后雷达邻居地址均不完整；显式加入前后雷达组播组、订阅 8 秒也收到 0 包。此时不能归因于 ONNX 或 GOS 解码算法。

用户随后确认休眠并将休眠时间调整为 30 分钟。另一个可直接验证的因素是原厂配置 `Device.lidarPower = [0,0]`，启动日志于 15:34:13 记录 `SetLIDAR_POWER success, lidar_power:0x0`。延长休眠时间后，仍无雷达数据。

通过原有 `/LIDAR/POWER` 服务恢复雷达供电后，前雷达 MAC `08:48:57:19:f4:be` 和后雷达 MAC `08:48:57:1a:b6:db` 重新出现，恢复 AOS 原始包转发后，GOS 独立解码开始输出点云。供电请求编码依据板端 `DeviceControl::SetLIDAR_POWER` 的实际实现核对：前后设备分别使用低/高 16 位，启用值为 1。请求为 65537，返回为 0；最初的临时客户端错误地按“回显请求”判断响应，因而报错，后续以实际 MAC、点云与有效观测确认供电恢复，没有重复发送该请求。原始回执保留在 `artifacts/local_migration_20261119/lidar_power_service_receipt.json`。

SDK 启用、休眠及雷达供电是不同状态；没有证据证明打开 SDK 会直接关闭雷达。原厂更新记录也记载过“休眠模式不关闭雷达”，不能只凭指示灯和时间接近判断具体关闭原因。

## 已恢复的链路与本次证据

本次从本机恢复的板端 root tmux 会话：

- AOS：`local45200_raw_lidar`。
- GOS：`local45200_front_lidar`、`local45200_lio_qos`、`local45200_gos`。

8 秒只读链路采样：`/LIDAR/POINTS` 收到 61 帧非空点云；QoS 适配、LIO 点云及里程计各 80 帧；六维感知包收到 400 条，全部 `reason=ok`。原始证据为 `artifacts/local_migration_20261119/gos_chain_restored.json`。

AOS 完整无电机预检查通过：999/999 条有效观测接受、连续健康检查 9177 次通过、关节与 IMU 有限且新鲜、原厂进程身份通过、遥控被动观察通过。证据为 `artifacts/local_migration_20261119/aos_preflight_restored.log`。此前一次失败来自进程扫描中的瞬时 `/proc/<pid>/comm` 读取异常，串行复查越过该项后明确暴露感知缺失；两次失败日志均保留，没有把它们算作通过。

板端已部署的 45200 ONNX 哈希为 `42558572cc833e278aebb3793ba3ca600395e3d0445658b07d400fe5bd61f594`，控制器 ELF 哈希为 `f084b8cef134678afc4e3dbb3b07a50003743814d33d83696d2124d59ad2a1dd`。本机重新导出的模型用于迁移核验，本次继续使用板端已安装且逐张量一致的模型。

## 恢复链路的检查顺序

完整板端命令见 [START_COMMANDS.md](START_COMMANDS.md)，此前修复依据见 [LIDAR_INPUT_REPAIR_20261118.md](LIDAR_INPUT_REPAIR_20261118.md)。先核对实际板端序列号/MAC和现有会话，避免重复启动。

1. AOS 传感器接口是否收到前雷达的 MSOP、DIFOP、IMU 原始包。
2. AOS `relay_lidar_packets.py` 是否仍运行，来源/目的身份及转发计数是否正确。
3. GOS 独立解码器是否在接收台账规定的转发端口；应区分本次独立进程和仍可能超时的原厂旧端口驱动。
4. `/LIDAR/POINTS` 和 `/IMU`、QoS 适配、LIO、registration、观测转发是否连续有效。
5. AOS 的 `aos-preflight` 是否实际接受新鲜有效观测；通过后再由板端正式启动器启动 AOS。

用户在预检查通过后明确确认现场支撑、急停监督、摇杆零位和 SDK 就绪，并要求启动 AOS。正式启动结果单独记录；本次没有进行行走、转向或楼梯运动验收。

## 正式启动结果

AOS root tmux 会话 `local45200_aos` 已启动，控制器 PID 为 21129。正式启动器再次接受 997/997 条有效观测，连续健康检查 9539 次通过，之后才启动控制器。记录器当前会话为 `21129_1795074613484217508`，状态为 Idle、摇杆全零、`safe_mode=0`、无关节故障锁存，等待现场手柄操作。旧的 `first_fault.json` 属于此前记录，不能仅凭文件仍存在判断本轮故障。

启动后用户补充说明先前忘记开启手柄 SDK。随后板端日志于 15:53:56 明确记录 `set SDK Enable: 200 success`，复查本会话仍为 Idle、无关节故障或新事故触发，因此未重启正在运行的 AOS。该实际启用时点已补入启动结果，不能用此前的现场确认代替 SDK 成功证据。

雷达供电恢复后，将原厂 `Device.lidarPower` 保存为 `[1,1]`，供下次开机启用雷达；保留用户设置的 30 分钟自动休眠。原文件备份在 AOS `/var/opt/robot/data/m20_lidar_config_before_local_20261119.toml`。配置保存时曾出现换行处理错误，已从备份重新生成，逐行核对仅雷达启用值改变，并通过板端和本机 TOML 解析。没有为验证该设置重启正在运行的机器人。

本轮结果总表见 [LOCAL_STARTUP_20261119.json](LOCAL_STARTUP_20261119.json)，当前状态与控制台快照见 `artifacts/local_migration_20261119/aos_started.json` 和 `aos_started_console.txt`。

从本机查看正在运行的 AOS：

```bash
ssh -tt -o HostKeyAlias=10.21.31.103 user@10.21.41.1 \
  'sudo tmux attach -t local45200_aos'
```

Ctrl-B 再 D 可退出视图并保留运行。查看当前会话不必重新启动控制器。
