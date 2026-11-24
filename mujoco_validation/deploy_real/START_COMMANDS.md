# 1515 保护版与感知链启动

**2026-11-20 最新启动状态：** 四膝 85 Nm 新版 AOS 已按用户指令启动，PID **178565**，root tmux `local45200_aos_knee85`。正式预检查接受 999/999 帧，9,587 次健康检查通过。启动后检查为 Idle、safe_mode=0、摇杆零位、无关节锁存、无首发故障；AOS 事故记录及 GOS 残差记录正常。本轮未发送运动按键。启动收据：[AOS_START_KNEE85_20261120.json](AOS_START_KNEE85_20261120.json)。以下状态为历史快照。

**2026-11-20 本次启动前的部署状态：** 四条腿 Knee 误差代理阈值已按用户要求改为 **85 Nm**，板端原生编译、完整无电机验证及实时只读预检查通过。稳定入口 `model45200_20261119_jointguard_aos` 已指向独立新版 `model45200_knee85_20261120`。旧自定义 AOS PID **104435** 已在 Idle 状态停止，当时自定义 AOS **未启动**。GOS 已保留旧故障会话并重建记录会话，最新检查为感知 active、记录正常；持续拒绝根因尚未修复。驱动实际总扭矩限制未核实，85 Nm 是软件准入门槛，不能视为硬件安全额定值。详见 [四膝 85 Nm 更新与核查](KNEE85_20261120.md) 和 [部署收据](KNEE85_DEPLOYMENT_20261120.json)。

**本次更新前的事故状态：** 用户上楼试验于 12:34:28 触发左前 Knee 跟踪误差保护，首发记录及冻结历史已由 R2 归档；GOS 于 12:36:44 开始独立的残差持续拒绝。事故结论和证据见 [上楼事故分析](STAIR_STOP_RECORDING_20261120.md)。

**2026-11-19 部署完成时状态：** HipY 95 Nm、R2 卸力/重新起立和满杆响应修改已部署，原生编译、完整无电机验证与实时预检查均通过。当时自定义 AOS **停止**，当时 45200 入口指向 `model45200_recovery95_20261119`，详见 [本次恢复说明](RECOVERY_FEATURES_20261119.md) 与 `RECOVERY_FEATURES_DEPLOYMENT_20261119.json`。以下启动记录为历史。

**历史状态：** 四条腿 HipX 已统一为 **0.65 rad**，新版 AOS 已启动，PID **231831**，tmux `local45200_aos_allhipx065`。编译、完整无电机验证和预检查通过，当前 Idle、安全就绪、摇杆零位、无关节故障。感知 2 秒宽限保留，真机运动尚未复测。启动记录见 `AOS_START_ALL_HIPX065_20261119.json`。

最新：限位修正版已按用户指令启动，tmux `local45200_aos_ref_limits`，PID 80931。
最新检查为 Idle、安全就绪、摇杆零位、无关节故障；启动记录见
`AOS_START_REFERENCE_LIMITS_20261119.json`。下方停止描述为修复前历史。

2026-11-19 16:01 前进试验触发右后 HipX 保护，原 PID 21129 已归档并停止。
当前修复状态以 [限位修正记录](REFERENCE_LIMITS_20261119.md) 和
`REFERENCE_LIMITS_DEPLOYMENT_20261119.json` 为准。以下启动成功描述是故障前记录。

2026-11-19 本机迁移后已恢复感知链并正式启动 AOS。本轮 AOS 会话为 `local45200_aos`，AOS 原始包转发为 `local45200_raw_lidar`；GOS 会话为 `local45200_front_lidar`、`local45200_lio_qos`、`local45200_gos`。查看与恢复时先核对已有会话，避免重复启动。实测记录及本机 SSH 命令见 [本机迁移记录](LOCAL_MIGRATION_20261119.md)和[本轮启动结果](LOCAL_STARTUP_20261119.json)。以下旧会话名保留作历史说明。

当前模型已切换为45200；AOS保护版目录沿用原42900命名，实际加载的是45200权重。
`model45200_20261119_jointguard_aos` 是同一目录的便捷入口，旧启动脚本和44600入口也使用新权重。
本次按用户要求仅替换权重，保留关节保护、事故记录、控制参数与GOS感知链，未重新编译或运行全套验证。
本次切换记录见 `MODEL_45200_DEPLOYMENT_20261119.json`，板端对应 `MODEL_WEIGHT_SWITCH.json`。
后续记录器启动等待及默认趴姿膝边界修正见 `JOINT_SAFETY_REPAIR_20261119.json`，不改变模型权重。
短时感知缺失的降级规则见 `PERCEPTION_GRACE.md`，实际安装状态以 `PERCEPTION_GRACE_DEPLOYMENT_20261119.json` 为准。
转向起步渐变及最新启动会话见 `YAW_RAMP_DEPLOYMENT_20261119.json`。
本轮只做权重部署和无电机加载检查，没有启动自定义运动控制。
点云断点、修复路径和验收边界见 `LIDAR_INPUT_REPAIR_20261118.md`。
最新运行结果见 `JOINT_SAFETY_DEPLOYMENT_20261118.json`；不能把感知恢复当作已完成行走验收。

## 当前会话

AOS原始包转发、GOS前雷达解码、QoS适配与感知主程序均使用板端root tmux。
2026-11-20 已复验带残差记录的 GOS 感知链。已运行的会话不要重复启动；先用下面命令查看：

```bash
sudo tmux list-sessions
```

电脑SSH断开或切换Wi-Fi后，板端会话继续运行；机器人内部传感器和板间网络仍须正常。
这些程序未配置开机自启。机器人重启后，按以下顺序恢复。
身份参数对应本轮实际核验的1515，不能直接用于相同IP的1516。

## 恢复AOS原始包转发

在1515 AOS上进入root终端，核对传感器和GOS当前地址及MAC，再启动：

```bash
sudo -s
cd /home/user/m20_native_releases/model42900_20261118_sensor_aos
stamp=$(date +%Y%m%d_%H%M%S)
tmux new-session -d -s model42900_raw_lidar \
  "exec python3 '$PWD/relay_lidar_packets.py' \
  --expected-mac 00:e0:9a:4a:ff:8d --expected-serial 4c7428ea7d51f4b \
  --sensor-interface eth0 --board-interface eth2 \
  --source-address 10.21.33.201 --source-mac 08:48:57:19:f4:be \
  --destination 10.21.31.104 --destination-mac 46:40:20:07:09:44 \
  --seconds 0 --output '$PWD/artifacts/raw_$stamp' \
  > '$PWD/artifacts/raw_$stamp.launch.log' 2>&1"
```

启动器同时核对本板序列号/MAC及两个邻居的MAC；地址变化时按实测结果填写。
不修改原厂服务、配置、路由或网卡队列。

## 恢复GOS解码、QoS适配与感知

本机必须经过1515 AOS跳板进入GOS。在GOS的root终端执行：

```bash
sudo -s
cd /home/user/m20_native_releases/model42900_20261118_sensor_gos
stamp=$(date +%Y%m%d_%H%M%S)
tmux new-session -d -s model42900_front_lidar \
  "exec python3 '$PWD/run_lidar_source.py' \
  --expected-mac 46:40:20:07:09:44 --expected-serial 6a3df90cc64ca70b \
  --sensor-interface eth0 --dds-interface lo --relay-input \
  --seconds 0 --output '$PWD/artifacts/source_$stamp' \
  > '$PWD/artifacts/source_$stamp.launch.log' 2>&1"
tmux new-session -d -s model42900_lio_qos \
  "bash -lc 'source /opt/ros/foxy/setup.bash; \
  export FASTRTPS_DEFAULT_PROFILES_FILE=/home/user/m20_native_releases/model45200_recording_20261120_gos/stage/direct_udp.xml; \
  exec python3 $PWD/bridge_lio_qos.py --seconds 0 \
  --output $PWD/artifacts/qos_$stamp.json \
  > $PWD/artifacts/qos_$stamp.launch.log 2>&1'"
cd /home/user/m20_native_releases/model45200_recording_20261120_gos
tmux new-session -d -s local45200_gos \
  "exec python3 '$PWD/validation_tools/record_gos_residuals.py' \
  --release '$PWD' --output '$PWD/artifacts/live_$stamp' \
  > '$PWD/artifacts/live_$stamp.launch.log' 2>&1"
```

查看会话用 `tmux attach -t 会话名`，离开视图用Ctrl-B再D。
停止程序用Ctrl-C；停止感知后AOS仍按现有新鲜度和故障门控处理。
原厂GOS旧端口驱动可能继续报告超时，本链路使用独立端口，检查新会话和验收产物。

## AOS无电机预检查与现场启用

在1515 AOS先执行：

```bash
cd /home/user/m20_native_releases/model45200_20261119_jointguard_aos
sudo -H bash launch_aos.sh aos-preflight 00:e0:9a:4a:ff:8d
```

完整预检查通过、现场支撑与急停监督就绪、摇杆归零并确认SDK模式后，
由操作员在AOS交互终端进入板端root tmux并执行：

```bash
sudo -H tmux new-session -s model45200_aos
cd /home/user/m20_native_releases/model45200_20261119_jointguard_aos
bash launch_aos.sh aos-run 00:e0:9a:4a:ff:8d
```

程序保留原提示，要求现场操作员输入 `SDK_READY` 和 `AOS_RUN`。
关节保护及事故记录见 `JOINT_SAFETY.md`。物理外参、零位/方向、驱动限位及真实步态尚待验收。
