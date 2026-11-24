# 感知残差与事故冻结记录（2026-11-20）

本次更新补齐定位证据。11:25 的直接保护原因为感知源 `source_timeout/source_invalid`；GOS 累计 `registration_residual=1231`。旧运行未开启残差诊断，感知保护也没有触发事故冻结，因此不能据现有证据声称运动时失效的根因已修复。

板端编译、完整无电机验证和 AOS 实时只读预检查均已通过。稳定启动入口已切到新版；编译验收后已按用户新指令启动 AOS，PID 104435，tmux `local45200_aos_recording`，最新检查为 Idle、安全就绪、摇杆零位、无关节故障。GOS 记录版持续运行，当前启动状态见 `AOS_START_RECORDING_20261120.json`。四条触发路径的冻结、首发证据与 R2 的四份归档均已通过无电机验证。

## 新增记录

GOS 在实际执行质量校验的同一次最近邻搜索中保存 p95 及其 128 个采样点：源点、最近参考点、距离、是否找到邻点。每次校验记录源时间戳（含 sec/nanosec）、接收年龄、配对时间差、位姿、阈值、参考点数量及最老/最新帧、转发/拒绝决定。未找到邻点或无穷残差用 `null` 和显式标志表达，不能当作 0。

记录独立于 `--diagnostics off`，自动启用。生产线程只向有界队列复制数据，后台线程写入 4 个 8 MiB 循环文件；`first_rejection.json` 独立保留本次进程的首个拒绝。`source_status` 新增 `quality_checks`、`residual_p95_m`、`residual_threshold_m`、`evidence_dropped`、`evidence_write_failed`，以便识别记录缺失。记录失败不修改配准的接收/拒绝结果。

AOS 的四条感知保护路径均登记 `perception_fault`：主状态机高增益状态检查失败、RL 控制循环检查失败、策略线程的源/推理故障、最终发布时发现源故障。先发阻尼命令，再登记故障；记录器保留故障前环形数据，追加原有 2 秒事后记录后冻结，`first_fault.json` 保存首发原因。有效短时感知丢失只进入原有宽限，不触发冻结。

首发记录包括触发路径、控制状态、摇杆、IMU、感知有效性、租约与故障代数，以及最新接收报文的源序号、源时间戳、epoch、原因和接收版本。所选租约与最新报文分别记录，不能假定它们在并发回调中必然对应同一版本。

R2 仍先卸力，再将冻结记录和首发文件归档到 `incident_<session>/`，然后重新开始记录。不能只在下一次事故后读取顶层 `first_fault.json`；操作者若已按过 R2，还应查看归档目录。

## 真机位置与脚本

AOS 新 release：`/home/user/m20_native_releases/model45200_recording_20261120`。
GOS 新 release：`/home/user/m20_native_releases/model45200_recording_20261120_gos`。
AOS 事故目录：`/var/opt/robot/data/m20_joint_guard_incidents`。
GOS 当前运行和记录位置以其 `artifacts/residual_recording_session.json` 为准；文件在运行目录的 `registration_evidence/` 下。

GOS 已在运行时会自动记录，无需再启动第二套感知。停机后需要单独启动带记录的 GOS，可在 GOS 上执行：

```bash
sudo python3 /home/user/m20_native_releases/model45200_recording_20261120_gos/validation_tools/record_gos_residuals.py \
  --release /home/user/m20_native_releases/model45200_recording_20261120_gos
```

该脚本校验板端编译收据与 ELF 身份，拒绝并行启动第二个 registration，不提供任何电机控制接口。原始雷达转发和 QoS 桥仍需运行。

导出 GOS 残差或 AOS 事故目录，使用部署到两端的 `validation_tools/collect_perception_evidence.py`：

```bash
sudo python3 /home/user/m20_native_releases/model45200_recording_20261120/validation_tools/collect_perception_evidence.py \
  --input /var/opt/robot/data/m20_joint_guard_incidents \
  --output /tmp/aos_perception_incident.tar.gz
```

导出脚本包含各文件的 SHA256 台账。循环文件运行时的导出是独立文件快照；AOS 冻结后的事故及 R2 归档可作为稳定证据。

## 保留的行为

模型 45200、HipX/HipY 软件保护阈值、目标变化率、满杆输入和 R2 恢复逻辑均保持。感知短时丢失保持原策略的 2 秒宽限保留；持续失效仍中断。GOS 残差阈值、质量检查节拍与参考点云更新规则保持。

板端验证、部署状态和 ELF 身份以 `PERCEPTION_RECORDING_DEPLOYMENT_20261120.json` 为准。编译验收阶段未启动 AOS；后续按用户新指令启动，未发送手柄运动键。静止时的记录验证不代表运动时感知故障已修复。

## 可审阅源码与验证产物

本机完整 AOS 候选源码在 `artifacts/20261120/perception_recording/src/`；增量补丁 `perception_incident_recording_20261120.patch` 对应上一版 recovery95 的 `src/M20_sdk_deploy`，可用 `patch -p1` 应用。GOS 修改及新旧决定一致性测试在同一证据目录的 `gos/`，补丁为 `registration_residual_recording_20261120.patch`。

板端编译、实时预检查和四份无电机首发事故已同步到该证据目录的 `aos/`。真机 GOS 静止验证与初始残差导出包也已保存。上一轮 PID 20961 的事故环形文件已完整归档到真机事故目录的 `incident_20961_1795144897874643930/`，下次启动将使用新的顶层记录。
