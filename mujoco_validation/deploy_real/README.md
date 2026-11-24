# 1516 原生部署迁移到 1515

**2026-11-20 最新启动状态：** 新版 AOS 已启动，PID **20961**，root tmux `local45200_aos_recovery95`；启动后核对为 Idle、`safe_mode=0`、摇杆零位、无关节故障。机器人重启后的原始包转发及 GOS 三个感知会话已恢复，正式启动前 999/999 条有效地形观测通过。手柄使用 R2 → L1，站稳后 L2；本次未发送手柄运动键。见 `AOS_START_RECOVERY95_20261120.json`。

**2026-11-19 部署完成时状态：** HipY 95 Nm、R2 卸力/重新起立和满杆响应修改已部署，原生编译、完整无电机验证与实时预检查均通过。当前自定义 AOS **停止**。45200 入口指向 `model45200_recovery95_20261119`，详见 [本次恢复说明](RECOVERY_FEATURES_20261119.md) 与 `RECOVERY_FEATURES_DEPLOYMENT_20261119.json`。以下启动记录为历史。

本目录放置真实部署工具。MuJoCo 继续使用 `../deploy/mujoco_v135`。
本机迁移、45200 原始配置导出与本轮输入链检查见 [LOCAL_MIGRATION_20261119.md](LOCAL_MIGRATION_20261119.md)。
关节保护和常驻事故记录的接线、参数来源与边界见 [JOINT_SAFETY.md](JOINT_SAFETY.md)。
当前模型为45200，切换记录见 `MODEL_45200_DEPLOYMENT_20261119.json`。
44600阶段记录保留在 `MODEL_44600_DEPLOYMENT_20261118.json`。
42900阶段保护与感知验证记录保留在 `JOINT_SAFETY_DEPLOYMENT_20261118.json`。
本次启动记录器和默认趴姿膝边界修正见 `JOINT_SAFETY_REPAIR_20261119.json`。
短时遮挡继续运行策略的接线和验证边界见 [PERCEPTION_GRACE.md](PERCEPTION_GRACE.md)。
承重转向触发保护的记录与转向起步渐变见 [YAW_RAMP.md](YAW_RAMP.md)。
2026-11-19 前进触发右后 HipX 保护后，已停止该控制会话；按用户指定旧仓库
修正逐腿范围与策略目标处理，见 [本次限位修正](REFERENCE_LIMITS_20261119.md)。
原始迁移过程保留在 `DEPLOYMENT_20261118.json`。

参考机读取路径为办公室 Wi-Fi → NX → AOS → GOS，全程只读。
当前参考 AOS 是 `current35300_20261113_receiver_reset`，
GOS 是 `gos_pose_recheck_20261113`，使用 `/M20_V11_STAIRCASE_PACKET`。
参考机当前记录属于部署/构建记录，不能当作本轮实机动作成功证明。
旧 `m20_stair_releases/10151515_aos` 和 V4 高程图程序不是当前参考。

跨机移植先核对 current 链接、实际 ELF、源码清单与模型张量；分别记录
构建、无电机运行和实机运动验收。目录存在或旧文档称为 baseline，不能代替
这些证据。同网段两台机器人可能使用相同 IP，写入与启动必须独立核验 MAC。

1515 的独立目录：

- AOS 关节保护版：`/home/user/m20_native_releases/model42900_20261118_jointguard_aos`
- AOS 当前45200入口：`/home/user/m20_native_releases/model45200_20261119_jointguard_aos`（指向 `model45200_recovery95_20261119`；上面的42900目录保留为修改前回退版本）
- GOS：`/home/user/m20_native_releases/model42900_20261118_gos`

`switch_native_weights.py` 按用户要求沿用同一训练的执行契约，仅替换Actor张量。
本次跳过配置重审、全量回放、重新编译和运动测试；仅验证转换输出、板端加载及文件完整性。
切换前取得控制锁并拒绝正在运行的自定义控制器，备份旧策略与清单后更新；异常时恢复备份。
控制ELF、关节边界台账、事故记录、GOS程序与传感器会话均保留。
板端当前权重记录为 `MODEL_WEIGHT_SWITCH.json`，被替换的44600备份位置记录在其中。
启动命令见 [START_COMMANDS.md](START_COMMANDS.md)，现场启动时的原有预检查和人工确认仍须通过。

`export_native_actor.py` 逐张量证明参考 ONNX 与本地原始 35300 一致，
保留 ONNX 图/opset/输入输出，只更换 Actor，并用历史真机观测比较输出。
它保存配置副本、哈希、元数据及生成的执行参数头。
42900 使用明确指定的兼容配置；未找到日志内字节相同的来源副本，
配置来源证明的边界在导出报告中保留。

`prepare_1516_release.py` 只接受精确匹配的当前参考源码与 ONNX。
它将训练默认姿态接入观测、关节目标、站起结束姿态和元数据检查。
裁剪遵循配置，原非有限检查及原始目标溢出拒绝在裁剪前执行。
观测租约、动作提交/回滚、来源故障、遥控、恢复、SDK 与急停相关逻辑保留。
本轮编译完整 AOS 控制包，复用字节未改的 drdds 依赖。

`build_1516_aos.sh` 只编译及运行无电机测试。资源依据来自参考机
`common.hpp::budget`：每个编译器受限，额外保留内存；启动前检查真实余量。
`make_native_fixture.py` 与 `native_contract_replay.cpp` 检查默认姿态、
关节顺序、裁剪、缩放、动作历史与零指令轮速门控。
数值/保护测试通过不代表楼梯或降噪验收通过。

GOS保留参考机reader/sender/registration/LIO算法二进制。LIO仅将两个输入话题改到本地QoS适配入口；本地发现配置补入loopback。详见 [LIDAR_INPUT_REPAIR_20261118.md](LIDAR_INPUT_REPAIR_20261118.md)。
消息依赖放入独立目录，原生入口只改为从该目录加载消息库。
`gos_main.cpp` 仅包含原 GOS 运行/停止入口，避免引入 GOS 不使用的 ONNX 包装依赖。
1516 的传感器外参是参考值，不能记作已在 1515 标定。

连接参考机：

```bash
ssh nvidia@192.168.0.156
# NX 上进入 AOS，再由 AOS 进入 GOS；不要绕过 AOS 直连 GOS。
ssh user@10.21.31.103
ssh user@10.21.31.104
```

本机连接 1515 AOS 的热点地址，并保留主机密钥检查：

```bash
ssh -tt -o HostKeyAlias=10.21.31.103 user@10.21.41.1
```

本机连接 1515 GOS：

```bash
ssh -tt -o 'ProxyCommand=ssh -o HostKeyAlias=10.21.31.103 user@10.21.41.1 -W %h:%p' user@10.21.31.104
```

启动命令在现场前置检查完成后，由操作员在板端 root tmux 内执行。
AOS 保留原工具要求的 `SDK_READY` 和 `AOS_RUN`，不能自动代填。
确切命令及当前检查结果会记录在部署记录中。尚未通过实时检查时不能宣布可运动。

回退只切换本次独立目录中的自定义程序；先停止当前自定义控制器，不能并行
启动另一控制器。厂商服务和参考机未在本轮修改。此前在 1515 创建的旧参考
暂存版已禁用入口，不能用作回退基准。

本机检查：

```bash
conda run -n isaaclab-5.1 python -m unittest discover -s mujoco_validation/deploy_real -p 'test_*.py' -v
```

GOS 重启后的同型号大核可能具有不同的 `cpu_capacity`。资源判定同时核对
ARM MIDR 型号、在线状态及相对小核算力，不能只用“等于最大算力”识别大核。
`repair_gos_cpu_topology.py` 验证板端身份、原源码与真实运行 CLI 后，单独编译
GOS 入口；不改角色 CPU 列表、优先级或安静服务列表。源码目录保留的参考
`main.cpp` 未必就是 ELF 的实际入口，新增构建明确保存 `gos_only_main.cpp`。

保护版发布由 `finalize_joint_guard_release.py` 核对完整校验清单、ELF、数值台账、
全部无电机结果及板端身份后进行。它只切换启动文件，不启动控制器。
旧启动入口转发到保护版；旧无保护二进制仅作追溯，不能手工绕过启动器运行。

1515传感器网与GOS板间网不同，最终采用原始UDP载荷转发和GOS本地解码。完整点云直接跨板发送的调试路径未采用。GOS启动线程核验使用有界等待，所有资源与来源门控保留。启动恢复命令见 [START_COMMANDS.md](START_COMMANDS.md)。
