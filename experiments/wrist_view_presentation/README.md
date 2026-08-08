# 腕部视觉反馈呈现实验

本目录实现唯一正式研究 `wrist_view_presentation`：在机器人平台、腕部摄像头、手机 IMU
控制、任务和模型训练配置相同的前提下，比较腕部视觉反馈与运动控制共置或空间分离时的
演示质量和模仿学习策略性能。代码不预设任一条件更优。

不可随意改变的研究设计见 [PROTOCOL.md](PROTOCOL.md)；本文只说明如何配置、运行和排错。

## 正式条件

| 条件标识 | 腕部视频显示 | 运动控制 | 反馈与控制关系 |
|---|---|---|---|
| `A_mobile_colocated` | 控制手机 | 手机 IMU | `colocated` |
| `B_desktop_separated` | 固定电脑屏幕 | 手机 IMU | `spatially_separated` |

两组都必须使用 `camera_present=true`、`camera_view=wrist`、`control=phone_imu`，并保存同源
腕部摄像头的 `video.mp4`、`video_meta.json` 和 `frame_timestamps.csv`。预检会把任何偏离作为
错误处理。

## 1. 准备本地配置

从仓库根目录运行：

```powershell
Copy-Item experiments/wrist_view_presentation/experiment_config.example.json `
  experiments/wrist_view_presentation/experiment_config.json
Copy-Item experiments/wrist_view_presentation/experiment_manifest.example.csv `
  experiments/wrist_view_presentation/experiment_manifest.csv
Copy-Item experiments/wrist_view_presentation/dh_params.example.json `
  experiments/wrist_view_presentation/dh_params.json
Copy-Item experiments/wrist_view_presentation/robot_bridge_config.example.json `
  experiments/wrist_view_presentation/robot_bridge_config.json
```

正式文件已被 Git 忽略。不要提交参与者数据、真实硬件配置、凭据、checkpoint 或输出。

必须按预注册内容替换以下占位项：

- `success_definition` 中的放置误差和超时阈值；
- `tasks` 中简单、中等、困难三档真实任务；
- `analysis.planned_participants`；
- 实测 `capture.max_clock_uncertainty_s`；
- 真实关节顺序、DH 参数、坐标系、安全边界和驱动时限；
- `training.condition_runs` 中的本地数据集 ID 和共同训练参数。

`training.condition_runs` 的两份配置除 `dataset_repo_id` 外必须逐字段相同。正式预检会验证
这一点，流水线也从同一份已验证配置启动两组训练。

## 2. 运行协议与清单预检

```powershell
uv run --extra training python experiments/wrist_view_presentation/validate_experiment_setup.py `
  --config experiments/wrist_view_presentation/experiment_config.json `
  --manifest experiments/wrist_view_presentation/experiment_manifest.csv
```

示例配置故意保留研究者必须决定的阈值和样本量，因此不能直接通过正式预检。不要用方便的
临时值绕过预注册。

清单中每个 `pair_id` 必须恰好有一条 `A_mobile_colocated` 和一条
`B_desktop_separated`，二者的 `participant_id`、任务、难度和试次编号必须一致；同一参与者
在所有试次中保持同一条件顺序，参与者之间平衡 A-first/B-first。

## 3. 采集目录与硬性数据契约

硬件采集适配器必须发布如下目录：

```text
raw_ts/
  A_mobile_colocated/episode_NNN/
    robot.csv
    phone.csv
    applied_actions.csv
    video.mp4
    video_meta.json
    frame_timestamps.csv
  B_desktop_separated/episode_NNN/
    ...相同文件...
```

- `robot.csv` 使用 `timestamp,<joint_names...>`；单位为秒和弧度。
- `phone.csv` 保存收到的全部 6D 手机 IMU 增量，仅用于审计。
- `applied_actions.csv` 只保存机器人实际执行的 6D 增量，训练只读该文件。
- `video_meta.json` 必须由真实解码帧录像器声明
  `artifact_type=video_episode_capture`、`writer_capability=decoded_frame_video_recorder` 和
  `video_capture_verified_by_this_writer=true`。
- `frame_timestamps.csv` 必须逐真实帧记录源时钟和 Unix 映射；不得用名义帧率推造时间戳。
- 两组都记录视频延迟、掉帧和时钟不确定度，后续作为协变量或质量控制量。

采集后运行带数据的预检；它会读取三路 CSV、核对元数据、逐帧时间戳和真实可解码帧数，并
检查各流时间交集：

```powershell
uv run --extra training python experiments/wrist_view_presentation/validate_experiment_setup.py `
  --config experiments/wrist_view_presentation/experiment_config.json `
  --manifest experiments/wrist_view_presentation/experiment_manifest.csv `
  --raw_root raw_ts
```

## 4. 正式流水线

`run_pipeline.ps1` 严格按以下顺序执行：采集门禁 → 时间同步 → LeRobot 转换 → 参与者安全
划分 → A/B 多种子训练 → 统一测试集比较 → 特征比较 → 真实机器人 rollout。

```powershell
.\experiments\wrist_view_presentation\run_pipeline.ps1 `
  -Task "与协议 task_id 对应的任务描述" `
  -CollectionScript "<硬件采集适配器.ps1>" `
  -RolloutScript "<统一真实机器人评估适配器.ps1>"
```

已有合格采集数据时可省略 `-CollectionScript`，脚本仍会执行完整数据预检。真实机器人适配器
尚未提供时，脚本会在完成离线阶段后明确失败，不会把训练 loss 或离线误差冒充 rollout
证据。适配器参数可通过 `-CollectionArguments` 和 `-RolloutArguments` 传入。

主要输出位于 `outputs/wrist_view_presentation/`，包括冻结划分、每个条件/种子的 checkpoint
映射、统一测试集误差、演示特征比较和复现元数据。`outputs/` 已被 Git 忽略。

## 5. 保留的通用实现

| 文件 | 职责 |
|---|---|
| `acquisition_interfaces.py` | 版本化手机动作、夹爪扩展、视频逐帧时钟契约 |
| `robot_bridge.py` | FK/IK、运动约束、watchdog 和 `RobotDriver` 接口 |
| `time_sync.py` | 按真实时间戳同步，并按顺序组合 SE(3) 增量 |
| `convert_raw_to_lerobot.py` | 全量预检、流式视频解码和原子数据集发布 |
| `make_episode_splits.py` | 按参与者隔离且按难度档案分层的划分 |
| `offline_compare.py` | 冻结测试集上的多种子物理单位误差比较 |
| `feature_extraction.py` | 速度、加速度、jerk、修正动作、轨迹和结果特征 |
| `features_compare.py` | 参与者级配对统计、效应量、FDR 和条件×难度模型 |
| `validate_experiment_setup.py` | 协议、配对、数据文件、视频和时钟的正式门禁 |

## 6. 尚待真实硬件集成

- 实现具体机械臂的 `KinematicsProvider`、`RobotDriver` 和独立急停通道；
- 将手机控制 DataChannel、腕部 WebRTC 解码帧录像器和条件显示端接入采集适配器；
- 冻结夹爪动作/状态 schema，并接入采集、同步和转换；
- 完成低速安全联调、试采集、真实 A/B 数据、多种子训练和统一 rollout；
- 为 rollout 适配器记录成功、完成时间、碰撞、放置误差和未见场景表现。

这些事项完成前，仓库只证明软件契约和离线流程可测试，不证明实机闭环完成，也不支持关于
任一条件优劣的结论。
