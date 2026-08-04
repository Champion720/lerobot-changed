# 视频显示端对照实验（Mobile vs PC Display）

量化“机械臂摄像头视频流显示在手机端还是电脑端”对遥操作数据质量和模仿学习质量的影响。两组数据都来自真实采集，且都包含机械臂摄像头视频；差异不再是“有无摄像头”，而是操作者观看视频反馈的终端不同。

> 目录名仍叫 `camera_ablation` 是历史遗留。当前研究问题应表述为：**同一机械臂摄像头视频流，在手机端显示与电脑端显示两种条件下，哪一组遥操作数据训练出的策略学习质量更高。**

本工程已固定这一设计：A/B 都有摄像头视频、都用手机 IMU 控制，唯一主要实验变量是
视频显示终端。不要再把旧版“无摄像头 vs 有摄像头”烟雾测试结果写进正式实验。

## 实验设计

| 条件 | 视频流显示方式 | 操控方式 | 数据集内容 | 训练的 ACT 输入 |
|------|----------------|----------|------------|----------------|
| **A：Mobile display** | 机械臂摄像头视频流经服务器发送到手机客户端，用户在手机上观看末端执行器位置 | 手机控制末端执行器位置 | state + action + camera video | `observation.state` + `observation.images` |
| **B：PC display** | 机械臂摄像头视频流经服务器发送到电脑端，用户在电脑屏幕上观看末端执行器位置 | 手机控制末端执行器位置 | state + action + camera video | `observation.state` + `observation.images` |

控制变量建议：

- 两组使用同一机械臂、同一摄像头、同一任务、同一操作者或平衡操作者顺序。
- 两组保持相同 `fps`、视频分辨率、任务时长、episode 数、训练步数和模型配置。
- 记录每条 episode 的显示端、视频延迟、掉帧情况、分辨率、操作者、任务难度，后续可作为协变量或分组变量。
- 旧版“条件A无视频/删视频列”的方案只适合作为占位烟雾测试，不再作为论文实验设计。

## 数据语义

```
action = [Δx, Δy, Δz, ΔYaw, ΔPitch, ΔRoll]            手机发出的 6 维末端 delta
state  = [关节角 j1..jN] ⊕ [x, y, z, Yaw, Pitch, Roll]  关节角 + FK 算出的末端位姿
video  = 机械臂摄像头视频流录制结果，A/B 都需要保存
```

单位：位置米、角度弧度（ZYX）。机械臂只输出关节角时，末端位姿由转换器用 DH 表做 FK。

## 完成实机框架还需要提供什么

软件侧的通用接口、数据契约、同步、转换和预检已经搭好；当前缺少的是与具体设备绑定的
资料和实现。后续实验人员至少需要提供下面这些内容：

1. **机械臂与控制器资料**：机械臂/控制器准确型号，厂商 SDK 或现有控制代码的本地路径，
   通信方式（CAN、串口或 TCP）以及连接、使能、清故障、读取关节角、下发关节目标、停止和
   物理急停 API。账号、令牌和密钥不得写入仓库。
2. **现有 FK/IK 代码和机械尺寸**：给出可计算 FK/IK 的代码路径、入口函数、输入输出单位、
   关节顺序、URDF 或完整 DH 表，以及基座和工具坐标变换。仅有连杆长度不足以确认零位、方向、
   坐标系和 IK 多解规则。
3. **手机/电脑/WebRTC 工程**：确认实际使用的 Flutter/PC/机械臂推流端工程路径，以及信令、
   STUN/TURN、视频轨道和 `control` DataChannel 的创建位置。当前审计到的候选 Flutter 文件是
   `<WebRTC项目根目录>/webrtc_receiver/lib/player_screen.dart`、同目录下的 `imu_teleop.dart` 和
   `robot_control_channel.dart`；接入前仍需由实验人员确认实际项目根目录和实验版本。
4. **摄像头与时钟资料**：摄像头型号、原生分辨率、目标帧率、编码格式、曝光/码率，以及手机、
   机械臂端和视频帧时钟如何映射到 Unix 时间。必须用实测结果确定最大钟差、视频延迟和掉帧，
   不能用名义帧率推算正式逐帧时间。
5. **实验参数**：三个真实任务及难度、成功放置误差、超时与失败判据、每位参与者每任务的轮次、
   计划参与者数、A/B 平衡顺序和训练参数。参与者数应来自预注册的功效分析。
6. **一对试采集 episode**：先各提供一条 `A_mobile` / `B_pc` 的配对原始数据，用于验证列名、
   时间戳、视频帧和关节顺序。当前尚无这组数据，因此只能完成接口和 dry-run，不能声明实机闭环完成。

夹爪按当前决定只保留接口：后续实验人员确定自动夹爪、二值开合或连续开度，以及单位、范围和
反馈语义后，再实现并接入完整数据链；现阶段不要直接给 6 维位姿动作增加第 7 维。

### 参数写入总表

先复制各 `.example` 模板再填写正式文件；示例值都是占位值，不能用于真机。

| 要填写的内容 | 写入文件与字段 | 说明 |
|---|---|---|
| DH 与机械尺寸 | `dh_params.json`：`convention`、`angle_unit`、`length_unit`、`joint_names`、`joints[].a/alpha/d/theta_offset/direction`，可选 `base`、`tool` 4×4 变换 | 只供离线转换中的 FK 使用；不会自动提供实时 IK |
| 关节顺序与初始位置 | `robot_bridge_config.json`：`joint_names`、`initial_joint_positions_rad` | 必须与驱动、FK/IK 和 CSV 列顺序完全一致 |
| 工作空间、关节和运动限制 | `robot_bridge_config.json`：`safety.workspace_*`、`joint_*`、`max_*step*`、`max_*velocity*`、`max_joint_acceleration_rad_s2`、`max_ik_*error*`、`driver_command_timeout_ms` | 依据手册、标定、工具长度和低速测试填写 |
| 坐标系与控制时效 | `robot_bridge_config.json`：`safety.translation_frame`、`rotation_frame`、`require_strictly_increasing_t`、`max_command_age_ms`、`max_future_skew_ms`、`max_receive_interval_ms` | 钟同步实测完成后才能冻结时效阈值 |
| 采集与时钟门限 | `experiment_config.json`：`capture.target_fps`、`joint_names`、`max_clock_uncertainty_s`、`record_video_latency_ms`、`record_dropped_frames` | `max_clock_uncertainty_s` 必须是实测正数；示例 `null` 会使正式预检失败 |
| 成功、失败和任务定义 | `experiment_config.json`：`success_definition.*`、`tasks[].task_id/difficulty` | 把三个占位任务和两个 `null` 阈值全部替换 |
| 训练与样本计划 | `experiment_config.json`：`training.random_seeds`、`split_seed`、三个 split fraction、`analysis.planned_participants`、`minimum_participants_per_split` | 划分命令的参数和训练脚本中的种子需手工保持一致 |
| 每条 episode 的设计与结果 | `experiment_manifest.csv`：`condition,episode,pair_id,participant_id,task_id,difficulty,trial_index,condition_order,success,failure_type,placement_error_m,completion_time_s,video_latency_ms,dropped_frames` | `pair_id` 必须包含恰好一条 A 和一条 B；`trial_index` 记录实际轮次 |
| 每个训练种子的模型目录 | `checkpoints_a.json`、`checkpoints_b.json`：`{seed: pretrained_model目录}` | 从对应 `.example.json` 复制后填写；正式离线比较会校验种子、目录和 A/B 模型身份，真实映射文件已被 Git 忽略 |
| 特征分析标签 | 从 `labels.example.csv` 复制为 A/B 标签文件 | 精度需按 `task_id` 的专家参考，成功与失败需人工或传感器真值 |
| 每条原始数据 | `raw_ts/cond_a_mobile/episode_NNN`、`raw_ts/cond_b_pc/episode_NNN` | 必需文件和精确 CSV 语义见“第一步：采集” |

`experiment_config.json` 中的 `split_seed`、三个比例和最小参与者数目前不会被
`make_episode_splits.py` 自动读取；运行划分命令时必须把相同值显式传给 `--seed`、
`--train_fraction`、`--validation_fraction`、`--test_fraction` 和
`--min_participants_per_split`。PowerShell 的 `$Seeds` 以及两个 checkpoint 映射也必须与
`training.random_seeds` 保持一致。

### 不能只靠填写 JSON 的对接项

- **真实 FK/IK**：实现 [robot_bridge.py](robot_bridge.py) 中的 `KinematicsProvider`，或用
  `src/lerobot/model/kinematics.py` 的 `RobotKinematics` 加 `LeRobotKinematicsAdapter` 包装现有
  求解器。后者还需要在代码中提供 `urdf_path`、`target_frame_name` 和 `joint_names`，原生角度单位是
  度；URDF 路径、IK 权重、多解选择、奇异和无解策略目前没有 JSON 字段。
- **真实机械臂驱动**：实现 `RobotDriver` 的连接、读关节、发关节目标、停止和独立急停，并在
  控制器侧执行 `JointMotionConstraints`。当前没有真实驱动工厂；仅修改 `mode`、
  `allow_hardware` 或 `driver.type` 会被拒绝。真机程序必须显式构造
  `RobotBridge(driver=..., kinematics=..., safety=..., allow_hardware=True)`，并由独立任务调用
  `check_watchdog()`。
- **WebRTC 与 DataChannel**：在实际接收进程把文本消息交给
  `RobotBridge.handle_control_message()`，再用返回值或 `read_joint_feedback()` 发送反馈；信令地址、
  STUN/TURN、codec、重连和 Flutter/PC 工程路径目前不属于本仓库配置。
- **视频录制**：实现 [acquisition_interfaces.py](acquisition_interfaces.py) 的
  `VideoEpisodeRecorder`，从真正解码的帧原子生成 `video.mp4`、`frame_timestamps.csv` 和
  `video_meta.json`。摄像头分辨率、codec、曝光/码率及延迟/掉帧测量方法仍需在具体录制器中冻结。
- **夹爪**：实验人员确定语义后实现 `GripperAdapter`，再把独立、版本化的 schema 接入桥接录制、
  `time_sync.py` 和转换器。接口已预留，传输和 CSV 落盘尚未接入。

### 实机 DataChannel 契约

一条 WebRTC PeerConnection/服务器转发链路承载一路视频轨道和一个双向 `control`
DataChannel：视频从机械臂端发往 A 手机显示或 B 电脑显示；手机 IMU delta 发往机械臂；机械臂
关节角反馈发回手机。建议通道参数为 `ordered=true, maxRetransmits=0`。

| 方向 | JSON 文本 | 单位 |
|---|---|---|
| 手机 → 机械臂 | `{"t":<Unix毫秒整数>,"d":[dx,dy,dz,dyaw,dpitch,droll]}` | 平移 m，ZYX 旋转 rad |
| 机械臂 → 手机 | `{"j":[j1,...,jN]}` | 关节角 rad |

录制应是事件驱动的：`phone.csv` 保存所有收到的 delta 作为审计流，
`applied_actions.csv` 只保存 `RobotBridge` 确认已执行的 delta 作为训练流，`robot.csv` 保存执行后的
实测关节角。可另存 `command_audit.csv` 和 `events.csv` 记录 arm、stop、watchdog 与安全拒绝事件；
它们是推荐审计文件，不在当前正式必需文件列表中。采集端不硬对齐三路数据，统一交给
`time_sync.py` 离线处理。

## 第零步：冻结实验协议并生成数据划分

正式采集前先复制并填写两个模板：

```powershell
Copy-Item experiments/camera_ablation/experiment_config.example.json `
  experiments/camera_ablation/experiment_config.json
Copy-Item experiments/camera_ablation/experiment_manifest.example.csv `
  experiments/camera_ablation/experiment_manifest.csv
```

- `experiment_config.json`：填写关节顺序、成功放置误差、超时时间、允许的失败类型和
  简单/中等/困难三档真实任务，并依据预注册的样本量/功效分析填写计划参与者数。
- `experiment_manifest.csv`：每条 episode 填 `condition`、`pair_id`、参与者、任务、难度、
  采集顺序和结果。一个 `pair_id` 必须恰好包含一条 `A_mobile` 和一条 `B_pc`。
- 真实配置、参与者清单和原始数据已加入 `.gitignore`，不要提交个人实验记录。

运行预检：

```powershell
uv run --extra training python experiments/camera_ablation/validate_experiment_setup.py `
  --config experiments/camera_ablation/experiment_config.json `
  --manifest experiments/camera_ablation/experiment_manifest.csv
```

采集后增加 `--raw_root raw_ts`，预检会实际读取三路 CSV、解码视频、核对真实逐帧
时间戳、列顺序、有限值、重复时间戳和时间交集，不只是检查文件是否存在。配置中的
`capture.max_clock_uncertainty_s` 必须换成实测的最大允许钟差；超过该值的 episode 会被拒绝。

在训练之前生成配对安全的数据划分：

```powershell
uv run --extra training python experiments/camera_ablation/make_episode_splits.py `
  --manifest experiments/camera_ablation/experiment_manifest.csv `
  --out outputs/experiment_splits.csv --seed 20260728 `
  --train_fraction 0.70 --validation_fraction 0.15 --test_fraction 0.15 `
  --min_participants_per_split 2
```

工具输出每个条件的 train/validation/test episode 列表。同一 `participant_id` 的全部
任务和 A/B 配对永远在同一 split 中；训练只能使用 `train` 列表。参与者不足以填满
所有 split×难度单元时工具会直接报错，不生成“看似可用”的空测试集。

## 完整流程（采集 → 同步 → 转换 → 训练 → 对比）

```
 三路采集(各带时间戳)        ① 时间同步          ② 转换(FK)          ③ 训练              ④ 对比/分析
 robot.csv (关节角)          ┐                                                     ┌ offline_compare.py
 applied_actions.csv (已执行) ├─► time_sync.py ─► convert_raw_to_lerobot.py ─► lerobot-train ─┤
 video.mp4 + 逐帧时间         ┘   SE(3)顺序合成/对齐   LeRobotDataset        ACT       └ feature_extraction.py
                                                                                       操控质量指标
```

## 第一步：采集

每条演示一个文件夹，三路各按自己的到达时刻记时间戳，不要求采集端同频；时间同步在离线阶段完成。

```
raw_ts/cond_a_mobile/episode_000/
    robot.csv            # timestamp,j1..jN；已执行命令后的实测关节角
    phone.csv            # 审计流：所有收到的 delta（包括被拒绝的）
    applied_actions.csv  # 训练流：只有 RobotBridge 实际执行的 delta
    video.mp4        # 手机端显示的同一路机械臂摄像头视频流录制
    video_meta.json  # Unix逐帧时间、display、源钟域→Unix映射及不确定度
    frame_timestamps.csv # 每个实际编码帧的序号、源钟时间、clock_id 和映射后 Unix 时间

raw_ts/cond_b_pc/episode_000/
    robot.csv
    phone.csv
    applied_actions.csv
    video.mp4        # 电脑端显示的同一路机械臂摄像头视频流录制
    video_meta.json  # 必须保存每个实际帧的Unix时间和可审计钟域映射，不能用episode开始时间代替
    frame_timestamps.csv
```

正式 `video_meta.json` 还必须由真正消费并编码解码帧的实现声明
`artifact_type=video_episode_capture`、`writer_capability=decoded_frame_video_recorder` 和
`video_capture_verified_by_this_writer=true`。`acquisition_interfaces.py` 的参考写入函数只会
生成 `timing_metadata_only`，用于接口联调，正式预检会有意拒绝它。预检会把 JSON、逐帧 CSV
与 `video.mp4` 的实际可解码帧数逐一核对。

把 [dh_params.example.json](dh_params.example.json) 复制成 `dh_params.json`，填入真实 DH 参数；
把 [robot_bridge_config.example.json](robot_bridge_config.example.json) 复制成
`robot_bridge_config.json`，由实验人员填入并复核真实坐标系语义和安全限值。同步脚本会读取
同一份坐标系/单步限值，避免训练动作与部署动作语义不一致。

## 第二步：同步 → 转换 → 训练 → 分析

```powershell
$env:Path = "$HOME\.local\bin;$env:Path"; $env:HF_HUB_DISABLE_XET = "1"
$DH = "experiments/camera_ablation/dh_params.json"
$ROBOT = "experiments/camera_ablation/robot_bridge_config.json"

# A：手机端显示
uv run --extra training python experiments/camera_ablation/time_sync.py `
  --in_dir raw_ts/cond_a_mobile --out_dir raw/cond_a_mobile --out_fps 30 `
  --robot_bridge_config $ROBOT

uv run --extra training python experiments/camera_ablation/convert_raw_to_lerobot.py `
  --raw_dir raw/cond_a_mobile --repo_id local/cond_a_mobile --fps 30 `
  --task "抓取放置" --dh_config $DH --resize 480x640

# B：电脑端显示
uv run --extra training python experiments/camera_ablation/time_sync.py `
  --in_dir raw_ts/cond_b_pc --out_dir raw/cond_b_pc --out_fps 30 `
  --robot_bridge_config $ROBOT

uv run --extra training python experiments/camera_ablation/convert_raw_to_lerobot.py `
  --raw_dir raw/cond_b_pc --repo_id local/cond_b_pc --fps 30 `
  --task "抓取放置" --dh_config $DH --resize 480x640

# A/B 分别按协议中的 0、1、2 三个随机种子训练；两组除数据外保持同配置
$Seeds = 0, 1, 2
foreach ($Seed in $Seeds) {
  uv run --extra training lerobot-train --dataset.repo_id=local/cond_a_mobile `
    --dataset.episodes='[把工具输出的A_mobile train列表填在这里]' `
    --dataset.video_backend=pyav --policy.type=act --policy.device=cuda --policy.push_to_hub=false `
    --seed=$Seed --output_dir="outputs/train/cond_a_mobile/seed_$Seed" `
    --job_name="cond_a_mobile_seed_$Seed" --batch_size=8 --steps=5000 `
    --eval_freq=0 --num_workers=0 --wandb.enable=false

  uv run --extra training lerobot-train --dataset.repo_id=local/cond_b_pc `
    --dataset.episodes='[把工具输出的B_pc train列表填在这里]' `
    --dataset.video_backend=pyav --policy.type=act --policy.device=cuda --policy.push_to_hub=false `
    --seed=$Seed --output_dir="outputs/train/cond_b_pc/seed_$Seed" `
    --job_name="cond_b_pc_seed_$Seed" --batch_size=8 --steps=5000 `
    --eval_freq=0 --num_workers=0 --wandb.enable=false
}

# 正式多种子离线比较：只读取预先冻结的 test split
uv run --extra training python experiments/camera_ablation/offline_compare.py `
  --checkpoints_a_json experiments/camera_ablation/checkpoints_a.json `
  --repo_a local/cond_a_mobile `
  --checkpoints_b_json experiments/camera_ablation/checkpoints_b.json `
  --repo_b local/cond_b_pc `
  --protocol_config experiments/camera_ablation/experiment_config.json `
  --split_manifest outputs/experiment_splits.csv --split test `
  --out outputs/offline_action_errors.csv --device cuda --video_backend pyav

# 操控质量特征：速度/平滑度/Jerk/精度/错误率
uv run --extra training python experiments/camera_ablation/feature_extraction.py `
  --repo_id local/cond_a_mobile --condition A_mobile --out outputs/features_A_mobile.csv `
  --ref_repo_id local/expert --labels_csv labels_A_mobile.csv

uv run --extra training python experiments/camera_ablation/feature_extraction.py `
  --repo_id local/cond_b_pc --condition B_pc --out outputs/features_B_pc.csv `
  --ref_repo_id local/expert --labels_csv labels_B_pc.csv

uv run --extra training python experiments/camera_ablation/features_compare.py `
  --features_a outputs/features_A_mobile.csv --features_b outputs/features_B_pc.csv `
  --out outputs/comparison.csv --excel outputs/comparison.xlsx
```

`offline_compare.py` 的正式入口读取 A/B 完整且相同的
`training.random_seeds → checkpoint` 映射，并严格使用 `--split_manifest` 中预先冻结的 split。
运行前必须把两个 `.example.json` 复制为不带 `.example` 的本地映射并填写真实
`pretrained_model` 目录；脚本会核对 checkpoint 的训练种子、路径/模型身份和 A/B 是否误复用，
然后在同一组测试 episode 上逐种子推理，输出物理单位下的逐维 MAE、MSE、RMSE 及复现元数据。
不要把示例映射或训练 split 用于论文结果。

`time_sync.py` 把每个 30 Hz 区间内“已执行”的离散 delta 按时间顺序做与
`RobotBridge` 一致的 SE(3) 合成，而不是逐列相加或零阶保持。这样不会错误地把旋转当成
可交换量，也能正确处理 tool-frame 平移；若合成后的单步超过实机限值，脚本会中止并要求
提高 `--out_fps`。`translation_frame=tool + rotation_frame=base` 中依赖初始 FK 位姿、
无法用单个 6 维动作精确表达的区间也会中止。视频优先使用 `frame_timestamps_s`，旧
`start_ts + n/fps` 只保留作历史 CFR 数据回退。同步输出和 LeRobot 转换均先写同父目录
staging，全部成功后才原子发布，失败不会留下看似可用的正式半成品。

`convert_raw_to_lerobot.py` 会先完整预检全部 episode，再创建数据集，并流式解码视频；
默认严格要求 state、action、video 完全等长。只有确认是可接受的
极小尾部差异时，才能显式设置 `--max_length_mismatch N`；该选项会裁到最短流，并打印记录。

## 学习质量指标

建议把“哪组数据学习质量更高”拆成三层：

| 层级 | 指标 | 说明 |
|------|------|------|
| 离线模仿学习 | 物理单位下逐维与平移/旋转 MAE、MSE、RMSE；训练 loss | 使用 checkpoint 后处理恢复 m/rad；训练清单无法核实时也默认终止 |
| 统一测试集 | 两个模型在同一专家测试集或同一任务集上的动作误差 | 比各自 held-out 更公平，可减少 A/B 数据本身难度差异带来的偏差 |
| 真机验证 | 成功率、完成时间、失败类型 | 最能说明训练出的策略是否真的更好，但需要真机 rollout |

本实验已确定为参与者内配对设计。划分和正式推断的独立单位都是参与者，不能把同一人
多个任务 episode 当成多个独立样本；训练随机种子也必须保留。参与者数量不足时不要报告
总体 p 值，应先增加参与者或采用预先声明的留一参与者/混合效应分析。

## 操控质量特征

| 类别 | 输出指标 | 说明 |
|------|---------|------|
| 速度 | raw_duration_s（仅 QC）、completion_time_s、mean/peak_speed、mean_angular_speed | 报告的完成时间优先取标签；只对成功任务进入性能比较 |
| 平滑度 | mean_acc, acc_sd, mean_jerk, jerk_sd, ang_speed_sd, log_dimensionless_jerk | 二/三阶导 + 无量纲 jerk |
| 精度 | mean/max/rmse_position_error、orientation_error | 专家标签必须按 `task_id` 建参考，禁止跨任务平均轨迹 |
| 错误率 | SR/FR + 失败分类 | 需 `--labels_csv`，列为 `episode,success[,failure_type]` |

## 文件说明

| 文件 | 作用 |
|------|------|
| `time_sync.py` | 已执行增量按区间做 SE(3) 顺序合成并复核单步限值、关节插值、视频按真实逐帧时间对齐、原子发布 |
| `convert_raw_to_lerobot.py` | 全量预检、流式视频、严格列契约、SI 单位和 FK 增广、数据集原子发布 |
| `forward_kinematics.py` / `dh_params.example.json` | 正运动学 + DH 参数模板 |
| `robot_bridge.py` / `robot_bridge_config.example.json` | 硬件无关的 FK/IK/驱动接口、IK→FK 残差门、控制器运动约束、独立 watchdog 和 dry-run |
| `acquisition_interfaces.py` | 版本化夹爪扩展接口、视频帧录制接口与跨时钟映射/元数据契约（不含厂商或 WebRTC 实现） |
| `experiment_config.example.json` | 固定 A/B 条件、成功标准、任务与训练参数模板 |
| `checkpoints_a.example.json` / `checkpoints_b.example.json` | A/B 预注册训练随机种子到 checkpoint 的映射模板；复制为不带 `.example` 的本地文件后供正式比较读取 |
| `experiment_manifest.example.csv` | A/B 配对 episode、参与者、任务和结果字段模板 |
| `validate_experiment_setup.py` | 正式实验前检查协议、清单和可选原始数据目录 |
| `make_episode_splits.py` | 按参与者隔离、难度档案分层的 train/validation/test 划分 |
| `offline_compare.py` | 按预注册种子映射和冻结 split 运行 A/B 多种子离线比较，校验模型身份并输出物理单位误差与复现元数据 |
| `feature_extraction.py` | 每条 episode 提取速度、平滑度、精度、错误率等特征 |
| `features_compare.py` | 参与者级配对推断/Cohen's dz/FDR、成功比例 exact sign test、条件×难度与顺序模型；独立数据可用 Welch/Fisher |
| `prepare_condition_a.py` / `run_pipeline.ps1` | 旧版“删视频列”占位烟雾测试，仅用于验证训练链路，不代表当前实验设计 |

## 已验证 / 尚待实机

- 自动测试已覆盖：米/毫米和度/弧度 FK、关节方向、刚体变换、非交换 SE(3) 增量合成和
  聚合限值、VFR 逐帧时间、流式视频、原子发布、严格列/流长度、参与者隔离划分、实验
  预检、物理单位离线指标、坏 episode 保留、严格配对、IK 残差和可抢占 watchdog。
- `robot_bridge.py` 默认只能 dry-run。真实硬件必须由实验人员实现 `KinematicsProvider`
  和 `RobotDriver`，填写真实尺寸/限制，并显式启用硬件；驱动必须在控制器侧执行
  `JointMotionConstraints` 的速度、加速度和 deadline，并提供可并发取消卡死发送的独立
  急停通道。实机还必须在独立控制任务中周期调用 `check_watchdog()`；代码不会自动连接机械臂。
- 当前真实实验必须 A/B 都保存 `video.mp4`，转换时不要加 `--no_camera`。
- 尚未完成：Flutter UI 集成、接收端 WebRTC 视频编码实现、机械臂推流端 DataChannel 适配、真实
  FK/IK 与电机驱动、试采集、真实 A/B 训练和真机 rollout。
- 抓取任务的夹爪动作语义仍未冻结（自动夹爪、开合二值或连续开度）。
  `acquisition_interfaces.py` 已提供独立、版本化的 `GripperAdapter` 和 action/state schema
  接口；在实验人员确定前，6 维位姿协议不会擅自增加第 7 维。确定后还需把该 schema
  接进桥接录制、同步和转换。
- 精度/错误率仍需专家参考轨迹和人工/传感器标签；在线成功率必须来自真机 rollout。
- 正式结果至少使用三个训练随机种子，并平衡 A/B 采集顺序。
