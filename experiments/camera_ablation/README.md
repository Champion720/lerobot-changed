# 视频显示端对照实验（Mobile vs PC Display）

量化“机械臂摄像头视频流显示在手机端还是电脑端”对遥操作数据质量和模仿学习质量的影响。两组数据都来自真实采集，且都包含机械臂摄像头视频；差异不再是“有无摄像头”，而是操作者观看视频反馈的终端不同。

> 目录名仍叫 `camera_ablation` 是历史遗留。当前研究问题应表述为：**同一机械臂摄像头视频流，在手机端显示与电脑端显示两种条件下，哪一组遥操作数据训练出的策略学习质量更高。**

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

## 完整流程（采集 → 同步 → 转换 → 训练 → 对比）

```
 三路采集(各带时间戳)        ① 时间同步          ② 转换(FK)          ③ 训练              ④ 对比/分析
 robot.csv (关节角)   ┐                                                           ┌ offline_compare.py
 phone.csv (delta)    ├─► time_sync.py ─► convert_raw_to_lerobot.py ─► lerobot-train ─┤  学习质量指标
 video.mp4 (A/B都有)  ┘   对齐到30Hz网格       LeRobotDataset        ACT            └ feature_extraction.py
                                                                                       操控质量指标
```

## 第一步：采集

每条演示一个文件夹，三路各按自己的到达时刻记时间戳，不要求采集端同频；时间同步在离线阶段完成。

```
raw_ts/cond_a_mobile/episode_000/
    robot.csv        # timestamp, j1..jN                 机械臂关节角
    phone.csv        # timestamp, dx,dy,dz,dyaw,dpitch,droll
    video.mp4        # 手机端显示的同一路机械臂摄像头视频流录制
    video_meta.json  # {"start_ts": <秒>, "fps": <视频fps>, "display": "mobile"}

raw_ts/cond_b_pc/episode_000/
    robot.csv
    phone.csv
    video.mp4        # 电脑端显示的同一路机械臂摄像头视频流录制
    video_meta.json  # {"start_ts": <秒>, "fps": <视频fps>, "display": "pc"}
```

把 [dh_params.example.json](dh_params.example.json) 复制成 `dh_params.json`，填入真实 DH 参数。

## 第二步：同步 → 转换 → 训练 → 分析

```powershell
$env:Path = "C:\Users\19034\.local\bin;$env:Path"; $env:HF_HUB_DISABLE_XET = "1"
$DH = "experiments/camera_ablation/dh_params.json"

# A：手机端显示
uv run --extra training python experiments/camera_ablation/time_sync.py `
  --in_dir raw_ts/cond_a_mobile --out_dir raw/cond_a_mobile --out_fps 30

uv run --extra training python experiments/camera_ablation/convert_raw_to_lerobot.py `
  --raw_dir raw/cond_a_mobile --repo_id local/cond_a_mobile --fps 30 `
  --task "抓取放置" --dh_config $DH --resize 480x640

# B：电脑端显示
uv run --extra training python experiments/camera_ablation/time_sync.py `
  --in_dir raw_ts/cond_b_pc --out_dir raw/cond_b_pc --out_fps 30

uv run --extra training python experiments/camera_ablation/convert_raw_to_lerobot.py `
  --raw_dir raw/cond_b_pc --repo_id local/cond_b_pc --fps 30 `
  --task "抓取放置" --dh_config $DH --resize 480x640

# 分别训练同配置 ACT
uv run --extra training lerobot-train --dataset.repo_id=local/cond_a_mobile `
  --dataset.video_backend=pyav --policy.type=act --policy.device=cuda --policy.push_to_hub=false `
  --output_dir=outputs/train/cond_a_mobile --job_name=cond_a_mobile `
  --batch_size=8 --steps=5000 --eval_freq=0 --num_workers=0 --wandb.enable=false

uv run --extra training lerobot-train --dataset.repo_id=local/cond_b_pc `
  --dataset.video_backend=pyav --policy.type=act --policy.device=cuda --policy.push_to_hub=false `
  --output_dir=outputs/train/cond_b_pc --job_name=cond_b_pc `
  --batch_size=8 --steps=5000 --eval_freq=0 --num_workers=0 --wandb.enable=false

# 模型层面对比：held-out 动作误差 + 独立样本 t 检验
uv run --extra training python experiments/camera_ablation/offline_compare.py `
  --ckpt_a outputs/train/cond_a_mobile/checkpoints/last/pretrained_model --repo_a local/cond_a_mobile `
  --ckpt_b outputs/train/cond_b_pc/checkpoints/last/pretrained_model --repo_b local/cond_b_pc `
  --test_frac 0.2 --device cuda

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

## 学习质量指标

建议把“哪组数据学习质量更高”拆成三层：

| 层级 | 指标 | 说明 |
|------|------|------|
| 离线模仿学习 | held-out action L1 / MSE、训练 loss 曲线、收敛速度 | `offline_compare.py` 当前计算动作 L1，越低表示越接近示教动作 |
| 统一测试集 | 两个模型在同一专家测试集或同一任务集上的动作误差 | 比各自 held-out 更公平，可减少 A/B 数据本身难度差异带来的偏差 |
| 真机验证 | 成功率、完成时间、失败类型 | 最能说明训练出的策略是否真的更好，但需要真机 rollout |

如果 A/B 是独立采集，默认使用独立样本 t 检验；只有两组 episode 一一对应时才使用 `--paired`。

## 操控质量特征

| 类别 | 输出指标 | 说明 |
|------|---------|------|
| 速度 | completion_time, mean/peak_speed, mean_angular_speed | 末端轨迹一阶导 |
| 平滑度 | mean_acc, acc_sd, mean_jerk, jerk_sd, ang_speed_sd, log_dimensionless_jerk | 二/三阶导 + 无量纲 jerk |
| 精度 | mean/max/rmse_position_error、orientation_error | 需 `--ref_repo_id` 指向专家参考轨迹 |
| 错误率 | SR/FR + 失败分类 | 需 `--labels_csv`，列为 `episode,success[,failure_type]` |

## 文件说明

| 文件 | 作用 |
|------|------|
| `time_sync.py` | 三路数据按时间戳对齐到公共 30Hz 网格 |
| `convert_raw_to_lerobot.py` | 对齐后数据转 LeRobotDataset，state 可做 FK 增广 |
| `forward_kinematics.py` / `dh_params.example.json` | 正运动学 + DH 参数模板 |
| `offline_compare.py` | A/B 模型动作误差对比 + t 检验 |
| `feature_extraction.py` | 每条 episode 提取速度、平滑度、精度、错误率等特征 |
| `features_compare.py` | 特征层 A vs B：Welch t 检验 + Cohen's d + FDR；成功率用 Fisher |
| `prepare_condition_a.py` / `run_pipeline.ps1` | 旧版“删视频列”占位烟雾测试，仅用于验证训练链路，不代表当前实验设计 |

## 已验证 / 注意

- 已验证：合成多频率三路 → 时间同步 → 转换(FK) → ACT 训练 → 特征提取。
- 当前真实实验必须 A/B 都保存 `video.mp4`，转换时不要加 `--no_camera`。
- 精度/错误率需额外输入参考轨迹和人工标注；在线成功率需真机 rollout。
- 严格对照建议多随机种子训练，且平衡采集顺序，避免学习效应或疲劳效应混入显示端差异。
- 8GB 显存训练 ACT 建议 `--batch_size=8`，OOM 时降到 4。
