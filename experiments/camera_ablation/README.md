# 摄像头有无对照实验（Camera Ablation）

量化"操控时有没有摄像头辅助"对机械臂操控/策略性能的提升。两套**真实采集**的数据集对比。

## 实验设计

| 条件 | 采集方式 | 数据集内容 | 训练的 ACT 输入 |
|------|----------|-----------|----------------|
| **A** | 手机 APP 操控，无摄像头辅助 | state + action（无视频） | `observation.state` |
| **B** | 电脑屏幕看摄像头画面操控 | state + action + 摄像头视频 | `observation.state` + `observation.images` |

> ⚠️ A=手机、B=电脑+摄像头，同时变了"操控设备"和"摄像头"两个因素，论文表述为"两种操控模式"对比。

## 数据语义

```
action = [Δx,Δy,Δz,ΔYaw,ΔPitch,ΔRoll]              手机发的 6 维末端 delta
state  = [关节角 j1..jN] ⊕ [x,y,z,Yaw,Pitch,Roll]   关节角 + FK 算出的末端位姿
```
单位：位置米、角度弧度(ZYX)。机械臂只输出关节角，末端位姿由转换器用 DH 表做 FK。

## 完整流程（含时间同步 + 特征提取）

```
 三路采集(各带时间戳)        ① 时间同步         ② 转换(FK)        ③ 训练           ④ 对比/分析
 robot.csv (关节角)   ┐                                                       ┌ offline_compare.py
 phone.csv (delta)    ├─► time_sync.py ─► convert_raw_to ─► lerobot-train ─►─┤  (动作L1 + t检验)
 video.mp4 (条件B)    ┘   对齐到30Hz网格    _lerobot.py      (ACT)            └ feature_extraction.py
                                                                                (速度/平滑度/Jerk + t检验)
```

## 第一步：采集（每条演示一个文件夹，三路各带时间戳）

```
raw_ts/cond_b/episode_000/
    robot.csv        # timestamp, j1..jN          机械臂关节角（手机端 robot_control_channel 落盘）
    phone.csv        # timestamp, dx..droll        手机 6 维 delta
    video.mp4        # 摄像头录像（条件B；另接录制）
    video_meta.json  # {"start_ts": <秒>}          供视频对齐
```
- 三路各按自己的到达时刻记时间戳（不要求同频）；时间同步在第②步离线做。
- 把 [dh_params.example.json](dh_params.example.json) 复制成 `dh_params.json` 填真实 DH 参数。

## 第二步：同步 → 转换 → 训练 → 分析

```powershell
$env:Path = "C:\Users\19034\.local\bin;$env:Path"; $env:HF_HUB_DISABLE_XET = "1"
$DH = "experiments/camera_ablation/dh_params.json"

# ① 时间同步：三路对齐到 30Hz，输出转换器能吃的 states.csv/actions.csv[/video.mp4]
uv run --extra training python experiments/camera_ablation/time_sync.py `
  --in_dir raw_ts/cond_b --out_dir raw/cond_b --out_fps 30

# ② 转换（带 FK 末端位姿）
uv run --extra training python experiments/camera_ablation/convert_raw_to_lerobot.py `
  --raw_dir raw/cond_b --repo_id local/cond_b_camera --fps 30 --task "抓取放置" --dh_config $DH --resize 480x640
#  条件A同理：time_sync raw_ts/cond_a → raw/cond_a，convert 加 --no_camera

# ③ 各训一个 ACT（条件A纯state已验证可训）
uv run --extra training lerobot-train --dataset.repo_id=local/cond_b_camera `
  --dataset.video_backend=pyav --policy.type=act --policy.device=cuda --policy.push_to_hub=false `
  --output_dir=outputs/train/cond_b --job_name=cond_b --batch_size=8 --steps=5000 --eval_freq=0 --num_workers=0 --wandb.enable=false

# ④a 模型层面对比：动作 L1 + 独立样本 t 检验
uv run --extra training python experiments/camera_ablation/offline_compare.py `
  --ckpt_a outputs/train/cond_a/checkpoints/last/pretrained_model --repo_a local/cond_a_phone `
  --ckpt_b outputs/train/cond_b/checkpoints/last/pretrained_model --repo_b local/cond_b_camera --test_frac 0.2

# ④b 操控质量对比：四类特征(速度/平滑度Jerk/加速度 + 精度 + 错误率)，每条演示一行 → 再做 t 检验/ANOVA
#    --ref_repo_id 指向熟练操作者录的专家数据集(精度参考)；--labels_csv 给成功/失败标注(错误率)
uv run --extra training python experiments/camera_ablation/feature_extraction.py `
  --repo_id local/cond_a_phone --condition A --out outputs/features_A.csv `
  --ref_repo_id local/expert --labels_csv labels_A.csv
uv run --extra training python experiments/camera_ablation/feature_extraction.py `
  --repo_id local/cond_b_camera --condition B --out outputs/features_B.csv `
  --ref_repo_id local/expert --labels_csv labels_B.csv

# ④c 特征层 A vs B 统计对比：Welch t检验 + Cohen's d + FDR；SR用Fisher；多难度ANOVA
#    --excel 额外导出多sheet的xlsx(显著行+大效应高亮)
uv run --extra training python experiments/camera_ablation/features_compare.py `
  --features_a outputs/features_A.csv --features_b outputs/features_B.csv `
  --out outputs/comparison.csv --excel outputs/comparison.xlsx
```

## 四类特征（实验方案 §4，feature_extraction.py 实现）

| 类别 | 输出指标 | 说明 |
|------|---------|------|
| 速度 | completion_time, mean/peak_speed, mean_angular_speed | 末端轨迹一阶导 |
| **平滑度** | mean_acc, acc_sd, **mean_jerk, jerk_sd**, ang_speed_sd, log_dimensionless_jerk | 二/三阶导 + 无量纲 jerk |
| 精度 | mean/max/rmse_position_error、orientation_error(四元数测地距离) | ✅ 需 `--ref_repo_id` 专家数据集(熟练操作者录的参考轨迹，时间归一化对齐) |
| 错误率 | SR/FR + 失败分类(碰撞/抓取/超时/中止) | ✅ 需 `--labels_csv`(列:episode,success[,failure_type]，见 labels.example.csv) |

## 文件说明

| 文件 | 作用 |
|------|------|
| `time_sync.py` | 三路(机械臂/手机/视频)按时间戳对齐到公共网格 |
| `convert_raw_to_lerobot.py` | 对齐后数据 → LeRobot 数据集，state 做 FK 增广 |
| `forward_kinematics.py` / `dh_params.example.json` | 正运动学 + DH 参数模板（复制成 dh_params.json 填真值） |
| `feature_extraction.py` | 四类特征（速度/平滑度/精度/错误率），每条演示一行 |
| `features_compare.py` | 特征层 A vs B：Welch t检验 + Cohen's d + FDR；SR用Fisher；多难度ANOVA |
| `offline_compare.py` | A/B 模型动作 L1 + t 检验 |
| `prepare_condition_a.py` / `run_pipeline.ps1` | 占位/烟雾测试（删列），无真机时验证流水线 |

## 已验证 / 注意

- ✅ 端到端跑通：合成多频率三路 → 时间同步(50/20Hz→30Hz) → 转换(FK) → ACT训练 → 特征提取。
- 为支持纯 state 条件A，改了 LeRobot 两处：`act/configuration_act.py`(validate_features) 与 `act/modeling_act.py:404`(forward batch_size)。
- 精度/错误率需额外输入（参考轨迹/标注）；在线评估（成功率等）需真机。
- 严格对照建议多随机种子各训几次再做检验。8GB 显存 `--batch_size=8`，OOM 降到 4。
