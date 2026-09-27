# Wrist-view Presentation 实验实现

本目录已按《实验方案》重构为两任务、两条件、SmolVLA 主线实验。研究协议见 [PROTOCOL.md](PROTOCOL.md)，逐项实施状态见 [MODIFICATION_PROGRESS.md](MODIFICATION_PROGRESS.md)。

## 当前状态

软件侧已经具备：

- 20 名参与者、G1–G4、200 次正式尝试的确定性清单生成与校验；
- 堆叠和颜色分类的逐 episode 语言指令；
- 等量平衡训练样本筛选和完整审计；
- SmolVLA/可选 ACT 的多种子训练入口；
- 240 次严格配对真实机器人 rollout 清单；
- 条件 × 任务的人类示范分析和 A−B rollout 配对汇总。

当前不能直接开展正式实验。示例配置故意将夹爪 schema、任务超时、真实布局/复位目录、时钟阈值、最低样本量、SmolVLA 资源和部署安全参数保留为 `null`，预检会阻止继续执行。

## 1. 创建本地正式配置

从仓库根目录运行：

```powershell
Copy-Item experiments/wrist_view_presentation/experiment_config.example.json `
  experiments/wrist_view_presentation/experiment_config.json
Copy-Item experiments/wrist_view_presentation/dh_params.example.json `
  experiments/wrist_view_presentation/dh_params.json
Copy-Item experiments/wrist_view_presentation/robot_bridge_config.example.json `
  experiments/wrist_view_presentation/robot_bridge_config.json
```

在正式采集前补齐并冻结：

- `capture.max_clock_uncertainty_s` 和 `capture.gripper`；
- 两项任务的 `max_observation_time_s`；
- `layout_catalog_path` 与 `evaluation.reset_catalog_path`（格式见 [CATALOG_FORMAT.md](CATALOG_FORMAT.md)）；
- `training.minimum_eligible_per_condition_task`；
- SmolVLA 的 `batch_size`、`steps`、`chunk_size` 与 `n_action_steps`；
- 机器人 DH、关节顺序、工作空间，以及 `evaluation.deployment` 中的控制频率、watchdog 和命令过期阈值；
- 独立急停、碰撞边界和真实硬件适配器。

本地正式配置、参与者数据、checkpoint 与 `outputs/` 不应提交到 Git。

## 2. 生成并冻结 200 行正式清单

```powershell
uv run --extra training python experiments/wrist_view_presentation/generate_experiment_manifest.py `
  --config experiments/wrist_view_presentation/experiment_config.json `
  --out experiments/wrist_view_presentation/experiment_manifest.csv
```

生成结果满足每个“条件 × 任务”50 次、每组 5 人和每位参与者 10 次正式尝试。清单一旦冻结，只填写结果字段，不重新随机化。

## 3. 协议和数据预检

仅检查配置与清单：

```powershell
uv run --extra training python experiments/wrist_view_presentation/validate_experiment_setup.py `
  --config experiments/wrist_view_presentation/experiment_config.json `
  --manifest experiments/wrist_view_presentation/experiment_manifest.csv
```

采集后连同原始数据检查：

```powershell
uv run --extra training python experiments/wrist_view_presentation/validate_experiment_setup.py `
  --config experiments/wrist_view_presentation/experiment_config.json `
  --manifest experiments/wrist_view_presentation/experiment_manifest.csv `
  --raw_root raw_ts
```

每个 episode 必须保存机器人状态、手机原始命令、机器人实际执行动作、夹爪动作/状态/schema、真实视频和逐帧时间戳。通用同步、预检和 LeRobot 转换链路已经支持夹爪；仍需用真实硬件适配器生成这些输入，并冻结量程与单位。

## 4. 等量筛选训练 episode

结果填写完成且通过预检后运行：

```powershell
uv run --extra training python experiments/wrist_view_presentation/select_training_episodes.py `
  --manifest experiments/wrist_view_presentation/experiment_manifest.csv `
  --selection_seed 20260926 `
  --minimum_per_cell <冻结的最小合格样本数> `
  --out outputs/wrist_view_presentation/training_selection.csv `
  --summary outputs/wrist_view_presentation/training_selection_summary.json
```

脚本先筛选完整成功且所有质量门槛合格的 episode，再把 A/B × 两任务四个单元下采样到共同最小值。不要手工删除失败记录或为某一条件单独挑选“更好”的轨迹。

## 5. 原始数据转换

转换器按清单与配置自动写入每个 episode 的英文任务 prompt，并生成源 episode 到 LeRobot episode 的映射：

```powershell
uv run --extra training python experiments/wrist_view_presentation/convert_raw_to_lerobot.py `
  --raw_dir raw_ts/A_mobile_colocated `
  --output_dir datasets/A_mobile_colocated `
  --repo_id local/wrist_view_A `
  --fps 30 `
  --manifest experiments/wrist_view_presentation/experiment_manifest.csv `
  --protocol_config experiments/wrist_view_presentation/experiment_config.json `
  --condition A_mobile_colocated `
  --source_map_out outputs/wrist_view_presentation/source_map_A_mobile_colocated.json
```

B 条件同理。转换器会强制核对夹爪 schema、量程、列名和逐 episode 长度；硬件适配器尚未提供真实数据时，预检会阻止正式转换。

## 6. 完整流水线

硬件采集和 rollout 适配器完成后：

```powershell
.\experiments\wrist_view_presentation\run_pipeline.ps1 `
  -CollectionScript "<硬件采集适配器.ps1>" `
  -RolloutScript "<短时域真实机器人评测适配器.ps1>"
```

流水线顺序为：预检 → 采集 → 同步/转换 → 冻结训练选择 → 多种子训练 → 人类示范分析 → 生成 240 次评测清单 → 真实机器人 rollout → 配对汇总。

rollout 适配器必须在 `outputs/wrist_view_presentation/rollout/rollout_results.csv` 写回完整冻结表；缺少任何正式结果时分析器会拒绝运行。

## 7. 单独生成和分析 rollout 清单

```powershell
uv run --extra training python experiments/wrist_view_presentation/make_rollout_manifest.py `
  --config experiments/wrist_view_presentation/experiment_config.json `
  --out outputs/wrist_view_presentation/rollout_manifest_smolvla.csv

uv run --extra training python experiments/wrist_view_presentation/analyze_rollouts.py `
  --results outputs/wrist_view_presentation/rollout/rollout_results.csv `
  --out_dir outputs/wrist_view_presentation/rollout/analysis `
  --expected_seeds 0,1,2
```

分析输出包括条件分层均值、120 个 A/B 严格配对行、A−B 差值汇总和描述性 95% t 置信区间。确认性模型与多重比较方案仍须在正式采集前预注册。

## 8. 主要文件

| 文件 | 作用 |
|---|---|
| `experiment_config.example.json` | v2 协议、任务、训练与评测配置模板 |
| `CATALOG_FORMAT.md` | 采集布局与 rollout 复位目录的版本化格式 |
| `generate_experiment_manifest.py` | 生成/验证 200 次正式尝试 |
| `validate_experiment_setup.py` | 配置、清单和原始数据停止门 |
| `select_training_episodes.py` | 四单元等量训练选择与审计 |
| `convert_raw_to_lerobot.py` | 逐 episode prompt 和源 ID 映射转换 |
| `run_pipeline.ps1` | SmolVLA 主线端到端编排 |
| `feature_extraction.py` / `features_compare.py` | 人类示范质量与条件 × 任务分析 |
| `make_rollout_manifest.py` | 冻结 240 次真实机器人评测 |
| `analyze_rollouts.py` | 验证完整结果并输出严格配对汇总 |

## 9. 测试

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests/experiments/wrist_view_presentation -q
```

测试通过只说明软件契约和统计数据流一致；不代表夹爪、碰撞保护、急停或真实机器人闭环已经验证。
