# 腕部视觉反馈呈现实验修改与进度基线

版本：2026-09-24  
状态：第一阶段软件重构已完成；夹爪、真实硬件与冻结参数仍阻塞正式采集  
适用目录：`experiments/wrist_view_presentation/`

## 1 文档用途

本文用于对照《实验方案.docx》修改当前实验框架，并持续记录实施进度、验收结果、阻断项和研究设计决策。

本文不是实验结果，也不预设 A 条件或 B 条件更优。任何涉及参与者、任务配额、成功定义、训练样本选择和实机评测的实质修改，都应在正式采集前冻结并记录版本。

状态标记：

- `DONE`：已实现并通过对应检查。
- `IN PROGRESS`：正在修改，尚未满足验收条件。
- `TODO`：尚未开始。
- `BLOCKED`：缺少硬件参数、研究者决定或外部实现，不能安全继续。
- `OPTIONAL`：不属于主实验最低交付范围。

## 2 复核结论

### 2.1 保留的实验核心

- 正式自变量仍是腕部视觉反馈与手机运动控制的空间关系。
- `A_mobile_colocated`：腕部视频显示在控制手机上。
- `B_desktop_separated`：腕部视频显示在固定电脑屏幕上。
- 两个条件使用同一个腕部摄像头、相同视角、相同手机 IMU 控制、相同机器人和相同任务物体。
- 20 名参与者均完成 A、B 两个条件，每人每条件 5 次，共 200 次正式人工尝试。
- 正式任务收敛为 `stacking` 和 `color_sorting` 两项；《实验方案》前部的三任务内容视为早期讨论，不再进入正式配置。
- SmolVLA 是主模型；ACT 只作为资源允许时的附加模型族。
- 真实机器人 rollout 成功率是模型比较的主要证据，训练损失和离线动作误差只能作为辅助证据。

### 2.2 相比上一版方向的改进

| 编号 | 调整 | 最终建议 | 原因 |
|---|---|---|---|
| R01 | 主训练数据划分 | 正式主模型使用全部经过平衡和质量筛选的成功轨迹；参与者隔离划分只用于独立诊断 | 20 人数据量有限，若采用 60/20/20，训练池在成功筛选前只有每任务每条件 30 次，不利于 SmolVLA；主评测本身已有独立实机 rollout |
| R02 | 夹爪动作 | 夹爪动作和夹爪状态必须在采集、同步、转换、训练和 rollout 全链路实现后才能训练 | 当前正式动作只有 6 维位姿增量，模型无法学习抓取和释放 |
| R03 | 任务语言 | 每条 episode 从 manifest 获取固定英文策略指令；中文参与者提示另存字段 | SmolVLA 需要逐 episode 的任务条件，不能用一个全局 `-Task` 覆盖两项任务 |
| R04 | 成功轨迹平衡 | 四个池 `A×堆叠`、`A×分类`、`B×堆叠`、`B×分类` 取共同最小样本量 | 同时控制条件间数据量和多任务采样权重 |
| R05 | 配对敏感性分析 | 主训练按文档使用等量成功池；资源允许时增加“双条件均成功的完整 pair”敏感性训练 | 主方案保留更多数据，敏感性方案检查参与者组成差异是否影响结论 |
| R06 | 条件内试次顺序 | 除 A/B 顺序外，冻结 5 次任务顺序模板和布局顺序 | 减少学习、疲劳和任务连续重复造成的混杂 |
| R07 | 实机执行 | 冻结 240 行 rollout manifest，并限制 action chunk 的实际执行步数 | 默认一次执行较长动作块对真实机械臂风险过高，也会削弱视觉闭环 |
| R08 | 文献结果 | 文献阶段分数与本研究完整成功率分表展示 | 两者定义不同，不能直接做提升幅度比较 |

## 3 当前框架的关键缺口

### 3.1 夹爪闭环尚未接入 `BLOCKED`

当前 `convert_raw_to_lerobot.py` 将动作固定为：

```text
dx, dy, dz, dyaw, dpitch, droll
```

`acquisition_interfaces.py` 虽然声明了独立的 `GripperAdapter`，但明确没有把夹爪值加入正式动作；`time_sync.py` 和转换器也只接受固定 6 维动作。堆叠和分类都必须执行闭合、保持和释放，因此这是正式训练前的硬阻断项。

推荐实现：

- 原始采集继续保留 6 维笛卡尔动作为独立、版本化 schema。
- 新增 `gripper_actions.csv`、`gripper_states.csv` 和 `gripper_schema.json`。
- 夹爪命令使用“持续保持的目标位置或目标开合度”，不要只记录瞬时 toggle 事件，避免重采样后丢失开合动作。
- 同步阶段以零阶保持方式对齐夹爪目标，以传感器实测值对齐夹爪状态。
- LeRobot 训练动作建议为 `[6D Cartesian delta, gripper_target]`，状态追加实际夹爪位置或宽度。
- schema 的确切名称、单位、范围和开闭方向必须由最终夹爪硬件确定，不能使用占位定义进入正式采集。

验收条件：模型数据能逐帧还原“移动、闭合、保持、移动、释放”的完整序列；回放时夹爪方向、数值范围和时间对齐正确。

### 3.2 转换器已支持逐 episode 任务文本 `DONE`

流水线已移除全局 `-Task`，正式数据按 manifest 和协议配置逐 episode 写入任务文本：

- `manifest.task_id` 决定每条 episode 的策略任务文本。
- `stacking` 和 `color_sorting` 使用固定、版本化的 `policy_task_en`。
- A/B 使用完全相同的策略文本，文本中不得出现条件名称或显示设备。
- 另设 `operator_instruction_zh` 保存给参与者阅读的中文说明，两者不能混用。
- 转换输出 `source_episode_id -> lerobot_episode_index` 映射，禁止依赖目录排序隐式推断。

### 3.3 SmolVLA 主训练入口已重构，资源参数待冻结 `IN PROGRESS`

当前流水线已把 SmolVLA 主训练改为预训练模型入口：

```text
--policy.path=lerobot/smolvla_base
```

并同时启用 `training` 和 `smolvla` 依赖。A/B 必须共享：

- 同一个预训练模型版本或本地快照哈希；
- 相同训练步数、batch、优化器、调度器和模型覆盖参数；
- 相同随机种子集合；
- 相同的任务采样权重；
- 相同的数据质量与成功筛选规则。

本机为 8GB 显存，正式 batch、训练步数、精度模式和 `n_action_steps` 必须先用非正式 pilot 数据完成显存与时延测试，再在查看 A/B 正式结果前冻结。

### 3.4 rollout 清单与结果分析已实现，硬件执行仍阻塞 `IN PROGRESS`

当前代码已强制检查 240 次配额、训练布局与新位置比例、共享 reset 清单和 trial-level 结果字段：

- `make_rollout_manifest.py`：生成冻结的 240 行 SmolVLA 主评测清单。
- `make_rollout_manifest.py` 内置验证器：检查条件、种子、任务和布局配额。
- `analyze_rollouts.py`：输出逐任务、逐布局、逐种子结果和 A/B 差异。
- rollout 硬件适配器必须回填 manifest 中已有的 `rollout_id`，不能自行生成不可追溯编号。

### 3.5 默认动作块执行长度存在安全风险 `BLOCKED`

SmolVLA 可以预测动作块，但真实机械臂不应未经视觉反馈直接执行过长的 6D 增量序列。正式部署前必须测量：

- 单次推理时延；
- 相机到策略的端到端延迟；
- 每次实际执行的 `n_action_steps`；
- 看门狗周期和命令过期阈值；
- 夹爪命令与机械臂动作的同步方式；
- action chunk 被安全限制截断时的记录方式。

推荐采用短执行窗口或 RTC/receding-horizon 方式，每次重新读取最新图像和机器人状态。具体步数不得在未做实机低速测试前写死。

## 4 推荐的正式实验流程

### 4.1 阶段 A 工程先导，不进入正式 200 次数据

目标：冻结硬件、任务和训练参数，而不是比较 A/B。

1. 完成真实机械臂、夹爪、相机和手机控制集成。
2. 校准显示分辨率、画面大小、帧率、亮度和端到端延迟。
3. 冻结两任务的物体尺寸、起始区域、目标区域、稳定判定和最大观察时间。
4. 冻结夹爪 action/state schema。
5. 用非正式 pilot episode 验证采集、同步、转换和回放。
6. 完成 SmolVLA 显存、训练时长和真实推理时延测试。
7. 冻结成功数据最低数量或“数据不足即不训练”的 go/no-go 规则。

禁止使用正式 A/B 结果反向调整上述参数。

### 4.2 阶段 B 参与者排程与正式采集

参与者分组：

| 组别 | 人数 | 条件顺序 | 每个条件的任务配额 |
|---|---:|---|---|
| G1 | 5 | A 后 B | 堆叠 3，分类 2 |
| G2 | 5 | B 后 A | 堆叠 3，分类 2 |
| G3 | 5 | A 后 B | 堆叠 2，分类 3 |
| G4 | 5 | B 后 A | 堆叠 2，分类 3 |

精确总量：

| 条件 | 堆叠 | 分类 | 合计 |
|---|---:|---:|---:|
| A | 50 | 50 | 100 |
| B | 50 | 50 | 100 |
| 合计 | 100 | 100 | 200 |

改进的排程要求：

- 生成正式清单时同时冻结 `global_trial_index`、`within_condition_trial_index`、`layout_id` 和任务顺序。
- 在满足 3:2 或 2:3 配额的前提下，平衡先做堆叠或先做分类的人数。
- 避免五次正式试验全部连续执行同一任务。
- 同一 `pair_id` 的 A/B 使用匹配的任务与布局定义，但实际执行顺序由参与者所属组决定。
- 若设置练习试次，必须使用 `is_practice=1` 明确标记，不计入 200 次，不进入训练和正式分析；练习次数和显示条件需在采集前冻结。

### 4.3 阶段 C 数据质量与人类表现分析

所有记录完整的成功和失败尝试都进入人类操作分析。建议字段：

```text
group_id
pair_id
participant_id
condition
task_id
layout_id
global_trial_index
within_condition_trial_index
condition_order
success
stage_score
completion_time_s
successful_grasps
grasp_retries
drop_count
manual_intervention
stable_for_2s
failure_type
video_latency_ms
dropped_frames
clock_uncertainty_s
recording_valid
sync_valid
training_eligible
exclusion_reason
```

分析原则：

- 完整成功率与阶段得分分开报告。
- 堆叠阶段分只能为 `0`、`0.5`、`1.0`。
- 分类阶段分只能为 `0`、`0.25`、`0.5`、`0.75`、`1.0`。
- 完整成功必须满足 `stage_score=1`、释放后稳定 2 秒且无人工接管。
- 失败 episode 仍可具有大于 0 的阶段得分。
- 先在“参与者×条件×任务”内聚合，再对两个任务等权汇总，避免 3:2 配额使某项任务被隐式赋予更高权重。
- 主要配对推断以参与者为单位；episode 不能被当作 200 个相互独立样本。
- 顺序、试次位置、视频延迟和掉帧作为预注册次要因素或敏感性分析变量。

### 4.4 阶段 D 主训练数据构建

训练候选必须同时满足：

```text
success == 1
stage_score == 1
manual_intervention == 0
recording_valid == 1
sync_valid == 1
夹爪动作和状态完整
视频、机器人状态和动作流通过质量门禁
```

主选择算法：

1. 构建四个候选池：A-堆叠、A-分类、B-堆叠、B-分类。
2. 计算 `N = min(四个池的合格数量)`。
3. 每个池固定选择 `N` 条。
4. 使用独立的 `selection_seed`，按参与者和 `layout_id` 分层抽样。
5. 输出不可变的 `training_selection.csv` 和输入文件哈希。
6. 三个模型随机种子都读取同一份选择文件。
7. 禁止根据速度、轨迹平滑度、重试次数或初步训练结果选择轨迹。

主模型使用全部被选中的平衡成功数据，不再为了离线测试额外损失正式训练样本。若需要数据集可学习性诊断，单独运行参与者隔离的交叉验证或 80/20 诊断划分；诊断 checkpoint 不得用于 240 次正式 rollout。

可选敏感性分析：只保留 A/B 两条都成功且质量合格的 `pair_id`，再按任务等量训练。该分析不取代主方案，并必须单独命名输出。

### 4.5 阶段 E SmolVLA 主训练

最低正式训练矩阵：

| 模型族 | 条件 | 种子数 | 运行数 |
|---|---|---:|---:|
| SmolVLA | A | 3 | 3 |
| SmolVLA | B | 3 | 3 |
| 合计 |  |  | 6 |

每次运行保存：

- 完整训练配置；
- 预训练模型标识和文件哈希；
- `training_selection.csv` 哈希；
- 数据集指纹；
- Git 状态；
- Python、PyTorch、CUDA 和 GPU 信息；
- 每个 checkpoint 的步数和验证信息；
- 最终用于 rollout 的 checkpoint 选择理由。

若使用固定最后一步 checkpoint，所有运行必须使用相同步数；若根据诊断数据选 checkpoint，选择规则必须预先冻结并在 A/B 间完全一致。

### 4.6 阶段 F 240 次真实机器人主评测

SmolVLA 主评测总数：

```text
2 个训练条件 × 3 个训练种子 × 2 个任务 × 20 次 = 240 次
```

每个“条件×种子×任务”包含：

- 10 次训练布局或训练位置范围；
- 10 次新位置；
- 相同的 20 个 `reset_id`；
- 相同的安全边界和终止规则；
- 随机或平衡的条件、种子和任务执行顺序；
- 可行时对策略来源实施评估者盲法。

每行至少记录：

```text
rollout_id
policy_family
training_condition
seed
task_id
layout_regime
reset_id
execution_order
success
stage_score
completion_time_s
grasp_retries
drop_count
manual_intervention
failure_type
safety_stop
inference_latency_ms
video_latency_ms
```

分析输出必须同时给出逐 seed 结果和跨 seed 汇总，不能只挑选最好种子。A/B 的主要比较使用相同 `task_id + layout_regime + reset_id + seed` 的匹配结构。

如果启用 ACT，需要额外 240 次并单独标为附加实验；不能把 ACT 试次计入 SmolVLA 的 240 次。

## 5 研究设计待冻结事项

以下事项必须在正式采集前由研究者确认。它们不是可以在实现时自行猜测的普通软件参数。

| ID | 待决定事项 | 推荐方向 | 状态 |
|---|---|---|---|
| D01 | A/B 的正式文字定义 | 保留现有 `A_mobile_colocated` 与 `B_desktop_separated` | DONE |
| D02 | 正式任务 | 仅堆叠和双色分类 | DONE |
| D03 | 参与者和正式尝试数 | 20 人、每人 10 次、总计 200 | DONE |
| D04 | 夹爪命令和状态单位 | 使用硬件可测的目标位置/宽度，保留持续目标值 | BLOCKED |
| D05 | 两任务最大观察时间 | 由工程先导确定并按任务冻结 | BLOCKED |
| D06 | 布局集合和新位置边界 | 建立版本化 `layout_id` 与坐标清单 | BLOCKED |
| D07 | 正式主训练数据范围 | 使用全部平衡后的合格成功轨迹，rollout 作为独立测试 | DONE |
| D08 | 最低合格成功轨迹数量 | 预注册 go/no-go 规则；不足时明确停止，不静默降低标准 | BLOCKED |
| D09 | SmolVLA batch、步数和精度 | 用非正式 pilot 做资源测试后冻结 | BLOCKED |
| D10 | rollout 每次实际执行的动作步数 | 使用短闭环窗口，实机低速测试后冻结 | BLOCKED |
| D11 | 是否运行 ACT | 默认不影响 SmolVLA 主实验；资源允许时单独启用 | BLOCKED |
| D12 | 正式练习试次数 | 若启用，固定数量并标记 `is_practice` | BLOCKED |
| D13 | 成功判定执行者 | 优先使用盲法人工判定加视频复核；可测项目自动计算 | BLOCKED |

## 6 实施进度清单

### M0 方案复核与基线

- [x] `M0-01` 完整读取《实验方案.docx》的正文、表格和论文截图。`DONE`
- [x] `M0-02` 对照当前配置、预检、同步、转换、训练、分析和 rollout 入口。`DONE`
- [x] `M0-03` 确认正式任务为两项而不是三个难度档。`DONE`
- [x] `M0-04` 发现并记录夹爪未进入正式训练数据链路的问题。`DONE`
- [x] `M0-05` 将改进后的方案和进度基线写入本文。`DONE`

### M1 协议与配置

- [x] `M1-01` 修改 `PROTOCOL.md` 为两任务、20 人、四组设计。`DONE`
- [x] `M1-02` 修改 `experiment_config.example.json`，删除简单/中等/困难占位任务。`DONE`
- [ ] `M1-03` 加入任务阶段评分、稳定 2 秒和任务级最大观察时间配置。`IN PROGRESS`（字段已加入，正式超时值待冻结）
- [ ] `M1-04` 加入参与者分组、任务顺序模板和布局清单配置。`IN PROGRESS`（分组与顺序已实现，真实布局目录待冻结）
- [x] `M1-05` 加入 `policy_families.smolvla` 和可选 `policy_families.act`。`DONE`
- [x] `M1-06` 加入训练选择种子、模型种子和 rollout 顺序种子，三者分离。`DONE`
- [ ] `M1-07` 记录 D04–D13 的最终研究者决定。`BLOCKED`

验收：示例配置能够完整表达正式实验，不含 `replace_with_*`，且没有把未冻结参数伪装成正式值。

### M2 夹爪采集与同步

- [ ] `M2-01` 实现真实 `GripperAdapter`。`BLOCKED`
- [ ] `M2-02` 定义并版本化夹爪 action/state schema。`IN PROGRESS`（通用契约已实现；真实 schema ID、单位与开合值待冻结）
- [ ] `M2-03` 采集 `gripper_actions.csv` 和 `gripper_states.csv`。`BLOCKED`（等待真实适配器）
- [ ] `M2-04` 在 raw episode 中保存 `gripper_schema.json`。`BLOCKED`（等待真实适配器）
- [x] `M2-05` 在 `validate_experiment_setup.py` 中验证夹爪文件、schema、范围和时间戳。`DONE`
- [x] `M2-06` 在 `time_sync.py` 中同步夹爪目标和实测状态。`DONE`（目标零阶保持、状态线性插值）
- [x] `M2-07` 将同步后的夹爪动作和状态写入统一训练 schema。`DONE`
- [ ] `M2-08` 增加开合、保持、异常时间戳和中断场景测试。`IN PROGRESS`（开合/保持与同步测试完成，真实中断场景待硬件验证）

验收：测试 episode 经同步后能完整复现夹爪闭合、保持和释放；不存在丢失的单帧 toggle。

### M3 正式清单与预检

- [x] `M3-01` 新增 `generate_experiment_manifest.py`。`DONE`
- [x] `M3-02` 生成 20 人、四组、200 行正式清单模板。`DONE`
- [x] `M3-03` 验证每人 A/B 各 5 次。`DONE`
- [x] `M3-04` 验证四个条件×任务单元格各 50 次。`DONE`
- [x] `M3-05` 验证每组恰好 5 人且顺序符合 G1–G4 定义。`DONE`
- [x] `M3-06` 验证任务顺序模板、`layout_id` 和 `pair_id`。`DONE`
- [x] `M3-07` 增加阶段分、稳定、重试、掉落、干预和质量字段检查。`DONE`
- [x] `M3-08` 更新 `experiment_manifest.example.csv` 与 `labels.example.csv`。`DONE`

验收：任何人数、组别、配额、顺序、阶段分或质量字段错误都会使正式预检失败。

### M4 数据转换与任务条件

- [x] `M4-01` 修改转换器，使其读取 manifest 而不是全局 `-Task`。`DONE`
- [x] `M4-02` 为每条 episode 写入固定 `policy_task_en`。`DONE`
- [x] `M4-03` 加入夹爪 action/state 特征。`DONE`
- [x] `M4-04` 输出 source episode 到 LeRobot episode 的显式映射。`DONE`
- [x] `M4-05` 校验 A/B 的图像、状态和动作 schema 完全一致。`DONE`（两条件共用冻结配置并逐 episode 校验）
- [x] `M4-06` 更新原子发布和失败清理测试。`DONE`

验收：转换后的两个数据集均含两类任务文本和完整夹爪通道，且 episode 映射可审计。

### M5 训练数据选择

- [x] `M5-01` 新增 `select_training_episodes.py`。`DONE`
- [x] `M5-02` 实现完整成功和数据质量门禁。`DONE`
- [x] `M5-03` 实现四单元格共同最小数量平衡。`DONE`
- [x] `M5-04` 按参与者和布局分层抽样。`DONE`
- [x] `M5-05` 输出选择原因、排除原因、随机种子和文件哈希。`DONE`
- [x] `M5-06` 验证三个模型种子读取完全相同的选择文件。`DONE`
- [x] `M5-07` 增加可选的完整配对成功敏感性选择模式。`DONE`（非主分析）
- [x] `M5-08` 强制执行预注册的每个条件×任务最低合格样本数。`DONE`（阈值本身待 pilot 冻结）

验收：同一输入和选择种子得到逐行一致的选择结果；A/B×任务数量完全相等。

### M6 SmolVLA 训练流水线

- [x] `M6-01` 将主入口改为 `--policy.path=lerobot/smolvla_base`。`DONE`
- [x] `M6-02` 按模型族选择 `training`/`smolvla` extras。`DONE`
- [x] `M6-03` 输出模型族×条件×种子的 checkpoint 清单。`DONE`
- [x] `M6-04` 强制校验 A/B 训练配置除数据集和输出标识外完全一致。`DONE`
- [x] `M6-05` 保存预训练模型和训练选择文件指纹。`DONE`
- [ ] `M6-06` 完成 8GB GPU 显存冒烟测试。`BLOCKED`
- [ ] `M6-07` 冻结 batch、步数、精度和 checkpoint 规则。`BLOCKED`
- [ ] `M6-08` 跑通两条件各一个短训练 smoke test。`TODO`
- [ ] `M6-09` 跑通两条件各三个正式随机种子。`TODO`
- [ ] `M6-10` 增加 ACT 并行训练。`OPTIONAL`

验收：六个 SmolVLA 运行除条件数据外配置相同，可从元数据重建全部训练命令。

### M7 人类演示统计

- [x] `M7-01` 确认并测试 `feature_extraction.py` 一对一保留新增结果与质量字段。`DONE`
- [x] `M7-02` 将 `condition × difficulty` 改为 `condition × task`，并删除旧难度接口。`DONE`
- [x] `M7-03` 在参与者×条件×任务内先聚合，再对任务等权汇总。`DONE`
- [x] `M7-04` 分别分析完整成功率和阶段得分。`DONE`（阶段分作为连续特征，成功率单独检验）
- [ ] `M7-05` 增加顺序、trial index、延迟和掉帧敏感性分析。`TODO`
- [x] `M7-06` 输出效应量、置信区间和多特征 FDR 结果。`DONE`（配对/独立诊断均输出描述性 95% t 区间）
- [x] `M7-07` 增加不把 episode 当独立参与者且任务等权的回归测试。`DONE`

验收：改变单个参与者的重复 episode 数量不会把统计样本量错误地增加为独立参与者数量。

### M8 真实机器人 rollout

- [x] `M8-01` 新增 240 行 rollout manifest 生成器。`DONE`
- [x] `M8-02` 校验每个条件×种子×任务为 20 次且 10/10 布局平衡。`DONE`
- [x] `M8-03` 冻结共享 `reset_id` 和执行顺序。`DONE`
- [x] `M8-04` 实现统一 rollout 结果 schema。`DONE`
- [ ] `M8-05` 实现短窗口闭环执行和推理延迟记录。`BLOCKED`
- [ ] `M8-06` 接入真实机器人、夹爪和独立急停。`BLOCKED`
- [x] `M8-07` 新增 rollout 结果完整性检查。`DONE`
- [x] `M8-08` 新增逐种子、逐任务、逐布局分析脚本。`DONE`（含描述性配对 95% t 区间；确认性方法待预注册）
- [ ] `M8-09` 完成 240 次 SmolVLA 主评测。`BLOCKED`
- [ ] `M8-10` 完成额外 240 次 ACT 评测。`OPTIONAL`

验收：缺少任意计划试次、重复 `rollout_id`、reset 不匹配或配额错误时，结果分析必须拒绝运行。

### M9 文档、测试与发布门禁

- [ ] `M9-01` 更新实验 README 和本地运行指南。`IN PROGRESS`（README 已更新，本地运行指南待同步）
- [ ] `M9-02` 将 `Reprot/armcam_bc` 明确标记为旧实验，不作为正式入口。`TODO`
- [x] `M9-03` 更新新版协议相关测试夹具并保留旧版兼容测试。`DONE`
- [x] `M9-04` 运行实验目录定向测试。`DONE`（当前 210 passed）
- [ ] `M9-05` 运行受影响的数据集、训练和 processor 测试。`TODO`
- [x] `M9-06` 运行格式、静态检查和预检 smoke test。`DONE`（Ruff、PowerShell 语法和预期停止门均验证）
- [ ] `M9-07` 保存最终协议、配置、清单模板和测试结果哈希。`TODO`

验收：文档命令可复制运行，示例配置不会伪装成可直接执行的正式硬件配置，所有相关测试通过。

## 7 文件修改映射

| 文件 | 修改目的 | 当前状态 |
|---|---|---|
| `PROTOCOL.md` | 冻结两任务、四组、成功定义、训练选择和 rollout 方案 | DONE |
| `README.md` | 更新配置、采集、训练和评测命令 | DONE |
| `experiment_config.example.json` | 新的任务、分组、模型族和评测配置 | IN PROGRESS（硬件值待冻结） |
| `experiment_manifest.example.csv` | 新的 200 次正式清单字段 | DONE |
| `labels.example.csv` | 阶段分、失败模式和质量字段 | DONE |
| `acquisition_interfaces.py` | 冻结真实夹爪扩展契约 | BLOCKED |
| `robot_bridge.py` | 记录并执行夹爪动作，接入独立急停 | BLOCKED |
| `gripper_contract.py` | 统一夹爪配置、每回合 schema 和取值范围契约 | DONE（真实值待冻结） |
| `time_sync.py` | 同步夹爪目标和实测状态 | DONE（真实采集待验证） |
| `convert_raw_to_lerobot.py` | 多任务文本、夹爪特征和 episode 映射 | DONE（真实数据待验证） |
| `validate_experiment_setup.py` | 强制 20 人、四组、200 次、目录和全量 schema | DONE（真实输入待验证） |
| `experiment_catalogs.py` / `CATALOG_FORMAT.md` | 校验版本化布局和复位坐标目录 | DONE（真实坐标待填写） |
| `make_episode_splits.py` | 降级为诊断划分工具，或重命名说明用途 | TODO |
| `run_pipeline.ps1` | 主数据选择、SmolVLA、多种子和 rollout 清单 | DONE（硬件运行受停止门约束） |
| `feature_extraction.py` | 新结果字段和任务级特征 | DONE（通用标签一对一保留） |
| `features_compare.py` | 参与者级 `condition × task` 分析 | DONE（含任务等权、阶段分、效应量、FDR 与描述性区间） |
| `generate_experiment_manifest.py` | 生成正式 200 行排程 | DONE |
| `select_training_episodes.py` | 冻结平衡训练集 | DONE |
| `make_rollout_manifest.py` | 生成 240 行实机评测计划 | DONE |
| `analyze_rollouts.py` | 汇总主要模型结果 | DONE（含描述性配对区间；确认性推断待预注册） |
| `tests/experiments/wrist_view_presentation/` | 覆盖新协议和失败门禁 | DONE（当前 210 passed） |

## 8 正式采集前的停止条件

只要以下任一项未完成，就不应开始正式 20 人采集：

- 真实夹爪 action/state schema 未冻结；
- 真实夹爪适配器尚未证明可持续生成符合契约的数据并完成回放；
- 任务布局、成功标准、稳定 2 秒判定或最大观察时间未冻结；
- A/B 显示的分辨率、画面大小、帧率和延迟没有校准记录；
- 200 行正式清单不能通过配额与顺序预检；
- 原始视频、动作、状态和夹爪流不能通过完整回放；
- SmolVLA 在目标 GPU 上不能完成最小 smoke test；
- 真实 rollout 尚无短窗口闭环、watchdog 和独立急停；
- 最低合格成功轨迹数量尚未由 pilot 冻结（软件已强制执行该门槛）。

## 9 进度更新规则

每次修改后按以下方式更新本文：

1. 只在对应实现和测试均完成后把项目改为 `DONE`。
2. `BLOCKED` 项解除时，在决策表记录最终值、决定日期和依据。
3. 若研究设计发生变化，先更新第 5 节决策，再修改代码。
4. 每次完成一个里程碑，在下方进度日志追加一行，不覆盖历史记录。

## 10 进度日志

| 日期 | 里程碑 | 变更 | 验证 | 结果 |
|---|---|---|---|---|
| 2026-09-24 | M0 | 复核实验方案与现有框架，建立修改和进度基线 | 对照 DOCX、配置、预检、同步、转换、训练、分析和 rollout 入口 | DONE |
| 2026-09-24 | M1/M3–M8 | 完成 v2 配置、200 行清单、逐回合指令、等量训练选择、SmolVLA 编排、240 行评测与配对汇总第一批实现 | 清理后 206 个实验目录测试通过；Ruff 通过；PowerShell 语法通过；示例配置正确触发 13 个未冻结参数停止门 | DONE（硬件项除外） |
| 2026-09-24 | Cleanup | 删除未引用的旧 checkpoint 示例、停用的诊断切分配置、旧难度统计兼容层和未使用的 episode 级成功比较 | 引用审计、JSON 解析、Ruff 与定向回归测试 | DONE |
| 2026-09-24 | M2/M4/M5/M7 | 完成通用夹爪同步/校验/转换、目录内容校验、最低样本量停止门、部署参数占位与描述性 95% 区间 | `210 passed`；Ruff 通过；PowerShell 语法通过；示例配置触发 `21 error + 1 warning` 的预期停止门 | DONE（真实参数与硬件适配除外） |

## 11 反向审计：剩余修改空间与代码证据

> 本节保留 2026-09-24 首次审计快照用于对照；其中已经完成的缺口以第 12 节二次审计为准。

审计日期：2026-09-24。方法：从 `run_pipeline.ps1` 反向追踪正式入口，分别核对配置、原始数据预检、同步、转换、训练、rollout 和统计输出；同时运行实验目录测试、Ruff、PowerShell 语法解析和示例配置预检。

结论：软件设计骨架已经成立，但正式采集闭环尚未成立。以下项目按优先级排序。

### P0 必须完成，否则不能正式采集或训练

1. **夹爪数据链路仍是断开的。**
   - 配置和协议要求 `gripper_actions.csv`、`gripper_states.csv`、`gripper_schema.json`。
   - `time_sync.py` 实际只要求并读取 `robot.csv`、`phone.csv`、`applied_actions.csv`。
   - `convert_raw_to_lerobot.py` 的 `ACTION_NAMES` 固定为 6D，输出 action 也固定为 6 维。
   - `validate_raw_tree()` 只检查夹爪文件是否存在，没有验证夹爪 CSV 表头、时间戳、范围、schema 和时间交集。
   - 修改位置：`acquisition_interfaces.py`、`robot_bridge.py`、`time_sync.py`、`validate_experiment_setup.py`、`convert_raw_to_lerobot.py` 及对应测试。

2. **真实 rollout 适配器和短时域执行参数未实现。**
   - 正式流水线仍依赖外部 `-RolloutScript`。
   - 当前仓库 SmolVLA 默认 `n_action_steps=50`，实验配置没有冻结覆盖值；直接使用默认队列不满足短窗口视觉闭环要求。
   - 修改位置：`experiment_config.example.json` 增加部署参数；预检强制 `n_action_steps`、控制频率、watchdog 和命令过期阈值；实现硬件 rollout 适配器并接入独立急停。

3. **布局与 reset 目录只验证“路径字符串”，没有验证内容。**
   - 当前预检只判断 `layout_catalog_path` 和 `reset_catalog_path` 是否为非空字符串。
   - 未检查文件存在、版本、坐标单位、任务归属、ID 唯一性、manifest 引用完整性以及训练布局/新位置边界。
   - 修改位置：新增版本化 catalog schema；在 `validate_experiment_setup.py` 中加载并交叉校验两个目录。

4. **训练数据缺少最低样本量 go/no-go 门槛。**
   - `select_training_episodes.py` 目前只在某个单元为零时停止，然后直接取四单元共同最小值。
   - 若每个单元只有极少数合格轨迹，脚本仍会继续训练。
   - 修改位置：在配置中冻结 `minimum_eligible_per_condition_task`，在选择器和预检中强制执行并输出停止原因。

### P1 应在正式分析前完成

1. `analyze_rollouts.py` 目前只有均值、标准差和 A−B 描述性差值，没有置信区间或预注册的确认性检验。
2. `features_compare.py` 尚未实现 trial index、视频延迟、掉帧和时钟不确定度的敏感性模型，也没有置信区间。
3. 示例配置中的两任务超时、时钟不确定度、SmolVLA batch/steps 仍为 `null`；预检会正确阻止运行，但需要 pilot 给出真实值。
4. 尚未对目标 8GB GPU 运行两条件短训练 smoke test，也未验证 checkpoint 能被 rollout 适配器加载。

### P2 清理与结构优化

1. `make_episode_splits.py` 仍使用旧 `difficulty` 设计，并被 v1 预检兼容分支引用；v2 主流程不调用它。可在确认不再支持 v1 后，一并删除旧验证分支和对应测试。
2. `offline_compare.py` 仍包含旧 ACT/单 checkpoint 诊断入口；v2 主流水线不调用，但现有测试仍覆盖它。应由研究者决定“保留为独立诊断工具”或整体移入 `legacy/`。
3. `convert_raw_to_lerobot.py` 仍保留全局 `--task` 旧入口；正式流程只使用 manifest/config/condition，可在不再兼容历史数据后删除。
4. `features_compare.py` 仍保留 `--independent_diagnostic`；若论文只允许参与者内配对主分析，可移除以缩小误用面。

### 审计自证结果

- 实验目录回归测试：`206 passed`。
- Ruff 静态检查：通过。
- PowerShell 语法解析：通过。
- 示例配置预检：按预期得到 `13 error + 1 warning`；错误均对应未冻结的硬件或研究参数。
- 因此，当前状态可以证明“v2 软件骨架和停止门可工作”，但不能证明“夹爪闭环、实机安全、模型训练和 240 次 rollout 已完成”。

## 12 二次审计：无需真实参数的修改结果

审计日期：2026-09-24。本节是当前有效结论，并覆盖第 11 节中已经完成的旧缺口。

### 已完成

1. **通用夹爪数据链路**
   - `gripper_contract.py` 分别校验 action/state schema、列名、单位、开合端点和合法范围，允许命令与实测状态采用不同单位。
   - `time_sync.py` 将目标命令按零阶保持同步，将实测状态线性插值，并原样复制冻结 schema。
   - `validate_experiment_setup.py` 校验文件、schema、时间戳、范围和所有流的共同时间区间。
   - `convert_raw_to_lerobot.py` 把夹爪实测值加入 state，把夹爪目标加入 action；正式 v2 模式缺少夹爪数据时拒绝转换。

2. **版本化布局与复位目录**
   - `experiment_catalogs.py` 校验文件存在性、版本、坐标系、米/弧度单位、有限三维位姿、ID 唯一性和配置精确覆盖。
   - `CATALOG_FORMAT.md` 给出结构说明，但明确禁止把示例零坐标当作真实实验值。

3. **训练与部署停止门**
   - `minimum_eligible_per_condition_task` 已接入协议预检、训练选择器和主流水线；任一条件×任务单元不足即明确停止。
   - SmolVLA 的 `chunk_size`、`n_action_steps`，以及控制频率、watchdog、命令过期时间均已成为强制冻结字段。
   - 示例配置继续用 `null`，防止未经 pilot 验证的值被误当作正式参数。

4. **统计输出**
   - 人类演示的连续特征和参与者成功率差异输出描述性 95% t 区间。
   - rollout 的每个 A−B 配对汇总输出描述性 95% t 区间。
   - 这些区间不替代尚待预注册的确认性模型和多重比较方案。

5. **清理与文档修正**
   - 主流水线只向具备相应字段的模型族传递 SmolVLA 动作块参数，避免误传给 ACT。
   - README 中选择器和转换器的旧命令参数已修正。

### 仍需真实数据或研究者决定

| 项目 | 需要提供或通过 pilot 决定的内容 | 当前保护措施 |
|---|---|---|
| 夹爪 | schema ID、单位、开/合值、真实适配器输出 | `null`/缺文件即预检失败；逐回合严格校验 |
| 时间同步 | 最大时钟不确定度 | 未冻结即预检失败 |
| 两项任务 | 最大观察时间 | 未冻结即预检失败 |
| 布局与复位 | 真实对象坐标、姿态及目录文件 | 目录内容与配置不精确一致即失败 |
| 训练样本量 | 每个条件×任务最低合格数 | 未冻结或实际数量不足即失败 |
| SmolVLA | batch、steps、chunk size、每次执行步数 | 未冻结即失败；`n_action_steps > chunk_size` 被拒绝 |
| 部署安全 | 控制频率、watchdog、命令过期阈值、急停和碰撞边界 | 未冻结即失败；流水线仍要求外部硬件 rollout 适配器 |
| 统计方案 | 确认性模型、多重比较和缺失数据规则 | 当前结果明确标记为描述性 |

### 当前验证证据

- 实验目录测试：`210 passed`。
- Ruff：通过。
- `run_pipeline.ps1` PowerShell 语法：通过。
- 示例配置预检：按预期得到 `21 error + 1 warning`；21 个错误全部对应故意保留的真实参数或目录停止门。
- 这些证据证明软件契约和失败门禁可工作，不证明真实夹爪、急停、碰撞保护、GPU 训练或 240 次实机 rollout 已完成。
