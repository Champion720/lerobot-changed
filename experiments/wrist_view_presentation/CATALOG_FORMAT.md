# 布局与复位目录格式

正式采集布局和真实机器人 rollout 复位位置必须由版本化 JSON 目录描述。仓库不填入虚构坐标；请在标定完成后录入真实测量值，再由预检程序检查。

## 采集布局目录

`layout_catalog_path` 指向的文件格式如下：

```json
{
  "schema_version": 1,
  "coordinate_frame": "robot_base",
  "position_unit": "m",
  "angle_unit": "rad",
  "layouts": [
    {
      "task_id": "stacking",
      "layout_id": "stacking_layout_01",
      "objects": {
        "red_cube": {
          "position_xyz": [0.0, 0.0, 0.0],
          "orientation_rpy": [0.0, 0.0, 0.0]
        }
      }
    }
  ]
}
```

示例中的零值仅展示结构，不可直接作为实验坐标。`layouts` 必须精确覆盖配置中所有 `collection_layout_ids`，不得遗漏或额外添加 ID。

## Rollout 复位目录

`evaluation.reset_catalog_path` 指向的文件使用相同坐标约定：

```json
{
  "schema_version": 1,
  "coordinate_frame": "robot_base",
  "position_unit": "m",
  "angle_unit": "rad",
  "resets": [
    {
      "task_id": "stacking",
      "layout_regime": "train_layout",
      "reset_id": "stack_train_01",
      "objects": {
        "red_cube": {
          "position_xyz": [0.0, 0.0, 0.0],
          "orientation_rpy": [0.0, 0.0, 0.0]
        }
      }
    }
  ]
}
```

`layout_regime` 只能是 `train_layout` 或 `novel_position`。`resets` 必须精确覆盖 `evaluation.reset_ids` 的全部任务、布局类别和复位 ID。

## 校验

目录路径相对于实验配置文件所在目录解析。填写后运行：

```powershell
uv run --extra training python experiments/wrist_view_presentation/validate_experiment_setup.py `
  --config experiments/wrist_view_presentation/experiment_config.json `
  --manifest experiments/wrist_view_presentation/experiment_manifest.csv
```

预检会拒绝：文件不存在、schema 版本错误、坐标单位错误、非有限坐标、重复 ID、未知任务/布局类别，以及目录与配置引用不完全一致。
