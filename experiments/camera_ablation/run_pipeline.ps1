# Legacy smoke-test pipeline (run from the inner lerobot-main project dir).
# This still validates the old "remove camera columns" training chain on a public dataset.
# It is NOT the current real experiment. The current A/B design is:
#   A = camera stream displayed on the phone client
#   B = camera stream displayed on the PC client
# and both real datasets should contain video.

$ErrorActionPreference = "Stop"

$env:Path = "C:\Users\19034\.local\bin;$env:Path"
$env:HF_HUB_DISABLE_XET = "1"

$RepoFull  = "lerobot/svla_so101_pickplace"
$RepoNoCam = "svla_so101_pickplace_nocam"
$Steps     = 5000
$Batch     = 8

Write-Host "=== [1/4] Legacy smoke test: building no-camera copy ===" -ForegroundColor Cyan
uv run --extra training python experiments/camera_ablation/prepare_condition_a.py `
  --repo_id $RepoFull --out_repo_id $RepoNoCam
if (-not $?) { throw "Step 1 (prepare) failed" }

Write-Host "=== [2/4] Legacy smoke test: training full public dataset ===" -ForegroundColor Cyan
uv run --extra training lerobot-train --dataset.repo_id=$RepoFull `
  --dataset.video_backend=pyav --policy.type=act --policy.device=cuda `
  --policy.push_to_hub=false --output_dir=outputs/train/cond_b --job_name=cond_b `
  --batch_size=$Batch --steps=$Steps --eval_freq=0 --num_workers=0 --wandb.enable=false
if (-not $?) { throw "Step 2 (train B) failed" }

Write-Host "=== [3/4] Legacy smoke test: training no-camera copy ===" -ForegroundColor Cyan
uv run --extra training lerobot-train --dataset.repo_id=$RepoNoCam `
  --dataset.video_backend=pyav --policy.type=act --policy.device=cuda `
  --policy.push_to_hub=false --output_dir=outputs/train/cond_a --job_name=cond_a `
  --batch_size=$Batch --steps=$Steps --eval_freq=0 --num_workers=0 --wandb.enable=false
if (-not $?) { throw "Step 3 (train A) failed" }

Write-Host "=== [4/4] Legacy smoke test: offline comparison + paired t-test ===" -ForegroundColor Cyan
uv run --extra training python experiments/camera_ablation/offline_compare.py `
  --ckpt_a outputs/train/cond_a/checkpoints/last/pretrained_model --repo_a $RepoNoCam `
  --ckpt_b outputs/train/cond_b/checkpoints/last/pretrained_model --repo_b $RepoFull `
  --test_frac 0.2 --device cuda --paired
if (-not $?) { throw "Step 4 (compare) failed" }

Write-Host "=== Pipeline finished ===" -ForegroundColor Green
