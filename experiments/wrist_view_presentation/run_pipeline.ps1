param(
  [string]$Config = "experiments/wrist_view_presentation/experiment_config.json",
  [string]$Manifest = "experiments/wrist_view_presentation/experiment_manifest.csv",
  [string]$RawTimestampRoot = "raw_ts",
  [string]$SyncedRoot = "raw",
  [string]$SplitManifest = "outputs/wrist_view_presentation/participant_splits.csv",
  [string]$DhConfig = "experiments/wrist_view_presentation/dh_params.json",
  [string]$RobotBridgeConfig = "experiments/wrist_view_presentation/robot_bridge_config.json",
  [string]$Task,
  [string]$Resize = "480x640",
  [string]$CollectionScript,
  [string]$RolloutScript,
  [string[]]$CollectionArguments = @(),
  [string[]]$RolloutArguments = @()
)

# Formal study pipeline. Run from the repository root containing pyproject.toml.
# CollectionScript and RolloutScript are hardware-specific adapters. The collection
# adapter must publish both formal condition directories under RawTimestampRoot. The
# rollout adapter must evaluate both checkpoint maps on the same preregistered real-
# robot task suite and write trial-level outcomes; this repository cannot supply a
# vendor driver or silently substitute an offline metric for that evidence.

$ErrorActionPreference = "Stop"
$StudyDir = "experiments/wrist_view_presentation"
$ConditionA = "A_mobile_colocated"
$ConditionB = "B_desktop_separated"
$OutputRoot = "outputs/wrist_view_presentation"

function Invoke-Uv {
  param([Parameter(Mandatory = $true)][string[]]$Arguments)
  & uv @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "uv command failed with exit code ${LASTEXITCODE}: uv $($Arguments -join ' ')"
  }
}

function Get-ConditionRun {
  param([Parameter(Mandatory = $true)][object]$Protocol, [Parameter(Mandatory = $true)][string]$Condition)
  return $Protocol.training.condition_runs.PSObject.Properties[$Condition].Value
}

if (-not (Test-Path -LiteralPath "pyproject.toml" -PathType Leaf)) {
  throw "Run this script from the repository root containing pyproject.toml."
}
foreach ($RequiredPath in @($Config, $Manifest, $DhConfig, $RobotBridgeConfig)) {
  if (-not (Test-Path -LiteralPath $RequiredPath -PathType Leaf)) {
    throw "Missing required formal-study input: $RequiredPath"
  }
}
if ([string]::IsNullOrWhiteSpace($Task)) {
  throw "-Task is required and must match the preregistered task semantics."
}

$Protocol = Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
$RunA = Get-ConditionRun -Protocol $Protocol -Condition $ConditionA
$RunB = Get-ConditionRun -Protocol $Protocol -Condition $ConditionB
$TargetFps = [double]$Protocol.capture.target_fps
$Seeds = @($Protocol.training.random_seeds)
$SplitSeed = [int]$Protocol.training.split_seed
$MinParticipants = [int]$Protocol.analysis.minimum_participants_per_split

Write-Host "=== [1/8] Collect or verify paired wrist-view episodes ===" -ForegroundColor Cyan
if ($CollectionScript) {
  if (-not (Test-Path -LiteralPath $CollectionScript -PathType Leaf)) {
    throw "Collection adapter does not exist: $CollectionScript"
  }
  & $CollectionScript @CollectionArguments
  if (-not $?) {
    throw "Collection adapter failed"
  }
}
foreach ($Condition in @($ConditionA, $ConditionB)) {
  $ConditionRoot = Join-Path $RawTimestampRoot $Condition
  if (-not (Test-Path -LiteralPath $ConditionRoot -PathType Container)) {
    throw "Missing collected condition directory: $ConditionRoot"
  }
}
Invoke-Uv -Arguments @(
  "run", "--extra", "training", "python", "$StudyDir/validate_experiment_setup.py",
  "--config", $Config, "--manifest", $Manifest, "--raw_root", $RawTimestampRoot
)

Write-Host "=== [2/8] Synchronize robot, applied-action, and per-frame video clocks ===" -ForegroundColor Cyan
foreach ($Condition in @($ConditionA, $ConditionB)) {
  Invoke-Uv -Arguments @(
    "run", "--extra", "training", "python", "$StudyDir/time_sync.py",
    "--in_dir", (Join-Path $RawTimestampRoot $Condition),
    "--out_dir", (Join-Path $SyncedRoot $Condition),
    "--out_fps", "$TargetFps", "--robot_bridge_config", $RobotBridgeConfig
  )
}

Write-Host "=== [3/8] Convert both video-bearing datasets to LeRobot ===" -ForegroundColor Cyan
foreach ($Condition in @($ConditionA, $ConditionB)) {
  $Run = Get-ConditionRun -Protocol $Protocol -Condition $Condition
  Invoke-Uv -Arguments @(
    "run", "--extra", "training", "python", "$StudyDir/convert_raw_to_lerobot.py",
    "--raw_dir", (Join-Path $SyncedRoot $Condition),
    "--repo_id", $Run.dataset_repo_id, "--fps", "$TargetFps", "--task", $Task,
    "--dh_config", $DhConfig, "--resize", $Resize
  )
}

Write-Host "=== [4/8] Freeze participant-safe train/validation/test splits ===" -ForegroundColor Cyan
Invoke-Uv -Arguments @(
  "run", "--extra", "training", "python", "$StudyDir/make_episode_splits.py",
  "--manifest", $Manifest, "--out", $SplitManifest, "--seed", "$SplitSeed",
  "--train_fraction", "$($Protocol.training.train_fraction)",
  "--validation_fraction", "$($Protocol.training.validation_fraction)",
  "--test_fraction", "$($Protocol.training.test_fraction)",
  "--min_participants_per_split", "$MinParticipants"
)
$SplitRows = Import-Csv -LiteralPath $SplitManifest

Write-Host "=== [5/8] Train paired multi-seed policies with one shared configuration ===" -ForegroundColor Cyan
$CheckpointA = [ordered]@{}
$CheckpointB = [ordered]@{}
foreach ($Seed in $Seeds) {
  foreach ($Condition in @($ConditionA, $ConditionB)) {
    $Run = Get-ConditionRun -Protocol $Protocol -Condition $Condition
    $TrainEpisodes = @(
      $SplitRows |
        Where-Object { $_.condition -eq $Condition -and $_.split -eq "train" } |
        ForEach-Object { [int]$_.episode }
    )
    if ($TrainEpisodes.Count -eq 0) {
      throw "No training episodes were assigned for $Condition"
    }
    $EpisodeJson = ConvertTo-Json -InputObject $TrainEpisodes -Compress
    $TrainOutput = "$OutputRoot/train/$Condition/seed_$Seed"
    $TrainArguments = @(
      "run", "--extra", "training", "lerobot-train",
      "--dataset.repo_id=$($Run.dataset_repo_id)", "--dataset.episodes=$EpisodeJson",
      "--dataset.video_backend=$($Run.video_backend)", "--policy.type=$($Run.policy_type)",
      "--policy.device=$($Run.device)",
      "--policy.push_to_hub=$($Run.push_to_hub.ToString().ToLowerInvariant())",
      "--seed=$Seed", "--output_dir=$TrainOutput", "--job_name=${Condition}_seed_$Seed",
      "--batch_size=$($Run.batch_size)", "--steps=$($Run.steps)",
      "--eval_freq=$($Run.eval_freq)", "--num_workers=$($Run.num_workers)",
      "--wandb.enable=$($Run.wandb_enabled.ToString().ToLowerInvariant())"
    )
    if ($Run.extra_cli_args) {
      $TrainArguments += @($Run.extra_cli_args)
    }
    Invoke-Uv -Arguments $TrainArguments
    $Checkpoint = "$TrainOutput/checkpoints/last/pretrained_model"
    if ($Condition -eq $ConditionA) { $CheckpointA["$Seed"] = $Checkpoint }
    else { $CheckpointB["$Seed"] = $Checkpoint }
  }
}
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$CheckpointAPath = "$OutputRoot/checkpoints_A_mobile_colocated.json"
$CheckpointBPath = "$OutputRoot/checkpoints_B_desktop_separated.json"
$CheckpointA | ConvertTo-Json | Set-Content -LiteralPath $CheckpointAPath -Encoding UTF8
$CheckpointB | ConvertTo-Json | Set-Content -LiteralPath $CheckpointBPath -Encoding UTF8

Write-Host "=== [6/8] Compare both policies on the same frozen test split ===" -ForegroundColor Cyan
Invoke-Uv -Arguments @(
  "run", "--extra", "training", "python", "$StudyDir/offline_compare.py",
  "--checkpoints_a_json", $CheckpointAPath, "--repo_a", $RunA.dataset_repo_id,
  "--checkpoints_b_json", $CheckpointBPath, "--repo_b", $RunB.dataset_repo_id,
  "--protocol_config", $Config, "--split_manifest", $SplitManifest, "--split", "test",
  "--out", "$OutputRoot/offline_action_errors.csv", "--device", $RunA.device,
  "--video_backend", $RunA.video_backend
)

Write-Host "=== [7/8] Extract and compare participant-level demonstration features ===" -ForegroundColor Cyan
$LabelsA = "$OutputRoot/labels_A_mobile_colocated.csv"
$LabelsB = "$OutputRoot/labels_B_desktop_separated.csv"
$SplitRows | Where-Object { $_.condition -eq $ConditionA } | Export-Csv -LiteralPath $LabelsA -NoTypeInformation
$SplitRows | Where-Object { $_.condition -eq $ConditionB } | Export-Csv -LiteralPath $LabelsB -NoTypeInformation
$FeaturesA = "$OutputRoot/features_A_mobile_colocated.csv"
$FeaturesB = "$OutputRoot/features_B_desktop_separated.csv"
Invoke-Uv -Arguments @(
  "run", "--extra", "training", "python", "$StudyDir/feature_extraction.py",
  "--repo_id", $RunA.dataset_repo_id, "--condition", $ConditionA,
  "--labels_csv", $LabelsA, "--out", $FeaturesA, "--video_backend", $RunA.video_backend
)
Invoke-Uv -Arguments @(
  "run", "--extra", "training", "python", "$StudyDir/feature_extraction.py",
  "--repo_id", $RunB.dataset_repo_id, "--condition", $ConditionB,
  "--labels_csv", $LabelsB, "--out", $FeaturesB, "--video_backend", $RunB.video_backend
)
Invoke-Uv -Arguments @(
  "run", "--extra", "training", "python", "$StudyDir/features_compare.py",
  "--features_a", $FeaturesA, "--features_b", $FeaturesB,
  "--out", "$OutputRoot/feature_comparison.csv",
  "--excel", "$OutputRoot/feature_comparison.xlsx"
)

Write-Host "=== [8/8] Run the shared real-robot rollout suite ===" -ForegroundColor Cyan
if (-not $RolloutScript) {
  throw (
    "Offline stages completed, but formal evidence is incomplete. Supply -RolloutScript " +
    "with the hardware-specific shared-suite adapter; offline loss/MSE cannot replace rollout."
  )
}
if (-not (Test-Path -LiteralPath $RolloutScript -PathType Leaf)) {
  throw "Rollout adapter does not exist: $RolloutScript"
}
& $RolloutScript --config $Config --split_manifest $SplitManifest `
  --checkpoints_a_json $CheckpointAPath --checkpoints_b_json $CheckpointBPath `
  --output_dir "$OutputRoot/rollout" @RolloutArguments
if (-not $?) {
  throw "Rollout adapter failed"
}

Write-Host "=== Formal wrist-view presentation pipeline finished ===" -ForegroundColor Green
