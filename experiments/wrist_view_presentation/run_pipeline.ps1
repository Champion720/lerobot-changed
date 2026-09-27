param(
  [string]$Config = "experiments/wrist_view_presentation/experiment_config.json",
  [string]$Manifest = "experiments/wrist_view_presentation/experiment_manifest.csv",
  [string]$RawTimestampRoot = "raw_ts",
  [string]$SyncedRoot = "raw",
  [string]$DhConfig = "experiments/wrist_view_presentation/dh_params.json",
  [string]$RobotBridgeConfig = "experiments/wrist_view_presentation/robot_bridge_config.json",
  [string]$SelectionManifest = "outputs/wrist_view_presentation/training_selection.csv",
  [string]$SelectionSummary = "outputs/wrist_view_presentation/training_selection.json",
  [string]$Resize = "480x640",
  [string]$CollectionScript,
  [string]$RolloutScript,
  [string[]]$CollectionArguments = @(),
  [string[]]$RolloutArguments = @()
)

# Formal two-task pipeline. The hardware-specific collection adapter must publish
# both condition directories. The rollout adapter must consume the frozen rollout
# manifest and write trial-level outcomes. Offline loss is never substituted for
# the preregistered real-robot evidence.

$ErrorActionPreference = "Stop"
$StudyDir = "experiments/wrist_view_presentation"
$Conditions = @("A_mobile_colocated", "B_desktop_separated")
$OutputRoot = "outputs/wrist_view_presentation"

function Invoke-Uv {
  param([Parameter(Mandatory = $true)][string[]]$Arguments)
  & uv @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "uv command failed with exit code ${LASTEXITCODE}: uv $($Arguments -join ' ')"
  }
}

function Get-ObjectProperty {
  param(
    [Parameter(Mandatory = $true)][object]$Object,
    [Parameter(Mandatory = $true)][string]$Name
  )
  $Property = $Object.PSObject.Properties[$Name]
  if ($null -eq $Property) {
    throw "Missing required configuration property: $Name"
  }
  return $Property.Value
}

function Get-DatasetId {
  param([Parameter(Mandatory = $true)][object]$Protocol, [Parameter(Mandatory = $true)][string]$Condition)
  return Get-ObjectProperty -Object $Protocol.training.condition_datasets -Name $Condition
}

function Get-MappedEpisodes {
  param(
    [Parameter(Mandatory = $true)][object[]]$Rows,
    [Parameter(Mandatory = $true)][string]$MapPath
  )
  $MapPayload = Get-Content -LiteralPath $MapPath -Raw | ConvertFrom-Json
  $EpisodeMap = @{}
  foreach ($Entry in @($MapPayload.episodes)) {
    $EpisodeMap[[string][int]$Entry.source_episode_id] = [int]$Entry.lerobot_episode_index
  }
  $Mapped = @()
  foreach ($Row in $Rows) {
    $Key = [string][int]$Row.episode
    if (-not $EpisodeMap.ContainsKey($Key)) {
      throw "Selected source episode $Key is absent from $MapPath"
    }
    $Mapped += $EpisodeMap[$Key]
  }
  return @($Mapped | Sort-Object -Unique)
}

if (-not (Test-Path -LiteralPath "pyproject.toml" -PathType Leaf)) {
  throw "Run this script from the repository root containing pyproject.toml."
}
foreach ($RequiredPath in @($Config, $Manifest, $DhConfig, $RobotBridgeConfig)) {
  if (-not (Test-Path -LiteralPath $RequiredPath -PathType Leaf)) {
    throw "Missing required formal-study input: $RequiredPath"
  }
}

$Protocol = Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
$TargetFps = [double]$Protocol.capture.target_fps
$Seeds = @($Protocol.training.random_seeds)
$SelectionSeed = [int]$Protocol.training.selection_seed
$PrimaryFamily = [string]$Protocol.evaluation.policy_family

Write-Host "=== [1/8] Validate protocol and collect paired formal episodes ===" -ForegroundColor Cyan
Invoke-Uv -Arguments @(
  "run", "--extra", "training", "python", "$StudyDir/validate_experiment_setup.py",
  "--config", $Config, "--manifest", $Manifest
)
if ($CollectionScript) {
  if (-not (Test-Path -LiteralPath $CollectionScript -PathType Leaf)) {
    throw "Collection adapter does not exist: $CollectionScript"
  }
  & $CollectionScript @CollectionArguments
  if (-not $?) { throw "Collection adapter failed" }
}
foreach ($Condition in $Conditions) {
  $ConditionRoot = Join-Path $RawTimestampRoot $Condition
  if (-not (Test-Path -LiteralPath $ConditionRoot -PathType Container)) {
    throw "Missing collected condition directory: $ConditionRoot"
  }
}
Invoke-Uv -Arguments @(
  "run", "--extra", "training", "python", "$StudyDir/validate_experiment_setup.py",
  "--config", $Config, "--manifest", $Manifest, "--raw_root", $RawTimestampRoot
)

Write-Host "=== [2/8] Synchronize robot, gripper, action, and video clocks ===" -ForegroundColor Cyan
foreach ($Condition in $Conditions) {
  Invoke-Uv -Arguments @(
    "run", "--extra", "training", "python", "$StudyDir/time_sync.py",
    "--in_dir", (Join-Path $RawTimestampRoot $Condition),
    "--out_dir", (Join-Path $SyncedRoot $Condition),
    "--out_fps", "$TargetFps", "--robot_bridge_config", $RobotBridgeConfig,
    "--protocol_config", $Config
  )
}

Write-Host "=== [3/8] Convert both conditions with per-episode task prompts ===" -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$SourceMaps = @{}
foreach ($Condition in $Conditions) {
  $DatasetId = Get-DatasetId -Protocol $Protocol -Condition $Condition
  $MapPath = "$OutputRoot/source_episode_map_${Condition}.json"
  $SourceMaps[$Condition] = $MapPath
  Invoke-Uv -Arguments @(
    "run", "--extra", "training", "python", "$StudyDir/convert_raw_to_lerobot.py",
    "--raw_dir", (Join-Path $SyncedRoot $Condition),
    "--repo_id", $DatasetId, "--fps", "$TargetFps",
    "--manifest", $Manifest, "--protocol_config", $Config, "--condition", $Condition,
    "--source_map_out", $MapPath, "--dh_config", $DhConfig, "--resize", $Resize
  )
}

Write-Host "=== [4/8] Freeze equal successful training demonstrations ===" -ForegroundColor Cyan
Invoke-Uv -Arguments @(
    "run", "--extra", "training", "python", "$StudyDir/select_training_episodes.py",
    "--manifest", $Manifest, "--out", $SelectionManifest, "--summary", $SelectionSummary,
    "--selection_seed", "$SelectionSeed",
    "--minimum_per_cell", "$($Protocol.training.minimum_eligible_per_condition_task)"
  )
$SelectionRows = @(Import-Csv -LiteralPath $SelectionManifest)

Write-Host "=== [5/8] Train enabled policy families with paired random seeds ===" -ForegroundColor Cyan
$CheckpointFiles = @{}
foreach ($FamilyProperty in $Protocol.training.policy_families.PSObject.Properties) {
  $Family = [string]$FamilyProperty.Name
  $Run = $FamilyProperty.Value
  if ($Run.enabled -ne $true) { continue }
  foreach ($Condition in $Conditions) {
    $DatasetId = Get-DatasetId -Protocol $Protocol -Condition $Condition
    $SelectedRows = @(
      $SelectionRows | Where-Object {
        $_.condition -eq $Condition -and $_.selected_for_training -eq "True"
      }
    )
    if ($SelectedRows.Count -eq 0) {
      throw "No selected training episodes for $Family/$Condition"
    }
    $TrainEpisodes = Get-MappedEpisodes -Rows $SelectedRows -MapPath $SourceMaps[$Condition]
    $EpisodeJson = ConvertTo-Json -InputObject ([object[]]$TrainEpisodes) -Compress
    $CheckpointMap = [ordered]@{}
    foreach ($Seed in $Seeds) {
      $TrainOutput = "$OutputRoot/train/$Family/$Condition/seed_$Seed"
      $TrainArguments = @("run")
      foreach ($Extra in @($Run.dependency_extras)) {
        $TrainArguments += @("--extra", [string]$Extra)
      }
      $TrainArguments += @(
        "lerobot-train",
        "--dataset.repo_id=$DatasetId", "--dataset.episodes=$EpisodeJson",
        "--dataset.video_backend=$($Run.video_backend)"
      )
      if ($Run.PSObject.Properties["policy_path"]) {
        $TrainArguments += "--policy.path=$($Run.policy_path)"
      } elseif ($Run.PSObject.Properties["policy_type"]) {
        $TrainArguments += "--policy.type=$($Run.policy_type)"
      } else {
        throw "Policy family $Family needs policy_path or policy_type"
      }
      if ($Run.PSObject.Properties["chunk_size"]) {
        $TrainArguments += "--policy.chunk_size=$($Run.chunk_size)"
      }
      if ($Run.PSObject.Properties["n_action_steps"]) {
        $TrainArguments += "--policy.n_action_steps=$($Run.n_action_steps)"
      }
      $TrainArguments += @(
        "--policy.device=$($Run.device)",
        "--policy.push_to_hub=$($Run.push_to_hub.ToString().ToLowerInvariant())",
        "--seed=$Seed", "--output_dir=$TrainOutput", "--job_name=${Family}_${Condition}_seed_$Seed",
        "--batch_size=$($Run.batch_size)", "--steps=$($Run.steps)",
        "--eval_freq=$($Run.eval_freq)", "--num_workers=$($Run.num_workers)",
        "--wandb.enable=$($Run.wandb_enabled.ToString().ToLowerInvariant())"
      )
      if ($Run.extra_cli_args) { $TrainArguments += @($Run.extra_cli_args) }
      Invoke-Uv -Arguments $TrainArguments
      $CheckpointMap["$Seed"] = "$TrainOutput/checkpoints/last/pretrained_model"
    }
    $CheckpointPath = "$OutputRoot/checkpoints_${Family}_${Condition}.json"
    $CheckpointMap | ConvertTo-Json | Set-Content -LiteralPath $CheckpointPath -Encoding UTF8
    $CheckpointFiles["${Family}|${Condition}"] = $CheckpointPath
  }
}

Write-Host "=== [6/8] Extract and compare all valid human demonstrations ===" -ForegroundColor Cyan
foreach ($Condition in $Conditions) {
  $LabelsPath = "$OutputRoot/labels_${Condition}.csv"
  $SelectionRows | Where-Object { $_.condition -eq $Condition } |
    Export-Csv -LiteralPath $LabelsPath -NoTypeInformation
  $DatasetId = Get-DatasetId -Protocol $Protocol -Condition $Condition
  Invoke-Uv -Arguments @(
    "run", "--extra", "training", "python", "$StudyDir/feature_extraction.py",
    "--repo_id", $DatasetId, "--condition", $Condition,
    "--labels_csv", $LabelsPath, "--out", "$OutputRoot/features_${Condition}.csv",
    "--video_backend", "pyav"
  )
}
Invoke-Uv -Arguments @(
  "run", "--extra", "training", "python", "$StudyDir/features_compare.py",
  "--features_a", "$OutputRoot/features_A_mobile_colocated.csv",
  "--features_b", "$OutputRoot/features_B_desktop_separated.csv",
  "--out", "$OutputRoot/feature_comparison.csv",
  "--excel", "$OutputRoot/feature_comparison.xlsx"
)

Write-Host "=== [7/8] Freeze the 240-trial primary rollout schedule ===" -ForegroundColor Cyan
$RolloutManifest = "$OutputRoot/rollout_manifest_${PrimaryFamily}.csv"
Invoke-Uv -Arguments @(
  "run", "--extra", "training", "python", "$StudyDir/make_rollout_manifest.py",
  "--config", $Config, "--out", $RolloutManifest, "--policy_family", $PrimaryFamily
)

Write-Host "=== [8/8] Run the shared real-robot rollout suite ===" -ForegroundColor Cyan
if (-not $RolloutScript) {
  throw (
    "Offline stages completed, but formal evidence is incomplete. Supply -RolloutScript " +
    "with the hardware-specific short-horizon adapter; offline metrics cannot replace rollout."
  )
}
if (-not (Test-Path -LiteralPath $RolloutScript -PathType Leaf)) {
  throw "Rollout adapter does not exist: $RolloutScript"
}
$CheckpointAKey = "${PrimaryFamily}|A_mobile_colocated"
$CheckpointBKey = "${PrimaryFamily}|B_desktop_separated"
if (-not $CheckpointFiles.ContainsKey($CheckpointAKey) -or -not $CheckpointFiles.ContainsKey($CheckpointBKey)) {
  throw "Primary checkpoint maps are missing for $PrimaryFamily"
}
& $RolloutScript --config $Config --rollout_manifest $RolloutManifest `
  --checkpoints_a_json $CheckpointFiles[$CheckpointAKey] `
  --checkpoints_b_json $CheckpointFiles[$CheckpointBKey] `
  --output_dir "$OutputRoot/rollout" @RolloutArguments
if (-not $?) { throw "Rollout adapter failed" }
$RolloutResults = "$OutputRoot/rollout/rollout_results.csv"
if (-not (Test-Path -LiteralPath $RolloutResults -PathType Leaf)) {
  throw "Rollout adapter must write the completed frozen table to $RolloutResults"
}
Invoke-Uv -Arguments @(
  "run", "--extra", "training", "python", "$StudyDir/analyze_rollouts.py",
  "--results", $RolloutResults, "--out_dir", "$OutputRoot/rollout/analysis",
  "--expected_seeds", (($Protocol.training.random_seeds | ForEach-Object { [string]$_ }) -join ",")
)

Write-Host "=== Formal two-task wrist-view presentation pipeline finished ===" -ForegroundColor Green
