# v44 Recovery Controller Checkpoints

These are the minimal binary artifacts needed to reproduce the current v44
recovery-controller chain without depending on ignored `logs/` checkpoints.

## Files

```text
amp_base_model_81999.pt
stage3_world_model_699.pt
```

## Source Paths

```text
amp_base_model_81999.pt
  copied from:
  logs/rsl_rl/run_amp_y/2026-05-11_17-47-06_amp_axis_ft_v2_20260511_174702/model_81999.pt

stage3_world_model_699.pt
  copied from:
  logs/rsl_rl/run_amp_track_adapter/2026-05-20_16-20-56_track_adapter_81999_stage3_recovery_v44_recovery_controller_20260520_162052_wm_pretrain_recovery_controller/model_699.pt
```

## SHA256

```text
37800e9a18b13e8e4d48124f279834e05b9e74f413f1a8f869d7907ad9b339d0  amp_base_model_81999.pt
b2030d5ad94286699a0f66a21789082d63d7353cd389ec49d40518e3a62d35f4  stage3_world_model_699.pt
```

## Usage

Use `amp_base_model_81999.pt` as `BOOSTER_TRACK_ADAPTER_BASE_CHECKPOINT`.
Use `stage3_world_model_699.pt` as the pre-trained Stage3 world-model
checkpoint when reproducing v44 PPO.

The deployed v44 adapter PPO checkpoint remains:

```text
logs/rsl_rl/run_amp_track_adapter/2026-05-20_18-04-18_track_adapter_81999_stage3_recovery_v44_recovery_controller_20260520_162052_ppo_recovery_controller/model_2475.pt
```

See `docs/repro_v44_recovery_controller.md` for the full reproduction chain.
