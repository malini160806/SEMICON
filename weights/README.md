
# Model Weights

This directory holds trained checkpoints produced by `train.py`:

```text
weights/
├── best_model.pth   # auto-selected checkpoint per config.training.select_best_metric
├── best_psnr.pth    # checkpoint with the highest validation PSNR seen so far
├── best_ssim.pth    # checkpoint with the highest validation SSIM seen so far
├── last_model.pth   # most recent epoch (used to auto-resume training)
└── epoch_N.pth       # periodic snapshots (config.training.checkpoint_every_n_epochs)
```

Each `.pth` file is a dict containing `model_state_dict`, `ema_state_dict`
(if EMA is enabled), optimizer/scheduler/scaler state, the epoch number, the
best-metrics-so-far, and the exact resolved config used to train it — so
`evaluate.py` / `inference.py` can rebuild the correct architecture
automatically without any manual flags.

## Weights are not committed to this repository

`*.pth` files are excluded via `.gitignore` because trained restoration model
checkpoints are typically tens to hundreds of MB, well past what a plain git
repository should carry. Once you have trained a model, distribute the
checkpoint through one of:

- **Git LFS** — `git lfs track "weights/*.pth"` then commit normally; keeps
  weights versioned alongside the code, best if your grader also has LFS.
- **Hugging Face Hub** — `huggingface-cli upload <repo> weights/best_model.pth`;
  best for public sharing with automatic versioning and a download URL.
- **Google Drive / cloud bucket** — simplest for a hackathon submission; link
  it from this README and from the main project README's "Model Weights"
  section once the file exists.

After placing/downloading a checkpoint here, evaluate with:

```bash
python evaluate.py --input_dir ./datasets/Test_NoisyLR --output_dir ./outputs/test_restored --weights ./weights/best_model.pth
```
