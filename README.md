# Audio Classification

University project comparing CNN and LSTM architectures on environmental sound classification using the [ESC-50](https://github.com/karolpiczak/ESC-50) dataset. Each architecture is trained on two audio representations — log-mel spectrogram and MFCC — giving four experiments total. Each team member owns one experiment end-to-end.

## Quick Start

Requires Python 3.12.* and [uv](https://docs.astral.sh/uv/) (`pip install uv`).

```bash
git clone git@github.com:de-yarp/audio-classification.git
cd audio-classification
uv sync
```

All commands go through `uv run`. See `docs/team/TEAM_SETUP.md` for full onboarding.

### Data Setup

1. Download the [ESC-50](https://github.com/karolpiczak/ESC-50) dataset. Copy all `.wav` files from the repository's `audio/` folder into `data/raw/esc50/audio/` and the `esc50.csv` file from `meta/` into `data/raw/esc50/meta/`.
2. Run feature extraction:

```bash
uv run python -c "from infra.preprocessing import get_features_esc50; get_features_esc50()"
```

This reads `config/features.yaml` and writes `.npy` files to `data/processed/esc50/mel/` and `data/processed/esc50/mfcc/`. Only needs to run once — it will refuse to overwrite existing files.

### Training

```bash
uv run audio-clf train <config_path> <csv_path> [--save_model] [--cross-val <cv_csv_path>]
```

**Quick run** (single train/val split):

```bash
uv run audio-clf train config/cnn_mfcc_1.yaml runs/cnn_mfcc_train_tracker.csv --save_model
```

**Cross-validation** (rotates validation fold across all train+val folds):

```bash
uv run audio-clf train config/cnn_mfcc_1.yaml runs/cnn_mfcc_train_tracker.csv --save_model --cross-val runs/cnn_mfcc_cross_val_tracker.csv
```

With `--cross-val`, the pipeline generates all combinations of validation folds from the folds listed in `folds_train` + `folds_val` in the config. For example, with `folds_train: [1, 2, 3]` and `folds_val: [4]` (val size = 1), this produces 4 runs — each using a different single fold for validation and the remaining 3 for training. The seed is reset before each fold run for reproducibility.

Quick runs save artifacts to `runs/artifacts/train/quick/<run_id>/`. Cross-validation runs save per-fold artifacts to `runs/artifacts/train/cv/<cv_run_id>/<child_run_id>/`, with aggregated CV curves (mean ± std) saved to `runs/artifacts/train/cv/<cv_run_id>/`. Each fold run is also logged as a separate row in the train tracker CSV.

Training automatically detects the best available device (CUDA → MPS → CPU).

### Evaluation

```bash
uv run audio-clf eval <config_path> <csv_path> <model_path> --eval-folds <fold_numbers>
```

Example:

```bash
uv run audio-clf eval runs/configs/cnn_mfcc_20260316_193622.yaml runs/cnn_mfcc_eval_tracker.csv runs/checkpoints/cnn_mfcc_20260316_193622.pt --eval-folds 5
```

The config and checkpoint must have matching stems (same training run). Eval folds must not overlap with the train or validation folds specified in the config. Evaluation artifacts (confusion matrix, classification report) are saved to `runs/artifacts/eval/<run_id>/`. A row is appended to the tracker CSV.

### Tests

```bash
uv run pytest -m "not slow"    # fast tests only
uv run pytest                  # all tests including slow ones
```

Tests marked `@pytest.mark.slow` load the full dataset. Don't run them routinely.

## Project Structure

```
.
├── config/
│   ├── cnn_mfcc_1.yaml              # experiment config (architecture + training params)
│   └── features.yaml                 # feature extraction params (n_fft, hop_length, etc.)
│
├── data/
│   ├── raw/esc50/
│   │   ├── audio/                    # .wav files (not committed)
│   │   └── meta/esc50.csv            # fold, target, category metadata
│   └── processed/esc50/              # .npy features (not committed)
│       ├── mel/
│       └── mfcc/
│
├── infra/
│   ├── artifacts.py                  # plot/report generation (confusion matrix, loss curves)
│   ├── cli.py                        # typer CLI entry point (train, eval commands)
│   ├── cli_utils.py                  # config validation, CLI arg validation, fold checks
│   ├── data_models.py                # dataclasses, enums, AudioDataset, constants
│   ├── io_utils.py                   # file I/O: YAML, CSV, checkpoints, orchestrators
│   ├── log_utils.py                  # JSON structured logging, emit closure
│   ├── pipeline.py                   # pipe_run: orchestrates train or eval end-to-end
│   └── preprocessing.py              # feature extraction (librosa → .npy)
│
├── models/
│   ├── cnn_mfcc.py                   # MFCC_CNN: dynamic conv/pool/FC via ModuleList
│   ├── cnn_mel.py                    # MEL_CNN (placeholder)
│   ├── lstm_mfcc.py                  # MFCC_LSTM: LSTM + dynamic FC head
│   ├── lstm_mel.py                   # MEL_LSTM (placeholder)
│   ├── cross_val.py                  # cross-validation loop, fold combination generation
│   ├── train.py                      # training loop, validation, model/optimizer setup
│   └── eval.py                       # evaluation loop (inference, preds/labels collection)
│
├── runs/                             # all training/eval outputs (not committed)
│   ├── configs/
│   │   ├── quick/<run_id>/           # saved config for quick runs
│   │   └── cv/<cv_run_id>/<child_run_id>/  # saved config per CV fold
│   ├── checkpoints/
│   │   ├── quick/<run_id>/           # model .pt for quick runs
│   │   └── cv/<cv_run_id>/<child_run_id>/  # model .pt per CV fold
│   ├── artifacts/
│   │   ├── train/
│   │   │   ├── quick/<run_id>/       # loss_curve.png, accuracy_curve.png
│   │   │   └── cv/<cv_run_id>/       # cv_loss_curve.png, cv_accuracy_curve.png
│   │   │       └── <child_run_id>/   # per-fold loss_curve.png, accuracy_curve.png
│   │   └── eval/<run_id>/            # confusion_matrix.{npy,png}, classification_report.json
│   ├── your_model_train_tracker.csv  # one per person, training runs (quick + CV folds)
│   ├── your_model_eval_tracker.csv   # one per person, eval runs
│   └── your_model_cross_val_tracker.csv  # one per person, CV summary stats
│
├── tests/
│   ├── test_preprocessing.py
│   └── test_dataset_class.py
│
├── logs/
│   └── log.jsonl                     # structured JSON logs (all runs)
│
├── docs/
│   ├── team/
│   │   ├── GIT_GUIDE.md              # branching, commits, merge workflow
│   │   └── TEAM_SETUP.md             # environment setup for new members
│   └── report/                       # final report (TBD)
│
├── .gitignore
├── pyproject.toml
└── uv.lock
```

## Config Format

Experiment configs live in `config/` and have two sections:

```yaml
model:
  model_type: "cnn"           # cnn | lstm
  repr_type: "mfcc"           # mfcc | mel
  mfcc_deltas: true           # true → 120 channels (MFCC + delta + delta2), false → 40 channels
  conv_layers:                # list of conv/pool layers (CNN only)
    - type: "conv"
      kernel_count: 16
      kernel_size: 3
      stride: 1
      padding: 0
      batch_norm: true
    - type: "pool"
      kernel_size: 2
      stride: 2
      padding: 0
  pool_type: "max"             # max | avg (applies to all pool layers)
  fc_layers: [512, 128]       # FC layer sizes (final → num_classes added automatically)
  dropout: 0.0                # dropout rate between FC layers (0.0 = disabled)
  num_classes: 50

run:
  seed: 42
  folds_train: [1, 2, 3]
  folds_val: [4]
  batch_size: 32
  num_epochs: 50
  optimizer: "SGD"            # SGD | ADAM | ADAMW
  lr: 0.001
  momentum: null              # only used with SGD
  weight_decay: 0.0           # L2 regularization (e.g. 1e-4), applies to all optimizers

  scheduler: null              # plateau | cosine | step | null (no scheduling)
  factor: 0.1                  # LR multiplier on trigger (plateau, step)
  patience: 5                  # epochs without improvement before LR reduction (plateau only)
  min_lr: 0.000001             # LR floor (plateau, cosine, step)
  step_size: null              # reduce LR every N epochs (step only)

  augment: false               # enable SpecAugment data augmentation (training only)
  freq_masks: 2                # number of frequency masks
  freq_mask_width: 10          # max frequency mask width in bins
  time_masks: 2                # number of time masks
  time_mask_width: 25          # max time mask width in frames
```

The CNN architecture is fully dynamic — conv/pool layers are built from the config list using `nn.ModuleList`, and the first FC layer's input size is auto-computed from the spatial dimensions after all conv/pool operations. Batch normalization is optional per conv layer.

**Learning rate scheduling** is optional. When `scheduler` is set, the specified PyTorch scheduler adjusts the learning rate during training. `plateau` (ReduceLROnPlateau) watches validation loss and reduces LR when improvement stalls. `cosine` (CosineAnnealingLR) smoothly decays LR from the initial value to `min_lr` over all epochs. `step` (StepLR) multiplies LR by `factor` every `step_size` epochs. Each type only uses its relevant parameters — unused fields are ignored. The current learning rate is logged per epoch alongside validation metrics.

**Data augmentation** applies frequency and time masking to training samples only — validation and evaluation always see unmodified data. When `augment: true`, each training sample gets random frequency and time masks applied on-the-fly in `__getitem__`. Each mask has a random width (between 1 and the configured max) and a random position within the tensor. The pipeline validates at dataset creation that the worst-case masking (`freq_masks * freq_mask_width` and `time_masks * time_mask_width`) does not exceed 50% of the respective tensor dimension.

**Note:** The config format above is for CNN. See below for the LSTM config format. The `run:` section is shared between both architectures; the `model:` section differs.

**Note:** `mfcc_deltas` is only relevant for MFCC experiments (ignored for mel). It must be manually aligned with how features were preprocessed — if `features.yaml` had `include_deltas: true`, set `mfcc_deltas: true` in the experiment config. There is no automatic link between feature extraction and training.

LSTM config uses the same `run:` section but a different `model:` section:

```yaml
model:
  model_type: "lstm"          # cnn | lstm
  repr_type: "mfcc"           # mfcc | mel
  mfcc_deltas: true           # true → 120 input size, false → 40
  hidden_size: 128             # LSTM hidden state size
  num_layers: 2                # number of stacked LSTM layers
  dropout: 0.3                 # dropout between LSTM layers (0.0 = disabled)
  fc_layers: [128]             # FC layer sizes after LSTM output (final → num_classes added automatically)
  num_classes: 50

run:
  # same as CNN
```

The LSTM takes the last hidden state from the final layer and passes it through the FC classifier head. Note that `dropout` in the LSTM config applies between stacked LSTM layers (PyTorch's built-in LSTM dropout), not between FC layers as in the CNN config.

## Tracker CSVs

Each team member maintains their own train, eval, and cross-validation tracker CSVs to avoid merge conflicts. These are append-only logs of every run. The template files in `runs/` should be renamed to match your experiment (e.g. `cnn_mfcc_train_tracker.csv`) before use. Keep your CSVs in `runs/` — it's the natural place alongside configs, checkpoints, and artifacts.

Train CSV columns: `ts, run_id, avg_loss_last_train_epoch, avg_loss_val, accuracy_val_pct, cfg_path, model_path, loss_curve_path_png, accuracy_curve_path_png, cv_run_id`

The `cv_run_id` column is empty for quick runs and contains the parent CV run ID for cross-validation fold runs.

Eval CSV columns: `ts, run_id, avg_loss, accuracy_pct, cfg_path, model_path, eval_folds, report_path_json, confusion_matrix_path_npy, confusion_matrix_path_png`

Cross-validation CSV columns: `ts, cv_run_id, child_run_ids, mean_accuracy, std_accuracy, mean_loss, std_loss, cfg_path`

`child_run_ids` is stored as a semicolon-joined string.

## Logging

All pipeline events are written as JSON lines to `logs/log.jsonl`. Each log entry includes a timestamp, run ID, component name, event type, and optional payload. Logs can be filtered by run ID using `parse_run_logs(run_id)` from `infra.log_utils`.

## Team Docs

- `docs/team/TEAM_SETUP.md` — environment setup, cloning, dependencies
- `docs/team/GIT_GUIDE.md` — branching strategy, commit conventions, merge rules
