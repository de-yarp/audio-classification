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
uv run audio-clf train <config_path> <csv_path> [--save_model]
```

Example:

```bash
uv run audio-clf train config/cnn_mfcc_1.yaml runs/cnn_mfcc_train_tracker.csv --save_model
```

This runs the training loop, validates each epoch, saves the config to `runs/configs/`, and optionally saves a model checkpoint to `runs/checkpoints/`. Training artifacts (loss curve, accuracy curve) are saved to `runs/artifacts/train/<run_id>/`. A row is appended to the tracker CSV.

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
│   ├── lstm_mfcc.py                  # MFCC_LSTM (placeholder)
│   ├── lstm_mel.py                   # MEL_LSTM (placeholder)
│   ├── train.py                      # training loop, validation, model/optimizer setup
│   └── eval.py                       # evaluation loop (inference, preds/labels collection)
│
├── runs/                             # all training/eval outputs (not committed)
│   ├── configs/                      # saved validated YAML configs per run
│   ├── checkpoints/                  # model state_dict .pt files
│   ├── artifacts/
│   │   ├── train/<run_id>/           # loss_curve.png, accuracy_curve.png
│   │   └── eval/<run_id>/            # confusion_matrix.{npy,png}, classification_report.json
│   ├── your_model_train_tracker.csv  # one per person, training runs
│   └── your_model_eval_tracker.csv   # one per person, eval runs
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
  fc_layers: [512, 128]       # FC layer sizes (final → num_classes added automatically)
  num_classes: 50

run:
  seed: 42
  folds_train: [1, 2, 3]
  folds_val: [4]
  batch_size: 32
  num_epochs: 2
  optimizer: "SGD"            # SGD | ADAM | ADAMW
  lr: 0.001
  momentum: null              # only used with SGD
```

The CNN architecture is fully dynamic — conv/pool layers are built from the config list using `nn.ModuleList`, and the first FC layer's input size is auto-computed from the spatial dimensions after all conv/pool operations. Batch normalization is optional per conv layer.

**Note:** The config format above is currently implemented for CNN only. LSTM config and model are not yet built. The `run:` section will be shared between both architectures; the `model:` section will differ (LSTM will have its own layer parameters instead of `conv_layers` and `fc_layers`).

**Note:** `mfcc_deltas` is only relevant for MFCC experiments (ignored for mel). It must be manually aligned with how features were preprocessed — if `features.yaml` had `include_deltas: true`, set `mfcc_deltas: true` in the experiment config. There is no automatic link between feature extraction and training.

## Tracker CSVs

Each team member maintains their own train and eval tracker CSVs to avoid merge conflicts. These are append-only logs of every run. The `your_model_train_runs_tracker.csv` and `your_model_eval_runs_tracker.csv` files in `runs/` are empty templates for reference — rename them to match your experiment (e.g. `cnn_mfcc_train_tracker.csv`) before use. Keep your CSVs in `runs/` — it's the natural place alongside configs, checkpoints, and artifacts.

Train CSV columns: `ts, run_id, avg_loss_last_train_epoch, avg_loss_val, accuracy_val_pct, cfg_path, model_path, loss_curve_path_png, accuracy_curve_path_png`

Eval CSV columns: `ts, run_id, avg_loss, accuracy_pct, cfg_path, model_path, eval_folds, report_path_json, confusion_matrix_path_npy, confusion_matrix_path_png`

## Logging

All pipeline events are written as JSON lines to `logs/log.jsonl`. Each log entry includes a timestamp, run ID, component name, event type, and optional payload. Logs can be filtered by run ID using `parse_run_logs(run_id)` from `infra.log_utils`.

## Team Docs

- `docs/team/TEAM_SETUP.md` — environment setup, cloning, dependencies
- `docs/team/GIT_GUIDE.md` — branching strategy, commit conventions, merge rules
