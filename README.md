# Audio Classification

University research project comparing CNN and LSTM architectures on environmental sound classification using the [ESC-50](https://github.com/karolpiczak/ESC-50) dataset. Each architecture is trained on two audio representations — log-mel spectrogram and MFCC — giving four experiments in total. Results and methodology are documented in `docs/report/`.

The configs currently in `config/` are the final configurations used to produce the results reported in the paper. Each config represents the best-performing hyperparameter set found during experimentation for that model.

## Quick Start

Requires Python 3.12.* and [uv](https://docs.astral.sh/uv/) (`pip install uv`).

```bash
git clone git@github.com:de-yarp/audio-classification.git
cd audio-classification
```

Then install PyTorch in the variant that matches your machine:

```bash
uv sync --extra cu128    # NVIDIA GPU with CUDA 12.8+ driver (560+)
uv sync --extra cpu      # no NVIDIA GPU, or Mac
```

Pick exactly one — the two extras are mutually exclusive. If you pick the wrong one, remove `.venv/` and `uv.lock`, then re-sync with the right extra. If `--extra cu128` succeeds but `torch.cuda.is_available()` returns `False`, update your NVIDIA driver from [nvidia.com/drivers](https://www.nvidia.com/drivers) and try again.

All commands go through `uv run`.

### Data Setup

1. Download the [ESC-50](https://github.com/karolpiczak/ESC-50) dataset. Copy all `.wav` files from the repository's `audio/` folder into `data/raw/esc50/audio/` and the `esc50.csv` file from `meta/` into `data/raw/esc50/meta/`.
2. Run feature extraction:

```bash
uv run python -c "from infra.preprocessing import get_features_esc50; get_features_esc50()"
```

This reads `config/features.yaml` and writes `.npy` files to `data/processed/esc50/mel/` and `data/processed/esc50/mfcc/`. Only needs to run once — it will refuse to overwrite existing files.

### Tracker CSVs

**Before running any training or evaluation, you must create your tracker CSV files manually.** The pipeline will not create them — it appends to existing files only and will error if the file is missing.

```bash
touch runs/train_tracker.csv
touch runs/eval_tracker.csv
touch runs/cross_val_tracker.csv
```

See the [Tracker CSVs](#tracker-csvs) section for the expected column schema.

### Training

```bash
uv run audio-clf train <config_path> <csv_path> [--save-model] [--cross-val <cv_csv_path>]
```

**Quick run** (single train/val split):

```bash
uv run audio-clf train config/cnn_mfcc_1.yaml runs/train_tracker.csv --save-model
```

**Cross-validation** (rotates validation fold across all train+val folds):

```bash
uv run audio-clf train config/cnn_mfcc_1.yaml runs/train_tracker.csv --save-model --cross-val runs/cross_val_tracker.csv
```

With `--cross-val`, the pipeline generates all combinations of validation folds from the folds listed in `folds_train` + `folds_val` in the config. For example, with `folds_train: [1, 2, 3, 4]` and `folds_val: [5]` (val size = 1), this produces 5 runs — each using a different single fold for validation and the remaining 4 for training. The seed is reset before each fold run for reproducibility.

Quick runs save artifacts to `runs/artifacts/train/quick/<run_id>/`. Cross-validation runs save per-fold artifacts to `runs/artifacts/train/cv/<cv_run_id>/<child_run_id>/`, with aggregated CV curves (mean ± std) saved to `runs/artifacts/train/cv/<cv_run_id>/`. Each fold run is also logged as a separate row in the train tracker CSV.

Per-fold artifacts include: `loss_curve.png`, `accuracy_curve.png`, `confusion_matrix_full.{npy,png}` (normalized 50×50), `confusion_matrix_category.{npy,png}` (5×5 semantic category aggregate), and `classification_report.json`. No aggregate confusion matrix or classification report is generated at the CV level — only the per-fold artifacts.

Training automatically detects the best available device (CUDA → MPS → CPU).

### Evaluation

```bash
uv run audio-clf eval <config_path> <csv_path> <model_path> --eval-folds <fold_numbers>
```

Example:

```bash
uv run audio-clf eval runs/configs/cnn_mfcc_20260316_193622.yaml runs/eval_tracker.csv runs/checkpoints/cnn_mfcc_20260316_193622.pt --eval-folds 5
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
│   ├── cnn_mfcc_1.yaml              # CNN + MFCC experiment config
│   ├── cnn_mel_4.yaml               # CNN + log-mel experiment config
│   ├── lstm_mfcc_1.yaml             # LSTM + MFCC experiment config
│   ├── lstm_mel_1.yaml              # LSTM + log-mel experiment config
│   └── features.yaml                # feature extraction params (n_fft, hop_length, etc.)
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
│   ├── cnn_mel.py                    # MEL_CNN: dynamic conv/pool/FC via ModuleList
│   ├── lstm_mfcc.py                  # MFCC_LSTM: LSTM + dynamic FC head
│   ├── lstm_mel.py                   # MEL_LSTM: LSTM + LayerNorm + BN FC head (see note below)
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
│   │   │   ├── quick/<run_id>/       # loss_curve.png, accuracy_curve.png,
│   │   │   │                         # confusion_matrix_full.{npy,png}, confusion_matrix_category.{npy,png},
│   │   │   │                         # classification_report.json
│   │   │   └── cv/<cv_run_id>/       # cv_loss_curve.png, cv_accuracy_curve.png
│   │   │       └── <child_run_id>/   # per-fold: loss_curve.png, accuracy_curve.png,
│   │   │                             # confusion_matrix_full.{npy,png}, confusion_matrix_category.{npy,png},
│   │   │                             # classification_report.json
│   │   └── eval/<run_id>/            # confusion_matrix.{npy,png}, classification_report.json
│   ├── train_tracker.csv             # training runs (quick + CV folds)
│   ├── eval_tracker.csv              # eval runs
│   └── cross_val_tracker.csv         # CV summary stats
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_dataset_class.py
│   └── test_lstm_mfcc.py
│
├── logs/
│   └── log.jsonl                     # structured JSON logs (all runs)
│
├── docs/
│   └── report/                       # final university report
│
├── .gitignore
├── pyproject.toml
└── uv.lock
```

## Features Config

Feature extraction is controlled by `config/features.yaml`:

```yaml
n_fft: 2048                    # FFT window size
hop_length: 512                # hop length between frames
n_mels: 128                    # number of mel filterbanks (mel only)
n_mfcc: 40                     # number of MFCC coefficients
normalize_mfcc: false          # per-coefficient normalization (divides by std per coeff axis)
include_deltas: true           # compute delta and delta-delta features
stack_deltas_as_channels: true # true → (3, 40, T) channel-stacked; false → (120, T) height-concat
```

**Note:** `stack_deltas_as_channels` only applies when `include_deltas: true`. With channel-stacking, static MFCCs, deltas, and delta-deltas are stored as separate channels — giving shape `(3, n_mfcc, T)` instead of `(3*n_mfcc, T)`. This must be manually aligned with `stack_deltas_as_channels` in the experiment config. If you change this setting, delete `data/processed/esc50/` first — feature extraction raises `FileExistsError` if the output directory already contains processed files.

## Config Format

Experiment configs live in `config/` and have two sections: `model:` and `run:`. The `run:` section is identical across all four model types. The `model:` section differs per architecture.

### CNN (MFCC and log-mel)

```yaml
model:
  model_type: "cnn"                # cnn | lstm
  repr_type: "mfcc"                # mfcc | mel
  mfcc_deltas: true                # true → include delta features; false → static only
  stack_deltas_as_channels: true   # true → input shape (3, 40, T); false → (120, T)
                                   # must match stack_deltas_as_channels in features.yaml
                                   # not used when repr_type is mel
  conv_layers:                     # list of conv/pool layers built dynamically via ModuleList
    - type: "conv"
      kernel_count: 32
      kernel_size: [3, 5]          # int for symmetric, [height, width] for asymmetric
      stride: 1                    # int or [height, width]
      padding: 0                   # int for symmetric, [height, width] for asymmetric
      batch_norm: true
    - type: "pool"
      kernel_size: [1, 2]
      stride: [1, 2]
      padding: 0
  pool_type: "max"                 # max | avg (applies to all pool layers)
  fc_layers: [512, 128]            # FC layer sizes (final → num_classes added automatically)
  dropout: 0.5                     # dropout rate applied before first FC and between FC layers
  num_classes: 50
  global_avg_pool: null            # [h, w] to apply AdaptiveAvgPool2d before FC, null to disable
                                   # when set, FC input size = channels only (spatial dims collapsed)
                                   # when null, FC input size = channels × height × width
```

The CNN architecture is fully dynamic — conv/pool layers are built from the config list, and the first FC layer's input size is auto-computed from the spatial dimensions after all conv/pool operations. Batch normalization is optional per conv layer. Both `kernel_size`, `stride`, and `padding` accept either a single integer (symmetric) or a `[height, width]` list (asymmetric).

### LSTM (MFCC and log-mel)

```yaml
model:
  model_type: "lstm"
  repr_type: "mfcc"                # mfcc | mel
  mfcc_deltas: true                # true → 120-dimensional input; false → 40-dimensional
  hidden_size: 256                 # LSTM hidden state size per direction
  num_layers: 2                    # number of stacked LSTM layers
  dropout: 0.3                     # dropout between stacked LSTM layers (0.0 = disabled)
                                   # PyTorch built-in LSTM dropout, not applied to FC layers
  bidirectional: true              # true → bidirectional LSTM; FC input size doubles
  pooling: "mean"                  # how to reduce LSTM output over time dimension:
                                   #   last     → final hidden state
                                   #   mean     → mean over all timesteps
                                   #   max      → max over all timesteps
                                   #   mean_max → concat of mean and max (doubles FC input size)
  fc_layers: [256, 128]            # FC layer sizes after LSTM output
  num_classes: 50
```

The LSTM takes the LSTM output reduced by the chosen pooling strategy and passes it through the FC classifier head. `dropout` applies between stacked LSTM layers (PyTorch's built-in LSTM dropout), not between FC layers. Channel-stacked MFCC inputs `(3, n_mfcc, T)` are automatically reshaped to `(n_mfcc*3, T)` in the forward pass — the LSTM sees the same 120-dimensional feature vector per timestep regardless of stacking mode.

**Known limitation (MEL_LSTM):** `MEL_LSTM` in `models/lstm_mel.py` has two hardcoded assumptions not driven by config: the input size is hardcoded to 128 (n_mels from `features.yaml` is not passed through the config dataclass), and the forward pass applies a double transpose to handle input orientation. Both assumptions hold for the current `features.yaml` configuration (`n_mels: 128`) and are consistent across all runs. Additionally, `MEL_LSTM` applies `LayerNorm` before the FC head and `BatchNorm1d` after each FC layer — these are not configurable and are fixed in the model class.

### Run section (all models)

```yaml
run:
  seed: 42
  folds_train: [1, 2, 3, 4]
  folds_val: [5]
  batch_size: 32
  num_epochs: 100
  optimizer: "SGD"                 # SGD | ADAM | ADAMW
  lr: 0.001
  momentum: 0.9                    # only used with SGD; set null for Adam/AdamW
  weight_decay: 0.01               # L2 regularization, applies to all optimizers

  warmup_lr: true                  # enable LR warmup phase
  warmup_epochs: 5                 # number of warmup epochs (LR ramps from warmup_lr_val to lr)
  warmup_lr_val: 0.00001           # starting LR for warmup; ignored if warmup_lr is false

  scheduler: "plateau"             # plateau | cosine | step | null (no scheduling)
  factor: 0.1                      # LR multiplier on trigger (plateau, step); null for cosine
  patience: 5                      # epochs without improvement before LR reduction (plateau only)
  min_lr: 0.000001                 # LR floor (plateau, cosine)
  step_size: null                  # reduce LR every N epochs (step only)

  augment: true                    # enable SpecAugment data augmentation (training only)
  freq_masks: 0                    # number of frequency masks
                                   # set 0 for MFCC — cepstral coefficient axis is not a frequency axis
  freq_mask_width: 10              # max frequency mask width in bins
  time_masks: 2                    # number of time masks
  time_mask_width: 25              # max time mask width in frames
```

**LR warmup** is optional. When `warmup_lr: true`, the training loop linearly ramps the learning rate from `warmup_lr_val` to `lr` over `warmup_epochs` epochs before handing off to the scheduler. The scheduler does not step during the warmup phase. Warmup applies per fold in cross-validation. Set `warmup_lr: false` to disable — `warmup_epochs` and `warmup_lr_val` are ignored in that case.

**Learning rate scheduling** is optional. `plateau` (ReduceLROnPlateau) watches validation loss and reduces LR when improvement stalls. `cosine` (CosineAnnealingLR) smoothly decays LR over all epochs. `step` (StepLR) multiplies LR by `factor` every `step_size` epochs. Each type only uses its relevant parameters — unused fields should be set to `null`.

**Data augmentation** applies frequency and time masking to training samples only — validation and evaluation always see unmodified data. When `augment: true`, each training sample gets random masks applied on-the-fly in `__getitem__`. For channel-stacked inputs `(3, n_mfcc, T)`, masking correctly targets axis 1 (coefficients) and axis 2 (time), leaving the channel axis untouched. The pipeline validates at dataset creation that the worst-case masking does not exceed 50% of the respective dimension.

**Note:** `mfcc_deltas` and `stack_deltas_as_channels` in the model config must be manually aligned with `include_deltas` and `stack_deltas_as_channels` in `features.yaml`. There is no automatic link between feature extraction and training.

## Tracker CSVs

Tracker CSVs are append-only logs of every run. **They must be created manually before running any command** — the pipeline appends to existing files and will error if they are missing.

```bash
touch runs/train_tracker.csv
touch runs/eval_tracker.csv
touch runs/cross_val_tracker.csv
```

Train CSV columns: `ts, run_id, avg_loss_last_train_epoch, avg_loss_val, accuracy_val_pct, cfg_path, model_path, loss_curve_path_png, accuracy_curve_path_png, cv_run_id`

The `cv_run_id` column is empty for quick runs and contains the parent CV run ID for cross-validation fold runs.

Eval CSV columns: `ts, run_id, avg_loss, accuracy_pct, cfg_path, model_path, eval_folds, report_path_json, confusion_matrix_path_npy, confusion_matrix_path_png`

Cross-validation CSV columns: `ts, cv_run_id, child_run_ids, mean_accuracy, std_accuracy, mean_loss, std_loss, cfg_path`

`child_run_ids` is stored as a semicolon-joined string.

## Logging

All pipeline events are written as JSON lines to `logs/log.jsonl`. Each log entry includes a timestamp, run ID, component name, event type, and optional payload. Logs can be filtered by run ID using `parse_run_logs(run_id)` from `infra.log_utils`.

The pipeline also prints human-readable progress to stdout during runs. For training, each epoch prints train loss, validation loss, validation accuracy, and current learning rate. For cross-validation, each fold is announced with its train/val fold split before training begins. Eval prints a summary line with accuracy, loss, and macro precision/recall/F1 on completion. These prints are separate from the structured log and require no configuration.
