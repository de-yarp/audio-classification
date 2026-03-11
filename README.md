# Audio Classification

Comparison of CNN and LSTM on environmental sound classification using the ESC-50 dataset. Each architecture is trained on two audio representations (log-mel-spectrogram, MFCC) — four experiments total.

## Project Structure

```
.
├── config/
│   └── features.yaml            # feature extraction parameters (n_fft, hop_length, n_mels, n_mfcc, etc.)
│
├── data/
│   ├── raw/
│   │   └── esc50/
│   │       ├── audio/            # ESC-50 .wav files (not committed)
│   │       └── meta/
│   │           ├── esc50.csv     # metadata: filename, fold, target, category
│   │           └── esc50-human.xlsx
│   └── processed/                # extracted .npy features (not committed)
│
├── infra/
│   ├── __init__.py
│   ├── preprocessing.py          # feature extraction pipeline (get_features_esc50)
│   ├── data_models.py            # AudioDataset class, ReprType enum
│   ├── io_utils.py               # file I/O helpers
│   └── log_utils.py              # logger setup (JSON file + stderr), log parser
│
├── models/
│   └── __init__.py               # model definitions go here (CNN, LSTM)
│
├── tests/
│   ├── test_preprocessing.py     # tests for feature extraction pipeline
│   └── test_dataset_class.py     # tests for AudioDataset loading and shapes
│
├── logs/
│   └── log.jsonl                 # structured training/run logs
│
├── docs/
│   ├── team/
│   │   ├── GIT_GUIDE.md          # git workflow, commit conventions, rules
│   │   ├── TEAM_SETUP.md         # onboarding guide for team members
│   │   └── infra_reference.md    # usage docs for logging, dataset class, tests
│   └── report/                   # final report (TBD)
│
├── .gitignore
├── pyproject.toml                # project metadata and dependencies
├── uv.lock                       # locked dependency versions
└── README.md
```
