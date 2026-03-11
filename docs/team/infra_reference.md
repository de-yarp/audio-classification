# Infra Reference

## Logging

Logger is configured once at the entry point of your script:

```python
from infra.log_utils import setup_logger

setup_logger()
```

This sets up two handlers: a file handler that writes JSON lines to the log file (INFO and above), and a stderr handler that prints warnings and errors to the console.

In any module that needs logging:

```python
import logging

logger = logging.getLogger(__name__)
```

### Log calls

Basic message:

```python
logger.info("training started")
```

With run ID and payload:

```python
logger.info(
    "epoch complete",
    extra={"run_id": run_id, "payload": {"epoch": 1, "train_loss": 0.42}},
)
```

The `run_id` should be generated once at the start of each training run. The `payload` dict can contain any metrics or metadata relevant to the event.

### Retrieving logs

```python
from infra.log_utils import parse_run_logs

entries = parse_run_logs("your-run-id")  # returns list[dict]
```

---

## AudioDataset

`AudioDataset` is a PyTorch `Dataset` subclass that serves preprocessed audio features and labels from memory.

### Initialization

```python
from infra.data_models import AudioDataset, ReprType

train_ds = AudioDataset(
    repr_type=ReprType.MEL,       # or ReprType.MFCC
    folds=[1, 2, 3],              # which ESC-50 folds to include
)

val_ds = AudioDataset(
    repr_type=ReprType.MEL,
    folds=[4],
)

test_ds = AudioDataset(
    repr_type=ReprType.MEL,
    folds=[5],
)
```

### Usage with DataLoader

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

for features, labels in train_loader:
    # features: torch.Tensor
    # labels: int
    ...
```

### Notes

- All `.npy` features are loaded into RAM on initialization. ESC-50 scale fits comfortably.
- Each fold contains exactly 400 samples. Folds are predefined in the dataset to prevent data leakage (clips from the same source recording stay in the same fold).
- `ReprType.MEL` loads mel-spectrograms, `ReprType.MFCC` loads MFCCs (with deltas if configured).
- Preprocessing must be run before creating a dataset (`infra.preprocessing.get_features_esc50`).

---

## Tests

Default test run (excludes slow tests):

```bash
uv run pytest -m "not slow"
```

Run everything including slow tests:

```bash
uv run pytest
```

**Convention:** any test that loads preprocessed data or takes more than a couple of seconds must be decorated with `@pytest.mark.slow`. Do not run `uv run pytest` without the marker filter during regular development — slow tests process the full dataset and take time.
