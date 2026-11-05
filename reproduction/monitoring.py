"""Use Trainroom's logger when available, with an identical local JSONL fallback."""
try:
    from terminal_sdk import TrainingLogger
except ImportError:
    import json
    import math
    import os
    from pathlib import Path
    import time

    class TrainingLogger:
        def __init__(self, run_dir=None, config=None, total_steps=None):
            self.path = Path(run_dir or os.environ.get('TRAIN_TERMINAL_RUN_DIR', 'runs/default')).resolve()
            self.enabled = int(os.environ.get('RANK', '0')) == 0
            if self.enabled:
                self.path.mkdir(parents=True, exist_ok=True)
                (self.path / 'metadata.json').write_text(json.dumps(
                    {'config': config or {}, 'total_steps': total_steps}, indent=2, default=str))

        def log(self, step, **metrics):
            if not self.enabled:
                return
            record = {'step': int(step), 'time': time.time()}
            for name, value in metrics.items():
                if hasattr(value, 'item'):
                    value = value.item()
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    record[name] = value if math.isfinite(value) else str(value)
            with (self.path / 'metrics.jsonl').open('a') as file:
                file.write(json.dumps(record, allow_nan=False) + '\n')

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.close()
