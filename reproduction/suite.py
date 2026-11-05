"""Run the evaluation CLI, then build the result report even if interrupted."""
import sys
import os
from pathlib import Path
import time

from reproduction.evaluate import main as evaluate, atomic_json, sha256
from reproduction.report import generate


def main():
    output = 'runs/evaluation'
    checkpoint = 'checkpoints/boq_b2_best.pt'
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == '--output-dir' and i + 1 < len(sys.argv):
            output = sys.argv[i + 1]
        elif arg.startswith('--output-dir='):
            output = arg.split('=', 1)[1]
        elif arg == '--checkpoint' and i + 1 < len(sys.argv):
            checkpoint = sys.argv[i + 1]
        elif arg.startswith('--checkpoint='):
            checkpoint = arg.split('=', 1)[1]
    execution = {'status': 'running', 'started_time': time.time(),
                 'arguments': sys.argv[1:], 'trainroom_directory': os.environ.get('TRAIN_TERMINAL_RUN_DIR')}
    execution['checkpoint_sha256'] = sha256(checkpoint) if Path(checkpoint).is_file() else None
    target = Path(output) / 'execution.json'
    atomic_json(target, execution)
    try:
        evaluate()
        execution['status'] = 'completed'
    except BaseException as error:
        execution['status'] = 'interrupted' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'failed'
        execution['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        execution['ended_time'] = time.time()
        atomic_json(target, execution)
        generate(f'{output}/summary.json')


if __name__ == '__main__':
    main()
