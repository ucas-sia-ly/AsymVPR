"""Read-only checks before inference or an explicitly requested training run."""
import argparse
import json
from pathlib import Path
import shutil
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/boq_b2.json')
    args = parser.parse_args()
    c = json.loads(Path(args.config).read_text())
    import torch
    import torchvision
    import timm
    report = {'python': sys.executable, 'torch': torch.__version__,
              'torchvision': torchvision.__version__, 'timm': timm.__version__,
              'cuda_available': torch.cuda.is_available(), 'paths': {},
              'disk_free_gib': round(shutil.disk_usage(Path.cwd()).free / 1024**3, 2)}
    for key in ('gsv_root', 'msls_root', 'teacher_weights', 'dino_source'):
        p = Path(c[key]).expanduser()
        report['paths'][key] = {'path': str(p), 'exists': p.exists()}
    if report['cuda_available']:
        free, total = torch.cuda.mem_get_info()
        report['gpu'] = {'name': torch.cuda.get_device_name(),
                         'total_gib': round(total / 1024**3, 2),
                         'free_gib': round(free / 1024**3, 2)}
    p = Path('/proc/meminfo')
    if p.exists():
        report['available_ram_gib'] = round(int(next(line.split()[1] for line in
            p.read_text().splitlines() if line.startswith('MemAvailable:'))) / 1024**2, 2)
    history = Path('results/training_history.json')
    if history.exists():
        report['existing_training'] = json.loads(history.read_text())['training']
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report['cuda_available'] or not all(p['exists'] for p in report['paths'].values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
