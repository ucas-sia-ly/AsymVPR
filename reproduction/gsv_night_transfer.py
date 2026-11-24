"""Evaluate fixed GSV-release checkpoints on the local night protocols."""
import argparse
import fcntl
import json
from pathlib import Path
import signal

import numpy as np
import torch

from reproduction import evaluate
from reproduction.asym_model import EfficientViTBoQ
from reproduction.benchmarks import Benchmark, build_benchmark, json_sha256, sha256
from reproduction.extra_night_benchmarks import EXTRA_BUILDERS
from reproduction.feature_io import BoundedFeatureRows
from reproduction.monitoring import TrainingLogger
from reproduction.night_backfill import ExtractionLogger, checked_metrics, snapshot_identity
from reproduction.night_benchmarks import BUILDERS
from reproduction.night_finalize import tokyo_night_layout
from reproduction.night_suite import verify_cache
from reproduction.run import configure_cuda

OUT = Path('runs/gsv_night_transfer')
BASE = Path('runs/gsv_night_comparison')
RUNS = {
    'A': BASE/'A/train',
    'B': BASE/'B/train',
    'C': BASE/'C/train',
    'C_minus_detail': Path('runs/gsv_night_ablation/confirm/C_minus_detail/train'),
}


class HistoricalStudent(EfficientViTBoQ):
    """Use the same attention computation as the four original evaluations."""
    def forward(self, images):
        return self.aggregator(self.backbone(images)[-1], return_attention=True)[0]


def protocols():
    tokyo = build_benchmark('tokyo247')
    layout = tokyo_night_layout(tokyo)
    if not layout['available'] or len(layout['indices']) != 105:
        raise ValueError('Verified Tokyo105 labels are required')
    night = Benchmark('tokyo105_night', tokyo.root, tokyo.references,
        [tokyo.queries[i] for i in layout['indices']], layout['positives'],
        layout['definition'], tokyo.source_files)
    old = json.loads(Path('results/night_backfill_v2/summary.json').read_text())
    original = json.loads(Path('results/evaluation/summary.json').read_text())
    sources = {name: next(iter(rows.values()))['gallery_cache']
               for name, rows in old['results'].items() if name != 'tokyo105_night'}
    sources[night.name] = original['datasets']['tokyo247']['gallery_cache']
    builders = {**BUILDERS, **EXTRA_BUILDERS}
    benches = [night]
    unavailable = {}
    for key, builder in builders.items():
        try:
            benches.append(builder())
        except FileNotFoundError as exc:
            if key != 'robotcar':
                raise
            unavailable['robotcar_gps_night_diagnostic'] = str(exc.filename)
    ordered = ('tokyo105_night', 'svox_test_night', 'robotcar_seasons_public_night',
               'robotcar_seasons_public_night_rain', 'gardenspoint_night_day_right',
               'gardenspoint_night_day_left', 'msls_test_night_day', 'msls_val_night_day',
               'sped_mixed_illumination_diagnostic', 'darkdriving_paired_diagnostic',
               'robotcar_gps_night_diagnostic')
    by_name = {b.name: b for b in benches}
    return [(by_name[name], sources[name]) for name in ordered if name not in unavailable], unavailable


def report(summary):
    evaluate.atomic_json(OUT/'summary.json', summary)
    lines = ['# GSV 最终版夜间跨域评测', '',
        '固定最佳权重。每格为 R@1 / R@5 / R@10（%）。',
        'Gardens Point 两行共享查询。SPED 是混合光照诊断。', '',
        '| 协议 | 查询数 | A 无退化 | B 旧七项 | C 新版 | C 去掉细节族 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, definition in summary['protocols'].items():
        scores = [summary['results'].get(name, {}).get(label, {}).get('metrics_percent') for label in RUNS]
        values = [' / '.join(f'{m[f"R@{k}"]:.2f}' for k in (1,5,10)) if m else '待完成' for m in scores]
        lines.append(f'| {name} | {definition["queries"]["count"]} | ' + ' | '.join(values) + ' |')
    lines += ['', f"状态：{summary['status']}。"]
    for name, missing in summary['unavailable'].items():
        lines.append(f'- {name} 缺少本地源文件 `{missing}`。')
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT/'.evaluation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = json.loads(Path('configs/boq_b2.json').read_text())
    teacher = sha256(config['teacher_weights'])
    specs = {}
    states = {}
    for label, directory in RUNS.items():
        path = directory/'best.pt'
        done_path = directory/'completed.json'
        done = json.loads(done_path.read_text())
        state = evaluate.load_inference_checkpoint(path)
        evaluate.validate_inference_checkpoint(state, config, teacher)
        selected = done['best']
        if (done['status'] != 'completed' or not selected['adapted']
                or any(state[k] != selected[k] for k in ('step','epoch'))
                or state['metrics']['svox_dev']['metrics_percent']['R@1'] != selected['score']
                or state['config'] != done['config'] or state['adaptation_lineage'] != done['lineage']):
            raise ValueError(f'Checkpoint differs from recorded selection: {label}')
        specs[label] = dict(path=str(path), sha256=sha256(path), completion_sha256=sha256(done_path),
                            epoch=selected['epoch'], step=selected['step'])
        states[label] = state
    benches, unavailable = protocols()
    descriptions = {b.name: b.describe() for b, _ in benches}
    registry = dict(models=specs, teacher_sha256=teacher, config_sha256=sha256('configs/boq_b2.json'),
                    gallery_sources={b.name: s for b,s in benches}, unavailable=unavailable)
    identity = snapshot_identity(registry, descriptions)
    identity['implementation_sha256'][__file__] = sha256(__file__)
    identity_path = OUT/'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Evaluation identity changed')
    evaluate.atomic_json(identity_path, identity)
    if (OUT/'complete.json').is_file():
        receipt = json.loads((OUT/'complete.json').read_text())
        if receipt['summary_sha256'] != sha256(OUT/'summary.json'):
            raise ValueError('Completed summary changed')
        return
    signal.signal(signal.SIGINT, evaluate.request_stop)
    signal.signal(signal.SIGTERM, evaluate.request_stop)
    torch.set_num_threads(6)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    configure_cuda()
    models = {}
    for label, state in states.items():
        student = HistoricalStudent(config, pretrained=False).eval()
        student.load_state_dict(state['model'], strict=True)
        models[label] = student.cuda()
    del state, states
    summary = dict(status='running', models=specs, teacher_sha256=teacher,
                   precision='float32', tf32=False, protocols=descriptions, unavailable=unavailable, results={})
    report(summary)
    verified = {}
    pairs, total = 0, len(benches)*len(models)
    with TrainingLogger(config=registry, total_steps=total) as logger:
        for b, source in benches:
            gallery = Path(source['path'])
            gi = evaluate.feature_identity(b.references, config, teacher, 'gallery_teacher')
            key = (gallery, json_sha256(gi))
            if key not in verified:
                verified[key] = verify_cache(gallery, gi)
            if verified[key] != source['feature_sha256']:
                raise ValueError(f'Gallery bytes differ from historical receipt: {b.name}')
            dest = OUT/b.name
            dest.mkdir(exist_ok=True)
            hits_by_label = {}
            for label, model in models.items():
                if evaluate.STOP:
                    raise SystemExit(130)
                qi = evaluate.feature_identity(b.queries, config, specs[label]['sha256'], 'query_student')
                query = OUT/'cache'/json_sha256(qi)/'features.npy'
                if not evaluate.cache_complete(query, qi):
                    evaluate.extract_features(model, b.queries, query, qi, config, 'cuda',
                        args.batch_size,args.workers,ExtractionLogger(logger,pairs),f'{b.name}/{label}')
                query_sha = verify_cache(query, qi)
                with BoundedFeatureRows(gallery) as refs, BoundedFeatureRows(query) as queries:
                    pred, _ = evaluate.exact_topk(refs,queries,10,64,4096,'cuda',
                        reference_bank='auto', cuda_reserve_gib=4.)
                metrics, hits = checked_metrics(pred,b)
                hits_by_label[label] = hits
                predpath, hitpath = dest/f'{label}_top10.npy', dest/f'{label}_hits.npy'
                np.save(predpath,pred)
                np.save(hitpath,hits)
                paired = {}
                for ref, ref_hits in hits_by_label.items():
                    if ref != label:
                        paired[ref] = {f'R@{k}': dict(gained=int(np.sum(hits[:,i] & ~ref_hits[:,i])),
                            lost=int(np.sum(~hits[:,i] & ref_hits[:,i]))) for i,k in enumerate((1,5,10))}
                row = dict(metrics_percent=metrics, paired_hits=paired, checkpoint_sha256=specs[label]['sha256'],
                    gallery_sha256=verified[key], query_features_sha256=query_sha, predictions_sha256=sha256(predpath),
                    hits_sha256=sha256(hitpath))
                evaluate.atomic_json(dest/f'{label}.json',row)
                summary['results'].setdefault(b.name,{})[label] = row
                pairs += 1
                logger.log(pairs, **{f'eval/{b.name}/{label}/{k}':v for k,v in metrics.items()},
                           **{'eval/completed_pairs':pairs})
                report(summary)
                print(f'{pairs}/{total} {b.name} {label}: {metrics}',flush=True)
                torch.cuda.empty_cache()
    summary['status'] = 'complete'
    report(summary)
    evaluate.atomic_json(OUT/'complete.json',dict(pairs=pairs,summary_sha256=sha256(OUT/'summary.json'),
                         report_sha256=sha256(OUT/'REPORT.md'),identity_sha256=json_sha256(identity)))


if __name__ == '__main__':
    main()
