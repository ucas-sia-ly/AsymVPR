"""Evaluate the matched epoch-2 GSV camera ablations on fixed night protocols."""
import argparse
import fcntl
import json
from pathlib import Path
import signal

import numpy as np
import torch

from reproduction import evaluate
from reproduction.benchmarks import json_sha256, sha256
from reproduction.data import Images, loader
from reproduction.feature_io import BoundedFeatureRows
from reproduction.gsv_night_transfer import BASE, HistoricalStudent, protocols
from reproduction.monitoring import TrainingLogger
from reproduction.night_backfill import checked_metrics, snapshot_identity
from reproduction.night_suite import verify_cache
from reproduction.run import configure_cuda

ROOT = Path('runs/gsv_night_ablation')
OUT = ROOT/'transfer_epoch02'
FAMILIES = {
    'Tokyo105': ('tokyo105_night',),
    'SVOX': ('svox_test_night',),
    'RobotCar-Seasons': ('robotcar_seasons_public_night', 'robotcar_seasons_public_night_rain'),
    'Gardens Point': ('gardenspoint_night_day_right', 'gardenspoint_night_day_left'),
    'MSLS test': ('msls_test_night_day',),
}
SUPPLEMENTARY = ('msls_val_night_day', 'sped_mixed_illumination_diagnostic')
PRIMARY = tuple(name for family in FAMILIES.values() for name in family)


def read(path):
    return json.loads(path.read_text())


def models(config, teacher):
    plan = read(ROOT/'plan.json')
    entries = [(label, BASE/label/'train', None) for label in 'ABC']
    entries.extend((row['tag'], Path(row['output']), row['group']) for row in plan['trials'])
    specs, students = {}, {}
    common = None
    for label, directory, parent in entries:
        path = directory/'epoch_02.pt'
        done = read(directory/'completed.json')
        state = evaluate.load_inference_checkpoint(path)
        evaluate.validate_inference_checkpoint(state, config, teacher)
        lineage, cfg = done['lineage'], done['config']
        cohort = (lineage['init_sha256'], lineage['snapshot']['manifest_sha256'],
                  lineage['night_teacher_identity'], cfg['seed'], cfg['night_batch_size'])
        if common is None:
            common = cohort
        if (common != cohort or done['status'] != 'completed' or state['epoch'] != 2
                or state['step'] != 3682 or state['config'] != cfg
                or state['adaptation_lineage'] != lineage
                or cfg.get('schedule_epochs',cfg['epochs']) != 10
                or (parent is not None and (done['epochs'] != 2 or done['steps'] != state['step']))):
            raise ValueError(f'Unmatched epoch-2 evaluation checkpoint: {label}')
        if parent is not None and not cfg.get('gsv_camera_omit'):
            raise ValueError(f'Expected an omission: {label}')
        specs[label] = dict(path=str(path), sha256=sha256(path), parent=parent,
                            removed=cfg.get('gsv_camera_omit',[]), epoch=2, step=state['step'])
        student = HistoricalStudent(config, pretrained=False).eval()
        student.load_state_dict(state['model'], strict=True)
        students[label] = student.cuda()
    return specs, students


def search(bench, students, config, gallery, batch_size, workers):
    """Decode each query once and search one verified gallery for all models."""
    count, dim = len(bench.queries), config['descriptor_dim']
    all_queries = np.empty((len(students),count,dim),dtype=np.float32)
    images = Images('/', np.asarray([str(p) for p in bench.queries]), config['image_size'])
    seen = np.zeros(count,dtype=bool)
    for batch, indices in loader(images,batch_size,workers):
        if evaluate.STOP:
            raise SystemExit(130)
        batch = batch.cuda(non_blocking=True)
        indices = indices.numpy()
        with torch.inference_mode():
            for j, student in enumerate(students.values()):
                values = student(batch).cpu().numpy()
                if values.dtype != np.float32 or not np.isfinite(values).all():
                    raise FloatingPointError('Invalid query descriptors')
                all_queries[j,indices] = values
        seen[indices] = True
    if not seen.all():
        raise ValueError('Missing query descriptors')
    with BoundedFeatureRows(gallery) as references:
        pred, _ = evaluate.exact_topk(references,all_queries.reshape(-1,dim),10,64,4096,
                                      'cuda',reference_bank='auto',cuda_reserve_gib=4.)
    return pred.reshape(len(students),count,10)


def compute_rank(summary):
    if not set(PRIMARY).issubset(summary['results']):
        return {}
    deltas = {}
    for label, spec in summary['models'].items():
        parent = spec['parent']
        if parent is None:
            continue
        by_protocol = {name: rows[label]['metrics_percent']['R@1']-rows[parent]['metrics_percent']['R@1']
                       for name, rows in summary['results'].items()}
        by_family = {family: float(np.mean([by_protocol[name] for name in names]))
                     for family,names in FAMILIES.items()}
        deltas[label] = dict(parent=parent, removed=spec['removed'], deltas_pp=by_protocol,
            family_deltas_pp=by_family, family_macro_pp=float(np.mean(list(by_family.values()))),
            families_up=sum(x>1e-8 for x in by_family.values()), families_down=sum(x < -1e-8 for x in by_family.values()),
            worst_protocol_pp=min(by_protocol[name] for name in PRIMARY))
    return deltas


def write_report(summary):
    evaluate.atomic_json(OUT/'summary.json',summary)
    ranked = compute_rank(summary)
    evaluate.atomic_json(OUT/'ranking.json',ranked)
    lines = ['# GSV 退化逐项消融 · 第 2 轮跨域结果','',
             '所有模型固定在第 2 轮。表中为移除退化后相对同家族完整策略的 R@1 变化（百分点）。',
             '主汇总等权看五个数据集家族，RobotCar 的夜/夜雨与 Gardens Point 的两种图库各自在家族内取平均。',
             'MSLS val 夜间子集与 SPED 为辅助诊断。DarkDriving 同名配对和 RobotCar GPS 诊断不进入本轮筛选。','']
    for parent in ('B','C'):
        rows = sorted(((label,row) for label,row in ranked.items() if row['parent']==parent),
                      key=lambda item: -item[1]['family_macro_pp'])
        lines.extend([f'## {parent} 家族','',
            '| 移除 | 五家族平均 | Tokyo105 | SVOX | RobotCar Night | RobotCar Rain | Gardens Right | Gardens Left | MSLS test |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|'])
        for label,row in rows:
            values = [row['family_macro_pp'], *(row['deltas_pp'][name] for name in PRIMARY)]
            lines.append('| '+label.removeprefix(parent+'_minus_')+' | '+' | '.join(f'{v:+.2f}' for v in values)+' |')
        lines.append('')
    lines.append(f"进度 {len(summary['results'])}/{len(summary['protocols'])} 个协议。")
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--workers',type=int,default=4)
    args = parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    lock = (OUT/'.evaluation.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGINT,evaluate.request_stop)
    signal.signal(signal.SIGTERM,evaluate.request_stop)
    torch.set_num_threads(6)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    configure_cuda()
    config = read(Path('configs/boq_b2.json'))
    teacher = sha256(config['teacher_weights'])
    specs, students = models(config,teacher)
    benches, _ = protocols()
    benches = [(b,s) for b,s in benches if b.name in PRIMARY+SUPPLEMENTARY]
    descriptions = {b.name:b.describe() for b,s in benches}
    if set(descriptions) != set(PRIMARY+SUPPLEMENTARY):
        raise ValueError('Missing scheduled night protocols')
    registry = dict(models=specs, gallery_sources={b.name:s for b,s in benches},
        teacher_sha256=teacher, config_sha256=sha256('configs/boq_b2.json'),
        source_sha256={str(p):sha256(p) for p in (Path(__file__),Path(__file__).with_name('gsv_night_transfer.py'))},
        families=FAMILIES, supplementary=SUPPLEMENTARY, precision='float32',tf32=False)
    identity = snapshot_identity(registry,descriptions)
    ident = OUT/'identity.json'
    if ident.exists() and read(ident) != json.loads(json.dumps(identity)):
        raise ValueError('Ablation evaluation identity changed')
    evaluate.atomic_json(ident,identity)
    summary = dict(status='running',models=specs,protocols=descriptions,results={})
    if (OUT/'summary.json').is_file():
        summary = read(OUT/'summary.json')
        for name, rows in summary['results'].items():
            if set(rows) != set(students):
                raise ValueError(f'Incomplete saved protocol: {name}')
            for label,row in rows.items():
                for kind in ('top10','hits'):
                    if sha256(OUT/name/f'{label}_{kind}.npy') != row[f'{kind}_sha256']:
                        raise ValueError(f'Saved predictions differ: {name}/{label}')
        if summary['status']=='complete':
            return
    write_report(summary)
    pairs = len(summary['results'])*len(students)
    total = len(benches)*len(students)
    verified = {}
    with TrainingLogger(config=registry,total_steps=total) as logger:
        for bench,source in benches:
            if bench.name in summary['results']:
                continue
            gallery = Path(source['path'])
            gi = evaluate.feature_identity(bench.references, config, teacher,'gallery_teacher')
            key = (gallery,json_sha256(gi))
            if key not in verified:
                verified[key] = verify_cache(gallery,gi)
            if verified[key] != source['feature_sha256']:
                raise ValueError(f'Historical gallery bytes differ: {bench.name}')
            print(f'{bench.name}: {len(bench.queries)} queries × {len(students)} models', flush=True)
            predictions = search(bench,students,config,gallery,args.batch_size,args.workers)
            dest = OUT/bench.name
            dest.mkdir(exist_ok=True)
            results, hits = {}, {}
            for label,pred in zip(students,predictions,strict=True):
                metrics, hits[label] = checked_metrics(pred,bench)
                topfile, hitfile = dest/f'{label}_top10.npy', dest/f'{label}_hits.npy'
                np.save(topfile,pred)
                np.save(hitfile,hits[label])
                row = dict(metrics_percent=metrics, top10_sha256=sha256(topfile),hits_sha256=sha256(hitfile))
                parent = specs[label]['parent']
                if parent:
                    paired = hits[label].astype(np.int8)-hits[parent]
                    row['paired_hits'] = {f'R@{k}':dict(gained=int((paired[:,i]>0).sum()),
                        lost=int((paired[:,i]<0).sum())) for i,k in enumerate((1,5,10))}
                results[label] = row
                pairs += 1
                logger.log(pairs, **{f'eval/{bench.name}/{label}/{k}':v for k,v in metrics.items()},
                           **{'eval/completed_pairs':pairs})
            summary['results'][bench.name] = results
            write_report(summary)
            print(f'{pairs}/{total}: {bench.name} complete',flush=True)
    summary['status'] = 'complete'
    write_report(summary)
    evaluate.atomic_json(OUT/'complete.json',dict(pairs=pairs,summary_sha256=sha256(OUT/'summary.json'),
                                                ranking_sha256=sha256(OUT/'ranking.json')))


if __name__ == '__main__':
    main()
