"""Evaluate retained original/mixed/mixed+camera checkpoints, never train/select.

Reuse audited read-only caches via hard links after full identity/checksum checks.
Only generic evaluation and resource I/O live here, no retired experiment logic.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
from reproduction.benchmarks import json_sha256, sha256
from reproduction.evaluate import (atomic_json, feature_identity, extract_features,
    load_inference_checkpoint, validate_inference_checkpoint, exact_topk, recall_at_k)
from reproduction.feature_io import BoundedFeatureRows
from reproduction.night_benchmarks import BUILDERS

ARCHIVE = Path('已放弃/night_confusion/results/extended_night_v1')
MODELS = {
    'original': Path('checkpoints/boq_b2_best.pt'),
    'mixed': Path('runs/boq_b2_night_v1/train/best.pt'),
    'mixed_camera': Path('runs/boq_b2_mixed_camera_v1/train/best.pt'),
}


def verify_cache(path, identity):
    """Validate completion and all feature bytes without opening a writable mmap."""
    path = Path(path)
    meta = json.loads(path.with_suffix('.provenance.json').read_text())
    progress = json.loads(path.with_suffix('.progress.json').read_text())
    if meta != identity or progress.get('identity_sha256') != json_sha256(identity):
        raise ValueError(f'Cache identity mismatch: {path}')
    expected = identity['paths']['count']
    if progress.get('rows') != expected or progress.get('total') != expected:
        raise ValueError(f'Cache is incomplete: {path}')
    with BoundedFeatureRows(path) as f:
        if f.shape != (expected,identity['descriptor_dim']):
            raise ValueError(f'Cache shape mismatch: {path}')
    checksum = sha256(path)
    if progress.get('feature_sha256') != checksum:
        raise ValueError(f'Cache checksum mismatch: {path}')
    return checksum


def reuse_cache(source, target, identity):
    checksum = verify_cache(source, identity)
    target = Path(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if verify_cache(target,identity) != checksum:
            raise ValueError('Existing promoted cache differs from verified source')
    else:
        os.link(source,target)
        atomic_json(target.with_suffix('.provenance.json'),identity)
        atomic_json(target.with_suffix('.progress.json'),json.loads(Path(source).with_suffix('.progress.json').read_text()))
    return dict(reused_from=str(Path(source).resolve()),feature_sha256=checksum,
                reuse='verified hard link; no feature copy or mutation')


def existing_tokyo(checkpoints):
    from reproduction.night_finalize import _read_sealed
    metrics, evidence, definition = {},{},None
    initial = json.loads(Path('results/evaluation/summary.json').read_text())
    if initial['checkpoint_sha256'] != checkpoints['original']:
        raise ValueError('Original Tokyo checkpoint mismatch')
    for label,folder in [('mixed','boq_b2_night_v1'),('mixed_camera','boq_b2_mixed_camera_v1')]:
        root = Path('runs')/folder
        cp = root/'final_evaluation/comparison.json'
        receipt = _read_sealed(root/'finalization.json')
        if receipt['outputs_sha256'].get(str(cp.resolve())) != sha256(cp):
            raise ValueError('Tokyo comparison does not match completed receipt')
        record = _read_sealed(cp)
        final = json.loads((root/'final_evaluation/summary.json').read_text())
        if final['checkpoint_sha256'] != checkpoints[label]:
            raise ValueError('Adapted Tokyo checkpoint mismatch')
        row = record['tokyo_night']
        if not row['available']:
            raise ValueError('Existing Tokyo night ground truth unavailable')
        active = dict(definition=row['definition'],query_ids=row['query_ids'],num_queries=row['num_queries'])
        if definition is not None and active != definition:
            raise ValueError('Tokyo night protocols differ between retained models')
        definition = active
        for model,key in [('original','initial_percent'),(label,'final_percent')]:
            if model in metrics and metrics[model]['metrics_percent'] != row[key]:
                raise ValueError('Original Tokyo night metrics disagree')
            metrics[model]=dict(metrics_percent=row[key],query_count=row['num_queries'],
                checkpoint_sha256=checkpoints[model],existing_evaluation=True)
        evidence[label]=dict(comparison=str(cp.resolve()),comparison_sha256=sha256(cp),
            finalization_sha256=sha256(root/'finalization.json'))
    return metrics,dict(protocol=definition,reused_evidence=evidence)


def report(out, summary):
    atomic_json(out/'summary.json',summary)
    lines=['# 混合训练与混合＋退化：真实夜间扩展评测','',
        '数值为 R@1 / R@5 / R@10（%）。同一行使用相同图库、查询顺序、真值和冻结教师。',
        '测试使用原始图片，预处理为 ToTensor → bicubic Resize → ImageNet Normalize；不附加人工退化。','',
        '| 协议 | Query 数 | 原始模型 | 混合 | 混合＋退化 |','| --- | ---: | ---: | ---: | ---: |']
    for dataset in summary['protocols']:
        result=summary['results'].get(dataset,{})
        count=next((x['query_count'] for x in result.values()),'—')
        values=[]
        for label in MODELS:
            m=result.get(label,{}).get('metrics_percent')
            values.append(' / '.join(f'{m[f"R@{k}"]:.4f}' for k in (1,5,10)) if m else '待完成')
        lines.append(f'| {dataset} | {count} | '+' | '.join(values)+' |')
    lines += ['', 'MSLS 为夜间标签与白天图库的自定义子集，不是官方完整 all/subtask 分数。',
        'SVOX test-night 为 test 划分的夜间查询，25m UTM 正例；与用于选模的 train-night 开发集分开报告。',
        'DarkDriving 为严格同名配对诊断；附近帧可能是地点正例，但在此计为负例。',
        'RobotCar 为本地 GPS 采样的 25m 检索诊断，不是 RobotCar-Seasons 官方定位协议。',
        'Tokyo105 为既有固定诊断结果，经权重、协议与封存收据核验后汇总，不伪装成本次重新推理。',
        '仅报告固定检查点，不利用本次测试调参或重新选模。','',f'状态：{summary["status"]}。','']
    (out/'REPORT.md').write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--output',default='results/mixed_night_suite_v1')
    p.add_argument('--datasets',nargs='+',choices=list(BUILDERS),default=list(BUILDERS))
    p.add_argument('--manifest-only',action='store_true')
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--workers',type=int,default=1)
    args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    benches=[BUILDERS[name]() for name in args.datasets]
    descriptions={b.name:b.describe() for b in benches}
    old_descriptions=json.loads((ARCHIVE/'manifests.json').read_text())
    for name,desc in descriptions.items():
        if old_descriptions[name] != desc:
            raise ValueError(f'Dataset differs from audited cache protocol: {name}')
        print(f'{name}: gallery={desc["references"]["count"]}, query={desc["queries"]["count"]}',flush=True)
    atomic_json(out/'manifests.json',descriptions)
    atomic_json(out/'coverage.json',json.loads((ARCHIVE/'coverage.json').read_text()))
    if args.manifest_only:
        return
    import torch
    import reproduction.evaluate as evaluation
    from reproduction.asym_model import EfficientViTBoQ
    from reproduction.monitoring import TrainingLogger
    signal.signal(signal.SIGINT,evaluation.request_stop)
    signal.signal(signal.SIGTERM,evaluation.request_stop)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    free,total=torch.cuda.mem_get_info()
    budget=min(8*1024**3,free-4*1024**3)
    if budget<2*1024**3:
        raise RuntimeError('GPU free memory too low after 4GiB reserve')
    torch.cuda.set_per_process_memory_fraction(budget/total)
    config=json.loads(Path('configs/boq_b2.json').read_text())
    teacher_sha=sha256(config['teacher_weights'])
    checkpoints={label:sha256(path) for label,path in MODELS.items()}
    summary=dict(status='running',operation='evaluation only; fixed selected checkpoints',
        precision='float32',tf32=False,teacher_sha256=teacher_sha,
        models={k:dict(checkpoint=str(v.resolve()),sha256=checkpoints[k]) for k,v in MODELS.items()},
        protocols=dict(descriptions),results={},resources=dict(batch_size=args.batch_size,workers=args.workers,
        cuda_allocator_budget_bytes=budget,cuda_free_at_start_bytes=free,cuda_reserve_bytes=4*1024**3))
    tokyo,tokyo_protocol=existing_tokyo(checkpoints)
    summary['protocols']['tokyo105_night_existing']=tokyo_protocol
    summary['results']['tokyo105_night_existing']=tokyo
    report(out,summary)
    with TrainingLogger(config=dict(operation='night evaluation',**vars(args))) as logger:
        for label,row in tokyo.items():
            logger.log(0,**{f'eval/tokyo105_night/{label}/{k}':v for k,v in row['metrics_percent'].items()})
        for stage,b in enumerate(benches,1):
            if evaluation.STOP:
                raise SystemExit(130)
            gallery=out/'cache'/b.name/'gallery.npy'
            gi=feature_identity(b.references,config,teacher_sha,'gallery_teacher')
            gallery_note=reuse_cache(ARCHIVE/'cache'/b.name/'gallery.npy',gallery,gi)
            for label,ckpt in MODELS.items():
                state=load_inference_checkpoint(ckpt)
                validate_inference_checkpoint(state,config,teacher_sha)
                qi=feature_identity(b.queries,config,checkpoints[label],'query_student')
                qt=gallery.parent/f'{label}.npy'
                query_note={}
                if label=='mixed':
                    query_note=reuse_cache(ARCHIVE/'cache'/b.name/'mixed.npy',qt,qi)
                elif qt.exists() and json.loads(qt.with_suffix('.progress.json').read_text()).get('rows')==len(b.queries):
                    verify_cache(qt,qi)
                else:
                    model=EfficientViTBoQ(config,pretrained=False)
                    model.load_state_dict(state['model'],strict=True)
                    del state
                    model=model.float().cuda().eval()
                    extract_features(model,b.queries,qt,qi,config,'cuda',args.batch_size,args.workers,logger,b.name+'/'+label)
                    del model;gc.collect();torch.cuda.empty_cache()
                    state=None
                del state
                query_sha=verify_cache(qt,qi)
                with BoundedFeatureRows(gallery) as refs,BoundedFeatureRows(qt) as queries:
                    search={}
                    pred,scores=exact_topk(refs,queries,10,32,2048,'cuda',reference_bank='auto',
                        cuda_reserve_gib=4,search_info=search)
                metrics=recall_at_k(pred,b.positives,(1,5,10))
                hits=np.asarray([[bool(np.isin(row[:k],pos).any()) for k in (1,5,10)] for row,pos in zip(pred,b.positives)])
                if not all(np.isclose(100*hits[:,i].mean(),metrics[f'R@{k}']) for i,k in enumerate((1,5,10))):
                    raise AssertionError('Per-query hits disagree with reported Recall')
                dest=out/b.name;dest.mkdir(exist_ok=True)
                np.save(dest/f'{label}_top10.npy',pred)
                np.save(dest/f'{label}_hits.npy',hits)
                row=dict(metrics_percent=metrics,hits_at_k={str(k):int(hits[:,i].sum()) for i,k in enumerate((1,5,10))},
                    query_count=len(b.queries),gallery_count=len(b.references),checkpoint_sha256=checkpoints[label],
                    teacher_sha256=teacher_sha,protocol_sha256=descriptions[b.name]['manifest_sha256'],
                    gallery_cache=dict(path=str(gallery.resolve()),**gallery_note),
                    query_cache=dict(path=str(qt.resolve()),feature_sha256=query_sha,**{k:v for k,v in query_note.items() if k!='feature_sha256'}),
                    predictions_sha256=sha256(dest/f'{label}_top10.npy'),hits_sha256=sha256(dest/f'{label}_hits.npy'),search_memory=search)
                atomic_json(dest/f'{label}_metrics.json',row)
                summary['results'].setdefault(b.name,{})[label]=row
                report(out,summary)
                logger.log(stage,**{f'eval/{b.name}/{label}/{k}':v for k,v in metrics.items()})
                print(json.dumps(dict(dataset=b.name,model=label,**metrics)),flush=True)
                gc.collect();torch.cuda.empty_cache()
    summary['status']='complete'
    report(out,summary)
    atomic_json(out/'complete.json',dict(status='complete',summary_sha256=sha256(out/'summary.json'),
        report_sha256=sha256(out/'REPORT.md')))


if __name__=='__main__':
    main()
