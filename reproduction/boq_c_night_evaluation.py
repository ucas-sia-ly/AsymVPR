"""Frozen epoch-10 BoQ comparison on existing night protocols; never selects models."""
from pathlib import Path
from types import SimpleNamespace
import fcntl
import gc
import json
import signal

import numpy as np
import torch
from torch.utils.data import DataLoader,Subset

from reproduction import evaluate as ev
from reproduction.benchmarks import json_sha256,sha256
from reproduction.boq_protocol import MixedBoQ,ValidationImages
from reproduction.train_dino_gsv_mixed import load_config
from reproduction.gsv_night_transfer import protocols
from reproduction.feature_io import BoundedFeatureRows
from reproduction.night_backfill import checked_metrics,ExtractionLogger
from reproduction.monitoring import TrainingLogger
from reproduction.run import configure_cuda

OUT=Path('runs/boq_c_multi_night_epoch10_v2')
BASE=Path('runs/boq_dinob_c_distillation_v1')
ABL=Path('runs/boq_dinob_c_ablation_control_v2')
SPECS={
    'baseline':Path('runs/boq_dinob_official_real_memory_v2/real/best.pt'),
    'C_no_distill':BASE/'control/checkpoints/last.ckpt',
    'C_distill':BASE/'distill/checkpoints/last.ckpt',
    **{f'no_{op}':ABL/op/'control/checkpoints/last.ckpt' for op in ('saturation','white_balance','defocus')},
}
TRANSFORM='RGB-uint8/Resize322-bicubic-antialias/float32/ImageNet-normalize'


def load_weights(path):
    state=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    if 'state_dict' in state:
        if state['epoch']!=9 or state['global_step']!=9770:
            raise ValueError('Comparison requires completed epoch 10')
        model={k.removeprefix('model.'):v for k,v in state['state_dict'].items() if k.startswith('model.')}
        return model,dict(epoch=10,step=9770,selection='fixed final epoch; no test selection')
    return state['model'],dict(epoch=state['epoch'],step=state['step'],selection='pre-finetuning MSLS-selected baseline')


def extract(model,paths,target,identity,config,logger,phase):
    values,start=ev.open_feature_cache(target,identity)
    if start==len(paths):
        del values;return
    dataset=ValidationImages(SimpleNamespace(references=list(paths),queries=[]),config['image_size'])
    loader=DataLoader(Subset(dataset,range(start,len(paths))),batch_size=128,num_workers=2,
                      prefetch_factor=1,pin_memory=False)
    for images,indices in loader:
        with torch.inference_mode():output=model(images.cuda()).float().cpu().numpy()
        if (not np.array_equal(indices.numpy(),np.arange(start,start+len(indices)))
                or output.shape!=(len(indices),config['descriptor_dim'])
                or not np.isfinite(output).all()
                or not np.allclose(np.linalg.norm(output,axis=1),1.,atol=5e-4,rtol=5e-4)):
            raise ValueError('Invalid ordered descriptors')
        values[start:start+len(output)]=output;start+=len(output);values.flush()
        progress=dict(rows=start,total=len(paths),identity_sha256=json_sha256(identity))
        if start==len(paths):progress['feature_sha256']=sha256(target)
        ev.atomic_json(target.with_suffix('.progress.json'),progress)
        logger.log(start,**{f'extract/{phase}/images':start})
        if ev.STOP:raise SystemExit(130)
    del values


def report(summary):
    ev.atomic_json(OUT/'summary.json',summary)
    lines=['# BoQ C版跨夜间数据集评测','','每格为 R@1 / R@5 / R@10（%）。微调模型固定第10轮，不用测试结果重选权重。',
           '所有模型使用各自同一编码器编码图库与查询，FP32，关闭TF32，相同预处理与精确检索。','',
           '| 协议 | 查询数 | '+' | '.join(SPECS)+' |','|---|---:|'+'---|'*len(SPECS)]
    for name,desc in summary['protocols'].items():
        cells=[]
        for label in SPECS:
            m=summary['results'].get(name,{}).get(label,{}).get('metrics_percent')
            cells.append(' / '.join(f'{m[f"R@{k}"]:.2f}' for k in (1,5,10)) if m else '待完成')
        lines.append(f'| {name} | {desc["queries"]["count"]} | '+' | '.join(cells)+' |')
    lines+=['','RobotCar public与DarkDriving为对应公开诊断；MSLS为自定义夜间子集。',
            'Gardens两行共用夜间查询；SPED不是严格夜间集，单独作为混合光照诊断。',
            '缺失协议：'+json.dumps(summary['unavailable'],ensure_ascii=False),f'状态：{summary["status"]}']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'.eval.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        benches,missing=protocols()
        benches=[b for b,_ in benches]
        order=['gardenspoint_night_day_right','gardenspoint_night_day_left','svox_test_night',
               'robotcar_seasons_public_night','robotcar_seasons_public_night_rain','tokyo105_night',
               'msls_val_night_day','msls_test_night_day','darkdriving_paired_diagnostic',
               'sped_mixed_illumination_diagnostic']
        benches.sort(key=lambda b:order.index(b.name))
        config=load_config('configs/boq_dinob_official_real_memory_v2.json')
        config['precision']='float32'
        specs={}
        for label,path in SPECS.items():
            weights,metadata=load_weights(path)
            specs[label]=dict(path=str(path),sha256=sha256(path),**metadata)
            del weights
        identity=dict(models=specs,protocols={b.name:b.describe() for b in benches},unavailable=missing,
                      transform=TRANSFORM,precision='float32',tf32=False,
                      sources={p:sha256(p) for p in (__file__,'reproduction/boq_protocol.py',
                        'src/boq.py','src/backbones.py','reproduction/evaluate.py')})
        receipt=OUT/'identity.json'
        if receipt.exists() and json.loads(receipt.read_text())!=identity:raise RuntimeError('Evaluation identity changed')
        ev.atomic_json(receipt,identity)
        if (OUT/'completed.json').exists():return
        summary=(json.loads((OUT/'summary.json').read_text()) if (OUT/'summary.json').exists() else
                 dict(status='running',models=specs,protocols=identity['protocols'],unavailable=missing,results={}))
        torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
        configure_cuda()
        signal.signal(signal.SIGINT,ev.request_stop);signal.signal(signal.SIGTERM,ev.request_stop)
        report(summary)
        total=len(benches)*len(SPECS)
        with TrainingLogger(config=identity,total_steps=total) as logger:
            # Protocol-first so a complete six-model comparison appears early.
            models={}
            for b in benches:
                for label,path in SPECS.items():
                    dest=OUT/b.name;dest.mkdir(exist_ok=True)
                    if label in summary['results'].get(b.name,{}):
                        row=summary['results'][b.name][label]
                        if sha256(dest/f'{label}_top10.npy')!=row['predictions_sha256']:raise RuntimeError('Prediction checksum changed')
                        continue
                    if ev.STOP:raise SystemExit(130)
                    # Only one model on GPU at a time, retaining a small CPU model cache.
                    model=MixedBoQ(config)
                    weights,_=load_weights(path);model.load_state_dict(weights,strict=True);del weights
                    model=model.float().cuda().eval()
                    files=[]
                    for role,paths in [('gallery',b.references),('query',b.queries)]:
                        fi=ev.feature_identity(paths,config,specs[label]['sha256'],'symmetric_'+role)
                        fi.update(transform=TRANSFORM,implementation_sha256=identity['sources'])
                        target=OUT/'cache'/json_sha256(fi)/'features.npy'
                        extract(model,paths,target,fi,config,
                            ExtractionLogger(logger,sum(len(r) for r in summary['results'].values())),
                            f'{b.name}/{label}/{role}')
                        files.append(target)
                    del model;gc.collect();torch.cuda.empty_cache()
                    with BoundedFeatureRows(files[0]) as refs,BoundedFeatureRows(files[1]) as queries:
                        predictions,_=ev.exact_topk(refs,queries,min(10,len(refs)),64,8192,'cuda')
                    metrics,hits=checked_metrics(predictions,b)
                    np.save(dest/f'{label}_top10.npy',predictions);np.save(dest/f'{label}_hits.npy',hits)
                    row=dict(metrics_percent=metrics,checkpoint=specs[label],
                        predictions_sha256=sha256(dest/f'{label}_top10.npy'),hits_sha256=sha256(dest/f'{label}_hits.npy'),
                        gallery_sha256=json.loads(files[0].with_suffix('.progress.json').read_text())['feature_sha256'],
                        query_sha256=json.loads(files[1].with_suffix('.progress.json').read_text())['feature_sha256'])
                    ev.atomic_json(dest/f'{label}.json',row)
                    summary['results'].setdefault(b.name,{})[label]=row;report(summary)
                    completed=sum(len(r) for r in summary['results'].values())
                    logger.log(completed,**{f'eval/{b.name}/{label}/{k}':v for k,v in metrics.items()})
                    print(f'{completed}/{total} {b.name} {label}: {metrics}',flush=True)
            summary['status']='completed';report(summary)
            ev.atomic_json(OUT/'completed.json',dict(pairs=total,summary_sha256=sha256(OUT/'summary.json')))


if __name__=='__main__':main()
