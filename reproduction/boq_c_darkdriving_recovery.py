"""Explicit strict Pillow fallback for a JPEG rejected by torchvision; no image filtering."""
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from PIL import Image,ImageFile
from reproduction import boq_c_night_evaluation as evaluation
from reproduction.augmentation_sweep import helper
from reproduction.evaluate import atomic_json,sha256

OLD=Path('runs/boq_c_multi_night_epoch10_v2')
SPED=Path('runs/boq_c_multi_night_remaining_v1')
OUT=Path('runs/boq_c_darkdriving_strict_fallback_v1')
OriginalImages=evaluation.ValidationImages
original_protocols=evaluation.protocols


class StrictFallbackImages(OriginalImages):
    def __getitem__(self,index):
        try:return super().__getitem__(index)
        except RuntimeError as exc:
            if 'incomplete or truncated' not in str(exc).lower():raise
            if ImageFile.LOAD_TRUNCATED_IMAGES:raise RuntimeError('Truncated-image tolerance must remain disabled')
            path=self.paths[index]
            with Image.open(path) as image:
                image.load()
                pixels=np.array(image.convert('RGB'),copy=True)
            record=dict(path=str(path),sha256=sha256(path),decoder='strict Pillow RGB',shape=list(pixels.shape))
            fd=os.open(OUT/'decoder_fallback.jsonl',os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o644)
            try:os.write(fd,(json.dumps(record)+'\n').encode())
            finally:os.close(fd)
            return self.transform(torch.from_numpy(pixels).permute(2,0,1)),index


def protocols():
    benches,missing=original_protocols()
    return [(b,s) for b,s in benches if b.name=='darkdriving_paired_diagnostic'],missing


def main():
    while True:
        run=helper('status','--id','e64c4098938c')['run']
        if run['status']=='completed' and run.get('exit_code')==0:break
        if run['status'] not in ('running','queued','stopping'):raise RuntimeError('SPED continuation failed')
        time.sleep(20)
    evaluation.OUT=OUT;evaluation.protocols=protocols;evaluation.ValidationImages=StrictFallbackImages
    evaluation.TRANSFORM += '/strict-Pillow-fallback-on-truncated-JPEG:'+sha256(__file__)
    evaluation.main()
    sources=[OLD/'summary.json',SPED/'summary.json',OUT/'summary.json']
    combined=json.loads(sources[0].read_text())
    for p in sources[1:]:
        extra=json.loads(p.read_text())
        if combined['models']!=extra['models']:raise RuntimeError('Model identities differ')
        combined['results'].update(extra['results'])
        combined['unavailable']=extra['unavailable']
    combined.update(status='completed',completed_pairs=sum(len(v) for v in combined['results'].values()),
        sources={str(p):sha256(p) for p in sources},
        decoder_note='Only torchvision incomplete/truncated errors use strict Pillow RGB; no missing-image tolerance, filtering or image modification.',
        decoder_fallback_log=str(OUT/'decoder_fallback.jsonl'))
    atomic_json(OLD/'combined_summary.json',combined)
    evaluation.OUT=OLD/'consolidated';evaluation.OUT.mkdir(exist_ok=True);evaluation.report(combined)
    print(f"Completed {combined['completed_pairs']}/60 comparisons; all DarkDriving queries retained.",flush=True)


if __name__=='__main__':main()
