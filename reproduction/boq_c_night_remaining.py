"""Finish readable pending protocols without discarding the 48 completed results."""
import json
from pathlib import Path
from reproduction import boq_c_night_evaluation as evaluation
from reproduction.evaluate import atomic_json,sha256

OLD=Path('runs/boq_c_multi_night_epoch10_v2')
OUT=Path('runs/boq_c_multi_night_remaining_v1')
original_protocols=evaluation.protocols


def remaining_protocols():
    benches,missing=original_protocols()
    failure=json.loads((OLD/'darkdriving_decode_failure.json').read_text())
    missing['darkdriving_paired_diagnostic']=dict(reason='strict RGB decoder rejected incomplete/truncated JPEG; not silently filtered',
        failure_receipt=str(OLD/'darkdriving_decode_failure.json'),bad_images=failure['bad_images'])
    return [(b,s) for b,s in benches if b.name=='sped_mixed_illumination_diagnostic'],missing


def main():
    evaluation.OUT=OUT
    evaluation.protocols=remaining_protocols
    evaluation.main()
    old=json.loads((OLD/'summary.json').read_text())
    new=json.loads((OUT/'summary.json').read_text())
    if old['models']!=new['models']:raise RuntimeError('Model identities differ')
    combined=dict(old,status='completed_with_unavailable_protocol',
        protocols={**old['protocols'],**new['protocols']},results={**old['results'],**new['results']},
        unavailable=new['unavailable'],sources={str(p):sha256(p) for p in
           (OLD/'summary.json',OUT/'summary.json',Path(__file__))})
    combined['completed_pairs']=sum(len(v) for v in combined['results'].values())
    atomic_json(OLD/'combined_summary.json',combined)
    saved=evaluation.OUT
    evaluation.OUT=OLD/'consolidated';evaluation.OUT.mkdir(exist_ok=True)
    evaluation.report(combined)
    evaluation.OUT=saved
    print(f"Available completed comparisons: {combined['completed_pairs']}; DarkDriving unavailable, original images unchanged.",flush=True)


if __name__=='__main__':main()
