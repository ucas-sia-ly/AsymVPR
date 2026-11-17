"""Re-audit surviving frozen pairs, preserving splits and teacher row mapping."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from reproduction.benchmarks import json_sha256
from reproduction.evaluate import (atomic_json, feature_identity, open_feature_cache, sha256)
from reproduction.night_data import (_audit_pair, _validate_frozen, _write_rows,
                                     load_records, NightDataIntegrityError)

ROOT = Path('runs/augmentation_screen_remaining_v2')


def audit_remaining(rows, workers=4):
    kept, excluded, mapping = [], [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (original, (audited, error)) in enumerate(zip(rows, pool.map(_audit_pair, rows))):
            if error is not None:
                excluded.append(dict(error, parent_teacher_index=i, split=original['split']))
            else:
                if audited['split'] != original['split']:
                    raise NightDataIntegrityError('Subset must preserve the original split')
                audited['parent_teacher_index'] = i
                kept.append(audited)
                mapping.append(i)
            if (i+1) % 2000 == 0:
                print(f'Remaining-pair SHA/decode audit: {i+1}/{len(rows)}; kept={len(kept)} excluded={len(excluded)}', flush=True)
    if not kept or {r['split'] for r in kept} != {'train','val'}:
        raise NightDataIntegrityError('Remaining snapshot requires nonempty train and val')
    train = {r['label'] for r in kept if r['split']=='train'}
    val = {r['label'] for r in kept if r['split']=='val'}
    if train & val:
        raise NightDataIntegrityError('Remaining snapshot has cross-split place labels')
    return kept, excluded, np.asarray(mapping, dtype=np.int64)


def parent_feature_identity(rows, config):
    # Reconstruct the original audited path identity without requiring deleted files.
    names = [str(Path(r['night_path']).resolve()) for r in rows]
    stats = [[name, r['night_size'], r['night_mtime_ns']] for name,r in zip(names,rows)]
    from reproduction.evaluate import TRANSFORM
    return dict(schema=1, role='night_teacher', model_sha256=sha256(config['teacher_weights']),
                image_size=config['image_size'], descriptor_dim=config['descriptor_dim'],
                transform=TRANSFORM, precision='float32', tf32=False,
                paths=dict(count=len(rows), ordered_paths_sha256=json_sha256(names),
                           input_file_stats_sha256=json_sha256(stats)))


def build_remaining(root=ROOT, parent_config='configs/boq_b2_night_v1.json'):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root/'.inputs.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        config = json.loads(Path(parent_config).read_text())
        parent = Path(config['input_dir'])
        parent_summary = json.loads((parent/'summary.json').read_text())
        _validate_frozen(config, parent, parent_summary)
        new_config = dict(config, input_dir=str(root/'inputs'),
                          night_teacher_cache=str(root/'cache/night_teacher.npy'))
        destination = root/'inputs'
        if (destination/'summary.json').exists():
            rows = load_records(new_config,'all')
            identity = feature_identity([Path(r['night_path']) for r in rows], new_config,
                                        sha256(config['teacher_weights']), 'night_teacher')
            values, done = open_feature_cache(new_config['night_teacher_cache'], identity)
            if done != len(rows):raise NightDataIntegrityError('Incomplete subset teacher cache')
            del values
            return json.loads((destination/'summary.json').read_text())
        if destination.exists():
            raise NightDataIntegrityError('Incomplete remaining input directory; preserve for inspection')
        rows = [json.loads(line) for line in (parent/'records.jsonl').open()]
        started = time.monotonic()
        kept, excluded, mapping = audit_remaining(rows)
        old_identity = parent_feature_identity(rows, config)
        old_values, done = open_feature_cache(config['night_teacher_cache'],old_identity)
        if done != len(rows):raise NightDataIntegrityError('Parent night teacher cache incomplete')
        old_progress = json.loads(Path(config['night_teacher_cache']).with_suffix('.progress.json').read_text())
        # Cache remains fixed teacher output: select exact old rows, no new inference.
        new_identity = feature_identity([Path(r['night_path']) for r in kept],new_config,
                                        sha256(config['teacher_weights']),'night_teacher')
        cache = root/'cache';cache.mkdir(exist_ok=True)
        target = Path(new_config['night_teacher_cache'])
        if target.exists():raise NightDataIntegrityError('Uncommitted subset teacher file exists')
        with tempfile.TemporaryDirectory(prefix='.remaining-',dir=root) as temporary:
            temporary = Path(temporary)
            feature_path = temporary/'night_teacher.npy'
            values = np.lib.format.open_memmap(feature_path,mode='w+',dtype=np.float32,
                                              shape=(len(kept),config['descriptor_dim']))
            for start in range(0,len(kept),128):
                chunk = np.asarray(old_values[mapping[start:start+128]])
                if not np.isfinite(chunk).all() or not np.allclose(np.linalg.norm(chunk,axis=1),1.,atol=5e-4,rtol=5e-4):
                    raise NightDataIntegrityError('Nonfinite/non-unit selected teacher rows')
                values[start:start+len(chunk)] = chunk
            values.flush();del values,old_values
            feature_sha = sha256(feature_path)
            stage = temporary/'inputs';stage.mkdir()
            (stage/'metadata').mkdir()
            atomic_json(stage/'metadata/index.json', [])
            _write_rows(stage/'records.jsonl',kept)
            _write_rows(stage/'excluded.jsonl',excluded)
            atomic_json(stage/'input_config.json',parent_summary['input_config'])
            np.save(stage/'parent_teacher_indices.npy',mapping)
            original_val = sorted({r['label'] for r in rows if r['split']=='val'})
            derivation = dict(schema=1, operation='user_authorized_remaining_images_subset',
                parent_input_dir=str(parent.resolve()), parent_records_sha256=sha256(parent/'records.jsonl'),
                parent_summary_sha256=sha256(parent/'summary.json'),
                parent_teacher_cache=str(Path(config['night_teacher_cache']).resolve()),
                parent_teacher_sha256=old_progress['feature_sha256'],
                selected_teacher_sha256=feature_sha, split_policy='preserve every original row split',
                replay_excluded_place_ids=original_val, parent_total=len(rows),
                excluded_reasons=dict(Counter(r['reason'] for r in excluded)),
                implementation_sha256=sha256(__file__), note='No new images, labels or teacher descriptors generated')
            atomic_json(stage/'derivation.json',derivation)
            files=['records.jsonl','excluded.jsonl','input_config.json','metadata/index.json',
                   'parent_teacher_indices.npy','derivation.json']
            outputs={name:sha256(stage/name) for name in files}
            counts=Counter(r['split'] for r in kept)
            places={split:len({r['label'] for r in kept if r['split']==split}) for split in ['train','val']}
            summary=dict(parent_summary, input_dir=str(destination.resolve()),
                records_path=str((destination/'records.jsonl').resolve()),total=len(kept),
                train=counts['train'],val=counts['val'],
                counts=dict(included=len(kept),train=counts['train'],val=counts['val'],excluded=len(excluded),receipt_rows=len(rows)),
                places=dict(total=sum(places.values()),**places),
                review_status_counts=dict(Counter(r.get('status','unspecified') for r in kept)),
                excluded_reasons=derivation['excluded_reasons'],output_sha256=outputs,
                manifest_sha256=outputs['records.jsonl'],metadata_files=0,metadata_bytes=0,
                created_unix=time.time(),audit_seconds=time.monotonic()-started,
                derivation=derivation)
            atomic_json(stage/'summary.json',summary)
            # Publish cache sidecars then snapshot; readers only use committed snapshot.
            feature_path.replace(target)
            atomic_json(target.with_suffix('.provenance.json'),new_identity)
            atomic_json(target.with_suffix('.progress.json'),dict(rows=len(kept),total=len(kept),
                        identity_sha256=json_sha256(new_identity),feature_sha256=feature_sha))
            stage.replace(destination)
        print(json.dumps(dict(total=len(kept),train=counts['train'],val=counts['val'],excluded=len(excluded),
                              input_dir=str(destination)),ensure_ascii=False),flush=True)
        return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=str(ROOT))
    args=parser.parse_args()
    build_remaining(args.root)


if __name__=='__main__':main()
