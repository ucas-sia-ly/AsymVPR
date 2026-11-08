"""Finalization tests use tiny CPU fixtures and a mocked evaluation subprocess."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from reproduction.benchmarks import Benchmark, DATASETS, json_sha256, sha256
from reproduction.evaluate import TRANSFORM, feature_identity
from reproduction.night_finalize import (compare_summaries, finalize, link_gallery_cache,
    _check_query_cache, tokyo_night_diagnostic, _mixed_reference, _receipt_identity_matches)


def write_cache(path, identity, rows=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    values = np.zeros((identity['paths']['count'], identity['descriptor_dim']), np.float32)
    values[:, 0] = 1
    np.save(path, values)
    path.with_suffix('.provenance.json').write_text(json.dumps(identity))
    path.with_suffix('.progress.json').write_text(json.dumps(dict(
        rows=len(values) if rows is None else rows, total=len(values),
        identity_sha256=json_sha256(identity), feature_sha256=sha256(path))))


class GalleryReuseTests(unittest.TestCase):
    def test_only_verified_gallery_files_are_linked_and_originals_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            identity = dict(paths={'count': 2}, descriptor_dim=3, model_sha256='teacher', role='gallery_teacher')
            source = p / 'original/gallery.npy'; write_cache(source, identity)
            originals = [source, source.with_suffix('.provenance.json'), source.with_suffix('.progress.json')]
            hashes = [sha256(path) for path in originals]
            row = {'gallery_cache': {'provenance': identity, 'feature_sha256': hashes[0]}}
            target = p / 'new/gallery.npy'
            link_gallery_cache(source, target, identity, row)
            self.assertTrue(all(path.is_symlink() for path in target.parent.iterdir()))
            self.assertEqual(len(list(target.parent.iterdir())), 3)
            self.assertEqual([sha256(path) for path in originals], hashes)
            self.assertFalse((target.parent / 'query.npy').exists())
            link_gallery_cache(source, target, identity, row)  # Idempotent identical reuse.

    def test_source_corruption_or_teacher_mismatch_is_not_reextracted(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            identity = dict(paths={'count': 2}, descriptor_dim=3, model_sha256='teacher')
            source = p / 'original/gallery.npy'; write_cache(source, identity)
            row = {'gallery_cache': {'provenance': identity, 'feature_sha256': sha256(source)}}
            with self.assertRaisesRegex(ValueError, 'provenance'):
                link_gallery_cache(source, p / 'new/gallery.npy', dict(identity, model_sha256='wrong'), row)
            arr = np.load(source, mmap_mode='r+'); arr[0, 0] = 2; arr.flush(); del arr
            with self.assertRaisesRegex(ValueError, 'checksum'):
                link_gallery_cache(source, p / 'new/gallery.npy', identity, row)

    def test_query_symlinks_and_wrong_checkpoint_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            identity = dict(paths={'count': 2}, descriptor_dim=3, model_sha256='new-best')
            source = p / 'old.npy'; write_cache(source, identity)
            link = p / 'query.npy'; link.symlink_to(source)
            with self.assertRaisesRegex(ValueError, 'never symlinks'):
                _check_query_cache(link, identity)
            with self.assertRaisesRegex(ValueError, 'another checkpoint'):
                _check_query_cache(source, dict(identity, model_sha256='different-best'))


class TokyoNightTests(unittest.TestCase):
    def test_csv_labels_and_official_mat_ids_map_reordered_prediction_rows(self):
        from scipy.io import savemat
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            official_refs = []
            for i in range(10):
                path = root / f'database_gsv_vga/g{i}.png'
                path.parent.mkdir(exist_ok=True); path.write_bytes(b'fixture')
                official_refs.append(path.resolve())
            queries = []
            for name, label in [('day.jpg', 'D'), ('night.jpg', 'N')]:
                path = root / 'queries' / name
                path.parent.mkdir(exist_ok=True); path.write_bytes(b'fixture'); queries.append(path.resolve())
                path.with_suffix('.csv').write_text(f'{name},0,0,x,x,{label},x,0,0\n')
            mat = root / 'official.mat'
            savemat(mat, dict(dbStruct=dict(whichSet='test', posDistThr=25., posDistSqThr=625.,
                dbImageFns=np.asarray([f'g{i}.jpg' for i in range(10)], dtype=object),
                qImageFns=np.asarray(['night.jpg', 'day.jpg'], dtype=object),
                utmDb=np.asarray([np.arange(10) * 100., np.zeros(10)]),
                utmQ=np.asarray([[0., 100.], [0., 0.]]))))
            source = root / 'source.json'
            source.write_text(json.dumps(dict(dataset='Tokyo247', split='test', mat_sha256=sha256(mat),
                night_label_rule='queries/<qImageFn stem>.csv, column 6 (zero-based index 5) == N',
                gallery_count=10, official_full_query_count=2, night_query_count=1, night_ids=['night.jpg'])))
            bench = Benchmark('tokyo247', root, official_refs[::-1], queries,
                              [np.asarray([8]), np.asarray([9])], {}, {})
            before = np.tile(np.arange(10), (2, 1))
            after = before.copy(); after[1] = np.roll(after[1], 1)
            np.save(root / 'before.npy', before); np.save(root / 'after.npy', after)
            result = tokyo_night_diagnostic(bench, root / 'before.npy', root / 'after.npy', source, mat)
            self.assertEqual(result['query_indices_in_full315'], [1])
            self.assertEqual(result['query_ids'], ['night.jpg'])
            self.assertEqual(result['initial_percent']['R@1'], 0.)
            self.assertEqual(result['final_percent']['R@1'], 100.)
            self.assertEqual(result['improved_at_1'], 1)
            self.assertEqual(result['regressed_at_1'], 0)
            json.dumps(result)  # Counts must serialize without NumPy scalar coercion.


class FinalizationTests(unittest.TestCase):
    def fixture(self, root):
        import torch
        root = Path(root)
        train = root / 'night/train'; train.mkdir(parents=True)
        inputs = root / 'night/inputs'; inputs.mkdir()
        (inputs / 'records.jsonl').write_text('{}\n')
        (inputs / 'summary.json').write_text(json.dumps(dict(manifest_sha256=sha256(inputs / 'records.jsonl'))))
        development = root / 'dev'; development.mkdir()
        (development / 'manifest.json').write_text('{}')
        (development / 'protocol.json').write_text('{}')
        teacher, initial_weights = root / 'teacher.pth', root / 'initial.pt'
        teacher.write_bytes(b'fixed teacher fixture'); initial_weights.write_bytes(b'initial weight fixture')
        config = dict(output_dir=str(train), input_dir=str(inputs), teacher_weights=str(teacher),
                      init_checkpoint=str(initial_weights), svox_dev_manifest=str(development / 'manifest.json'),
                      student_name='toy', descriptor_dim=3, proj_channels=3, num_queries=1,
                      num_layers=1, image_size=8, precision='float32', epochs=10, max_msls_drop_pp=1.)
        metrics = {'R@1': 55., 'R@5': 75., 'R@10': 100.}
        definition = dict(manifest_sha256=sha256(development / 'manifest.json'),
                          protocol_receipt_sha256=sha256(development / 'protocol.json'))
        baseline = dict(svox_dev=dict(metrics_percent=metrics, definition=definition))
        best_dev = dict(metrics_percent=dict(metrics, **{'R@1': 56.}), definition=definition)
        provenance = dict(teacher_sha256=sha256(teacher), image_size=8, descriptor_dim=3,
                          transform=TRANSFORM, precision='float32')
        torch.save(dict(model={'weight': torch.tensor([1.])}, config=config, epoch=2, step=6,
                        metrics={'svox_dev': best_dev}, provenance=provenance), train / 'best.pt')
        (train / 'baseline.json').write_text(json.dumps(baseline))
        (train / 'completed.json').write_text(json.dumps(dict(status='completed', epochs=10, config=config,
            best=dict(epoch=2, step=6, adapted=True, score=56.))))
        source_cache = root / 'original_cache'
        original = dict(checkpoint_sha256=sha256(initial_weights), teacher_sha256=sha256(teacher), datasets={})
        benches = []
        for name in DATASETS:
            images = root / 'images' / name; images.mkdir(parents=True)
            refs, queries = [], []
            for role, paths, count in [('r', refs, 10), ('q', queries, 2)]:
                for i in range(count):
                    p = images / f'{role}{i}.jpg'; p.write_bytes(b'fixture'); paths.append(p.resolve())
            protocol = {'protocol_variant': 'paper'} if name == 'nordland' else {'split': 'fixture'}
            bench = Benchmark(name, images, refs, queries, [np.array([0]), np.array([1])], protocol, {})
            benches.append(bench)
            identity = feature_identity(refs, config, sha256(teacher), 'gallery_teacher')
            gallery = source_cache / name / 'gallery.npy'; write_cache(gallery, identity)
            original['datasets'][name] = dict(**bench.describe(), metrics_percent=metrics,
                gallery_cache=dict(path=str(gallery), provenance=identity, feature_sha256=sha256(gallery)))
        summary = root / 'original/summary.json'; summary.parent.mkdir()
        for name in DATASETS:
            predicted = summary.parent / f'{name}_top10.npy'
            np.save(predicted, np.tile(np.arange(10), (2, 1)))
            original['datasets'][name]['predictions_sha256'] = sha256(predicted)
        summary.write_text(json.dumps(original))
        return config, summary, source_cache, benches, original

    def test_completed_run_evaluates_once_with_new_queries_and_sealed_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, source, source_cache, benches, initial = self.fixture(tmp)

            def fake_evaluate(command, *, cwd, env, check):
                self.assertEqual(command[command.index('--batch-size') + 1], '128')
                self.assertEqual(command[command.index('--workers') + 1], '4')
                cache, output = Path(command[command.index('--cache-dir') + 1]), Path(command[command.index('--output-dir') + 1])
                self.assertEqual(env['TRAIN_TERMINAL_RUN_DIR'], str(output / 'logs'))
                best_sha = sha256(command[command.index('--checkpoint') + 1])
                final = dict(checkpoint_sha256=best_sha, teacher_sha256=initial['teacher_sha256'],
                             precision='float32', tf32=False, datasets={})
                for b in benches:
                    identity = feature_identity(b.queries, config, best_sha, 'query_student')
                    query = cache / b.name / 'query.npy'; write_cache(query, identity)
                    row = dict(initial['datasets'][b.name])
                    row['metrics_percent'] = dict(row['metrics_percent'], **{'R@1': 56.})
                    row['query_cache'] = dict(path=str(query), provenance=identity, feature_sha256=sha256(query))
                    final['datasets'][b.name] = row
                    np.save(output / f'{b.name}_top10.npy', np.tile(np.arange(10), (2, 1)))
                (output / 'summary.json').write_text(json.dumps(final))

            with patch('reproduction.night_finalize.SOURCE_SUMMARY', source), \
                 patch('reproduction.night_finalize.SOURCE_CACHE', source_cache), \
                 patch('reproduction.night_finalize._benchmarks', return_value=benches), \
                 patch('reproduction.night_finalize.tokyo_night_diagnostic', return_value={'available': False}), \
                 patch('reproduction.night_finalize.subprocess.run', side_effect=fake_evaluate) as run:
                first = finalize(config)
                second = finalize(config)
                self.assertEqual(run.call_count, 1)
                self.assertEqual(first['identity'], second['identity'])
                self.assertTrue(first['comparison']['adapted'])
                self.assertEqual(first['comparison']['five_benchmark_comparison']['msls']['difference_percentage_points']['R@1'], 1.)
                report = Path(config['output_dir']).parent / 'NIGHT_REPORT.md'
                self.assertIn('pending_verification', report.read_text())
                report.write_text(report.read_text() + '\ncorrupted')
                with self.assertRaisesRegex(ValueError, 'output checksum'):
                    finalize(config)
                self.assertEqual(run.call_count, 1)

    def test_incomplete_training_never_starts_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, *_ = self.fixture(tmp)
            path = Path(config['output_dir']) / 'completed.json'
            row = json.loads(path.read_text()); row['status'] = 'running'; path.write_text(json.dumps(row))
            with patch('reproduction.night_finalize.subprocess.run') as run:
                with self.assertRaisesRegex(ValueError, 'completed training'):
                    finalize(config)
                run.assert_not_called()

    def test_protocol_changes_cannot_be_reported_as_a_model_delta(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, _, _, initial = self.fixture(tmp)
            after = json.loads(json.dumps(initial))
            after['datasets']['tokyo247']['ground_truth_sha256'] = 'different truth'
            with self.assertRaisesRegex(ValueError, 'protocols differ'):
                compare_summaries(initial, after)


    def test_legacy_receipt_reuses_old_implementation_without_rewriting_history(self):
        old = dict(config_sha256='same-config', implementation_sha256={
            'night_finalize.py': 'historic-code', 'evaluate.py': 'unchanged'})
        current = dict(config_sha256='same-config', implementation_sha256={
            'night_finalize.py': 'new-finalizer-code', 'evaluate.py': 'unchanged'})
        self.assertTrue(_receipt_identity_matches(old, current, {}))
        self.assertFalse(_receipt_identity_matches(old, current, {'query_domain': 'night_only'}))
        self.assertFalse(_receipt_identity_matches(old, dict(current, config_sha256='other-config'), {}))
        different_evaluator = dict(current, implementation_sha256={
            'night_finalize.py': 'new-finalizer-code', 'evaluate.py': 'changed-evaluator'})
        self.assertFalse(_receipt_identity_matches(old, different_evaluator, {}))


if __name__ == '__main__':
    unittest.main()
