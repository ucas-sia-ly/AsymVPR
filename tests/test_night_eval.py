"""CPU-only tests of night development splits, geography and mode restoration."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from reproduction.benchmarks import sha256
from reproduction.night_eval import (IndexedFeatures, _queries, load_svox_dev,
                                      synthetic_layout, _teacher_baseline)


def synthetic_row(night, source, label, city='city'):
    return dict(night_path=night, source_index=source, label=label,
                city=city, place_id=label, source_path=f'/day/{source}.jpg')


class SyntheticProtocolTests(unittest.TestCase):
    def test_all_snapshot_sources_are_distractors_and_same_place_positives(self):
        rows = [synthetic_row('/night/a.png', 8, 0),
                synthetic_row('/night/b.png', 12, 1),
                synthetic_row('/night/c.png', 17, 1)]
        layout = synthetic_layout(rows, rows[1:])
        self.assertEqual(layout['reference_indices'], [8, 12, 17])
        self.assertEqual(layout['query_indices'], [1, 2])
        for row in layout['positives']:
            np.testing.assert_array_equal(row, [1, 2])
        self.assertEqual(layout['num_heldout_places'], 1)

    def test_same_place_cannot_cross_augmentation_split(self):
        rows = [synthetic_row('/a', 0, 1), synthetic_row('/b', 1, 1)]
        with self.assertRaises(ValueError):
            synthetic_layout(rows, rows[:1])

    def test_conflicting_source_label_is_rejected(self):
        rows = [synthetic_row('/a', 0, 1), synthetic_row('/b', 0, 2)]
        with self.assertRaises(ValueError):
            synthetic_layout(rows, rows[1:])

    def test_indexed_mmap_view_preserves_order_and_bounded_slices(self):
        values = np.arange(30, dtype='float32').reshape(10, 3)
        indexed = IndexedFeatures(values, [8, 2, 9])
        self.assertEqual(indexed.shape, (3, 3))
        np.testing.assert_array_equal(indexed[:2], values[[8, 2]])
        self.assertEqual(indexed.dtype, np.float32)


class SvoxProtocolTests(unittest.TestCase):
    def fixture(self, root):
        root = Path(root)
        refs = []
        for name in ['@10@0@ref1.jpg', '@20@0@ref2.jpg']:
            relative = f'images/train/gallery/{name}'
            path = root / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'x')
            refs.append(dict(id=relative, path=relative))
        q = 'images/train/queries_night/@0@0@night.jpg'
        path = root / q; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'x')
        for split in ['val', 'test']:
            p = root / f'images/{split}/gallery/@100@100@excluded.jpg'
            p.parent.mkdir(parents=True); p.write_bytes(b'x')
        manifest = root / 'metadata/svox_train_weather.json'; manifest.parent.mkdir()
        body = dict(dataset='svox_train_weather', references=refs,
                    queries=[dict(id=q, path=q, positives=[r['id'] for r in refs])])
        manifest.write_text(json.dumps(body))
        protocol = dict(manifest_sha256=sha256(manifest), role='natural_weather_validation',
                        source='official SVOX train split', gallery_images=2,
                        queries_per_domain={'night': 1},
                        positive_rule='UTM distance <= 25 m to train gallery',
                        exclusion='drop queries within 50 m of any SVOX test/val gallery or query image')
        (manifest.parent / 'protocol.json').write_text(json.dumps(protocol))
        return dict(svox_dev_manifest=str(manifest), svox_root=str(root))

    def test_frozen_train_night_and_purge_are_audited(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self.fixture(tmp)
            data = load_svox_dev(config)
            self.assertEqual(len(data['references']), 2)
            self.assertEqual(len(data['queries']), 1)
            np.testing.assert_array_equal(data['positives'][0], [0, 1])
            self.assertGreaterEqual(data['definition']['minimum_distance_to_val_test_m'], 50)

    def test_query_within_50m_of_val_or_test_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self.fixture(tmp)
            old = Path(tmp) / 'images/test/gallery/@100@100@excluded.jpg'
            old.rename(old.with_name('@49@0@excluded.jpg'))
            with self.assertRaisesRegex(ValueError, '50 m'):
                load_svox_dev(config)

    def test_receipt_and_full_gallery_cannot_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self.fixture(tmp)
            (Path(tmp) / 'images/train/gallery/@300@0@extra.jpg').write_bytes(b'x')
            with self.assertRaisesRegex(ValueError, 'complete train gallery'):
                load_svox_dev(config)
        with tempfile.TemporaryDirectory() as tmp:
            config = self.fixture(tmp)
            manifest = Path(config['svox_dev_manifest'])
            manifest.write_text(manifest.read_text() + ' ')
            with self.assertRaisesRegex(ValueError, 'receipt/hash'):
                load_svox_dev(config)


class LiveInferenceTests(unittest.TestCase):
    def test_teacher_baseline_is_reused_only_with_intact_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {'output_dir': tmp}
            metrics = {'R@1': 50., 'R@5': 100.}
            with patch('reproduction.night_eval._search', return_value=(metrics, None)) as search:
                first = _teacher_baseline('test', None, None, None, {'protocol': 'fixed'}, config, 'cpu')
                second = _teacher_baseline('test', None, None, None, {'protocol': 'fixed'}, config, 'cpu')
                self.assertEqual(first, second)
                self.assertEqual(search.call_count, 1)
            path = Path(tmp) / 'evaluation/test/teacher_baseline.json'
            row = json.loads(path.read_text()); row['metrics_percent']['R@1'] = 75.
            path.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                _teacher_baseline('test', None, None, None, {'protocol': 'fixed'}, config, 'cpu')

    def test_modes_are_restored_even_when_query_model_raises(self):
        import torch
        from PIL import Image

        class FailingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.child = torch.nn.Identity()

            def forward(self, images):
                self.asserted_eval = not self.training and not self.child.training
                raise RuntimeError('synthetic test failure')

        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / 'query.png'
            Image.new('RGB', (8, 8)).save(image)
            model = FailingModel().train(); model.child.eval()
            before = [m.training for m in model.modules()]
            tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
            with self.assertRaisesRegex(RuntimeError, 'synthetic test failure'):
                _queries(model, [image], dict(image_size=8, descriptor_dim=2, num_workers=0), 'cpu')
            self.assertTrue(model.asserted_eval)
            self.assertEqual([m.training for m in model.modules()], before)
            self.assertEqual((torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32), tf32)


if __name__ == '__main__':
    unittest.main()
