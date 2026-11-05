"""CPU protocol/integrity checks; no downloads or real-model inference."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from reproduction.benchmarks import (checked_positives, frame_positives, radius_positives)
from reproduction.evaluate import (exact_topk, feature_identity, json_sha256,
                                    open_feature_cache, recall_at_k, sha256,
                                    load_inference_checkpoint, cuda_reference_bank,
                                    resident_reference_budget)


class ProtocolTests(unittest.TestCase):
    def test_25m_inclusive(self):
        actual = radius_positives([[0, 0], [15, 20], [25.001, 0]], [[0, 0]])
        np.testing.assert_array_equal(actual[0], [0, 1])

    def test_nordland_raw_frame_tolerance_and_boundaries(self):
        rows = frame_positives(21, [0, 10, 20], 3)
        for row, expected in zip(rows, [range(4), range(7, 14), range(17, 21)]):
            np.testing.assert_array_equal(row, list(expected))

    def test_invalid_gt_is_not_silently_filtered(self):
        with self.assertRaises(ValueError):
            checked_positives([np.array([], dtype=np.int64)], 2, 1)
        with self.assertRaises(ValueError):
            checked_positives([np.array([2])], 2, 1)

    def test_exact_chunked_search_and_ties(self):
        rng = np.random.default_rng(2)
        refs = rng.normal(size=(31, 8)).astype('float32')
        refs[1] = refs[0]  # Cross-chunk exact ties must use ascending index.
        queries = np.vstack([refs[0], rng.normal(size=(6, 8))]).astype('float32')
        refs /= np.linalg.norm(refs, axis=1, keepdims=True)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)
        expected = np.argsort(-(queries @ refs.T), kind='stable', axis=1)[:, :10]
        for qc, rc in [(1, 1), (3, 7), (20, 100)]:
            predictions, scores = exact_topk(refs, queries, 10, qc, rc)
            np.testing.assert_array_equal(predictions, expected)
            np.testing.assert_allclose(scores, np.take_along_axis(queries @ refs.T, expected, axis=1), atol=2e-7)

    def test_recall_any_positive_and_denominator(self):
        result = recall_at_k(np.array([[2, 0, 1], [0, 2, 1]]),
                             [np.array([2]), np.array([1])], (1, 3))
        self.assertEqual(result, {'R@1': 50., 'R@3': 100.})


class CacheTests(unittest.TestCase):
    def test_interrupted_extraction_resumes_after_committed_batch(self):
        import torch
        from PIL import Image
        import reproduction.evaluate as evaluation

        class Toy(torch.nn.Module):
            calls = 0

            def forward(self, x):
                self.calls += 1
                return torch.nn.functional.normalize(x.mean((2, 3))[:, :2], dim=1)

        class Log:
            def log(self, *_args, **_kwargs):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            paths = [p / f'{i}.png' for i in range(3)]
            for i, path in enumerate(paths):
                Image.new('RGB', (8, 8), color=(20 + i, 60, 90)).save(path)
            config = dict(image_size=8, descriptor_dim=2)
            identity = feature_identity(paths, config, 'toy', 'query_student')
            model = Toy()
            evaluation.STOP = True
            try:
                with self.assertRaises(SystemExit):
                    evaluation.extract_features(model, paths, p / 'features.npy', identity,
                                                config, 'cpu', 1, 0, Log(), 'synthetic')
            finally:
                evaluation.STOP = False
            self.assertEqual(json.loads((p / 'features.progress.json').read_text())['rows'], 1)
            evaluation.extract_features(model, paths, p / 'features.npy', identity,
                                        config, 'cpu', 1, 0, Log(), 'synthetic')
            self.assertEqual(model.calls, 3)
            arr, rows = open_feature_cache(p / 'features.npy', identity)
            self.assertEqual(rows, 3)
            self.assertTrue(np.isfinite(arr).all())

    def test_numpy_scalar_checkpoint_remains_weights_only(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'best.pt'
            torch.save(dict(model={'weight': torch.tensor([1.])}, metrics={'R@1': np.float64(92.3)}), path)
            state = load_inference_checkpoint(path)
            self.assertEqual(state['metrics']['R@1'], 92.3)

    def test_resume_and_reject_changed_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            images = [p / 'a.jpg', p / 'b.jpg']
            for image in images:
                image.write_bytes(b'fake input, no decoder in this test')
            config = dict(image_size=322, descriptor_dim=3)
            identity = feature_identity(images, config, 'test-sha', 'query_student')
            cache = p / 'features.npy'
            arr, done = open_feature_cache(cache, identity)
            self.assertEqual(done, 0)
            arr[0] = [1, 0, 0]
            arr.flush()
            progress_path = cache.with_suffix('.progress.json')
            progress_path.write_text(json.dumps(dict(rows=1, total=2, identity_sha256=json_sha256(identity))))
            del arr
            arr, done = open_feature_cache(cache, identity)
            self.assertEqual(done, 1)
            np.testing.assert_array_equal(arr[0], [1, 0, 0])
            del arr
            changed = feature_identity(images[::-1], config, 'test-sha', 'query_student')
            with self.assertRaises(RuntimeError):
                open_feature_cache(cache, changed)

    def test_complete_cache_checksum_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            image = p / 'a.jpg'; image.write_bytes(b'x')
            identity = feature_identity([image], dict(image_size=322, descriptor_dim=2), 'sha', 'gallery_teacher')
            target = p / 'x.npy'
            arr, _ = open_feature_cache(target, identity)
            arr[0] = [1, 0]; arr.flush(); del arr
            target.with_suffix('.progress.json').write_text(json.dumps(dict(
                rows=1, total=1, identity_sha256=json_sha256(identity), feature_sha256=sha256(target))))
            arr, rows = open_feature_cache(target, identity)
            self.assertEqual(rows, 1)
            arr[0, 0] = 2; arr.flush(); del arr
            with self.assertRaises(RuntimeError):
                open_feature_cache(target, identity)


class ResidentBankTests(unittest.TestCase):
    """Mocks exercise CUDA decisions and upload chunks without opening a GPU context."""

    def test_budget_keeps_reserve_and_workspace(self):
        budget = resident_reference_budget((100, 32), 100 * 1024**3, 8, 20, 3.)
        self.assertEqual(budget['bank_bytes'], 100 * 32 * 4)
        self.assertEqual(budget['reserve_bytes'], 3 * 1024**3)
        self.assertEqual(budget['workspace_budget_bytes'], 8 * 32 * 4 + 8 * 20 * 4 + 20 * 32)
        required = budget['required_bytes']
        self.assertTrue(resident_reference_budget((100, 32), required, 8, 20, 3.)['fits'])
        self.assertFalse(resident_reference_budget((100, 32), required - 1, 8, 20, 3.)['fits'])
        with self.assertRaises(ValueError):
            resident_reference_budget((100, 32), required, 8, 20, 1.)

    def test_upload_copies_only_bounded_chunks_once(self):
        import torch
        references = np.arange(35, dtype='float32').reshape(7, 5)
        real_empty, real_array = torch.empty, np.array
        with patch('torch.cuda.mem_get_info', return_value=(20 * 1024**3, 48 * 1024**3)), \
             patch('torch.empty', side_effect=lambda shape, dtype, device: real_empty(shape, dtype=dtype, device='cpu')) as empty, \
             patch('reproduction.evaluate.np.array', wraps=real_array) as array:
            bank, info = cuda_reference_bank(references, 'cuda', 2, 3)
        self.assertEqual(info['mode'], 'cuda_resident')
        empty.assert_called_once()
        self.assertEqual(empty.call_args.kwargs['device'].type, 'cuda')
        self.assertEqual([call.args[0].shape[0] for call in array.call_args_list], [3, 3, 1])
        np.testing.assert_array_equal(bank.numpy(), references)

    def test_insufficient_memory_does_not_allocate(self):
        import torch
        with patch('torch.cuda.mem_get_info', return_value=(1024, 48 * 1024**3)), \
             patch('torch.empty') as empty:
            bank, info = cuda_reference_bank(np.zeros((2, 3), dtype='float32'), 'cuda', 2, 2)
        self.assertIsNone(bank)
        self.assertEqual(info['mode'], 'cuda_streamed')
        empty.assert_not_called()

    def test_oom_falls_back_but_nonfinite_errors_propagate(self):
        import torch
        refs = np.zeros((2, 3), dtype='float32')
        with patch('torch.cuda.mem_get_info', return_value=(20 * 1024**3, 48 * 1024**3)), \
             patch('torch.empty', side_effect=torch.cuda.OutOfMemoryError('synthetic allocation failure')), \
             patch('torch.cuda.empty_cache') as release:
            bank, info = cuda_reference_bank(refs, 'cuda', 2, 2)
        self.assertIsNone(bank)
        self.assertIn('OOM', info['fallback_reason'])
        release.assert_called_once()
        refs[1, 1] = np.nan
        real_empty = torch.empty
        with patch('torch.cuda.mem_get_info', return_value=(20 * 1024**3, 48 * 1024**3)), \
             patch('torch.empty', side_effect=lambda shape, dtype, device: real_empty(shape, dtype=dtype, device='cpu')):
            with self.assertRaises(FloatingPointError):
                cuda_reference_bank(refs, 'cuda', 2, 1)

    def test_cpu_path_does_not_probe_cuda(self):
        refs = np.eye(3, dtype='float32')
        info = {}
        with patch('torch.cuda.mem_get_info', side_effect=AssertionError('CPU must not query CUDA')):
            predictions, _ = exact_topk(refs, refs, 2, 2, 2, 'cpu', search_info=info)
        np.testing.assert_array_equal(predictions[:, 0], [0, 1, 2])
        self.assertEqual(info['mode'], 'cpu_streamed')


if __name__ == '__main__':
    unittest.main()
