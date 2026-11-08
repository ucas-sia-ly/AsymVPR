"""CPU checks for exact, bounded feature reads without feature-file mmap."""
import errno
import hashlib
import os

import numpy as np
import pytest

from reproduction.feature_io import BoundedFeatureRows


@pytest.fixture
def features(tmp_path):
    path = tmp_path / 'features.npy'
    # Include special FP32 bit patterns to detect conversion or normalization.
    values = np.arange(12 * 7, dtype=np.float32).reshape(12, 7)
    values[0, :4] = [np.nan, np.inf, -np.inf, -0.0]
    np.save(path, values)
    return path, values


def test_disordered_duplicates_are_byte_identical_and_independent(features, monkeypatch):
    path, values = features
    original = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('Reader must not call np.load'))
    with BoundedFeatureRows(path) as reader:
        rows = np.array([11, 0, 4, 3, 4, 1, 0], dtype=np.int64)
        actual = reader[rows]
        assert reader.shape == values.shape and reader.dtype == values.dtype
        assert len(reader) == len(values) and reader.ndim == 2
        assert actual.flags.owndata and actual.flags.c_contiguous
        assert actual.tobytes() == values[rows].tobytes()
        actual[:] = 123
        assert reader[rows].tobytes() == values[rows].tobytes()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original


def test_scalar_single_row_empty_and_slice(features):
    path, values = features
    with BoundedFeatureRows(path) as reader:
        assert reader[3].tobytes() == values[3].tobytes()
        assert reader[np.array([3])].shape == (1, 7)
        assert reader[np.array([], dtype=np.int64)].shape == (0, 7)
        assert reader[1:9:2].tobytes() == values[1:9:2].tobytes()
        assert reader[9:1:-2].tobytes() == values[9:1:-2].tobytes()


@pytest.mark.parametrize('index', [np.array([-1]), np.array([12]), -1, 12,
                                        np.array([2 ** 64 - 1], dtype=np.uint64)])
def test_out_of_range_is_rejected(features, index):
    with BoundedFeatureRows(features[0]) as reader:
        with pytest.raises(IndexError):
            reader[index]
        # An invalid caller index must not invalidate the reader.
        assert reader[np.array([1])].shape == (1, 7)


@pytest.mark.parametrize('index', [True, np.array([True]), np.array([1.]), np.array([[1]])])
def test_non_integer_vector_is_rejected(features, index):
    with BoundedFeatureRows(features[0]) as reader:
        with pytest.raises(TypeError):
            reader[index]


def test_short_preadv_is_retried_and_contiguous_rows_are_coalesced(features, monkeypatch):
    path, values = features
    real_preadv = os.preadv
    calls = []

    def short_read(fd, buffers, offset):
        calls.append(offset)
        return real_preadv(fd, [buffers[0][:13]], offset)

    monkeypatch.setattr(os, 'preadv', short_read)
    with BoundedFeatureRows(path) as reader:
        actual = reader[np.array([3, 1, 2, 3])]
    assert actual.tobytes() == values[[3, 1, 2, 3]].tobytes()
    # Rows 1-3 are fetched as one contiguous region, allowing short reads.
    assert len(calls) == (3 * 7 * 4 + 12) // 13
    assert np.diff(calls).tolist() == [13] * (len(calls) - 1)


def test_interrupted_read_retries(features, monkeypatch):
    real_preadv = os.preadv
    calls = 0

    def interrupted(fd, buffers, offset):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise InterruptedError()
        return real_preadv(fd, buffers, offset)

    monkeypatch.setattr(os, 'preadv', interrupted)
    with BoundedFeatureRows(features[0]) as reader:
        assert reader[[2]].tobytes() == features[1][[2]].tobytes()
    assert calls == 2


@pytest.mark.parametrize('failure', ['eof', 'oserror'])
def test_read_failure_closes_descriptor(features, monkeypatch, failure):
    reader = BoundedFeatureRows(features[0])
    fd = reader._fd

    def fail(*_):
        if failure == 'eof':
            return 0
        raise OSError(errno.EIO, 'synthetic read failure')

    monkeypatch.setattr(os, 'preadv', fail)
    with pytest.raises(EOFError if failure == 'eof' else OSError):
        reader[np.array([1])]
    with pytest.raises(OSError):
        os.fstat(fd)
    with pytest.raises(ValueError, match='closed'):
        reader[[1]]


def test_pread_fallback(features, monkeypatch):
    monkeypatch.delattr(os, 'preadv')
    with BoundedFeatureRows(features[0]) as reader:
        assert reader[[5, 1, 5]].tobytes() == features[1][[5, 1, 5]].tobytes()


def test_cache_advice_is_file_scoped_and_page_aligned(features, monkeypatch):
    calls = []
    monkeypatch.setattr(os, 'posix_fadvise', lambda *args: calls.append(args))
    with BoundedFeatureRows(features[0]) as reader:
        reader[[2, 3]]
        reader.trim_cache()
        fd = reader._fd
        page_size = reader._page_size
    assert calls[0] == (fd, 0, 0, os.POSIX_FADV_RANDOM)
    ranges = [call for call in calls if call[3] == os.POSIX_FADV_DONTNEED and call[2]]
    assert len(ranges) == 1
    assert ranges[0][1] % page_size == ranges[0][2] % page_size == 0
    assert calls[-1] == (fd, 0, 0, os.POSIX_FADV_DONTNEED)
    assert all(call[0] == fd for call in calls)


def test_read_buffers_are_chunk_bounded(features, monkeypatch):
    real_preadv = os.preadv
    sizes = []

    def record(fd, buffers, offset):
        sizes.append(len(buffers[0]))
        return real_preadv(fd, buffers, offset)

    monkeypatch.setattr(os, 'preadv', record)
    monkeypatch.setattr(BoundedFeatureRows, 'READ_CHUNK_BYTES', 2 * 7 * 4)
    with BoundedFeatureRows(features[0]) as reader:
        actual = reader[np.array([8, 0, 1, 1, 2, 3, 4, 4, 5])]
    assert max(sizes) <= 2 * 7 * 4
    assert actual.tobytes() == features[1][[8, 0, 1, 1, 2, 3, 4, 4, 5]].tobytes()


@pytest.mark.parametrize('values', [np.ones((2, 3), np.float64), np.ones(3, np.float32),
    np.ones((0, 3), np.float32), np.ones((2, 3), object),
    np.asfortranarray(np.ones((2, 3), np.float32)), np.ones((2, 3), '>f4')])
def test_invalid_payload_layout_is_rejected(tmp_path, values):
    path = tmp_path / 'invalid.npy'
    np.save(path, values)
    with pytest.raises(ValueError):
        BoundedFeatureRows(path)


def test_truncated_payload_is_rejected(features):
    path, _ = features
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match='Truncated'):
        BoundedFeatureRows(path)


def test_close_is_idempotent_and_output_survives(features):
    reader = BoundedFeatureRows(features[0])
    output = reader[[4, 1]]
    reader.close()
    reader.close()
    assert output.tobytes() == features[1][[4, 1]].tobytes()
    with pytest.raises(ValueError, match='closed'):
        reader[[1]]
    with pytest.raises(ValueError, match='closed'):
        reader.trim_cache()
