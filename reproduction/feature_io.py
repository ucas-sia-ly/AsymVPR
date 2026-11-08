"""Bounded, read-only FP32 descriptor reads without a resident file mapping.

Only the requested output and a small contiguous read buffer occupy host RAM.
File-cache advice is confined to this descriptor file; it is best effort and
does not evict unrelated applications' caches or change any feature bytes.
"""
from __future__ import annotations

import errno
import os
from pathlib import Path

import numpy as np


class BoundedFeatureRows:
    """Read rows of a C-contiguous, two-dimensional FP32 ``.npy`` file.

    Row requests retain their order and duplicates. Negative indices are
    rejected deliberately. Outputs always own their memory and do not retain
    an open file or mmap. The reader itself must not be shared across a close
    and a concurrent read; independent readers can safely use the same file.
    """

    READ_CHUNK_BYTES = 8 * 1024 ** 2

    def __init__(self, path):
        self.path = Path(path)
        self._fd = None
        self._page_size = os.sysconf('SC_PAGE_SIZE')
        fd = os.open(self.path, os.O_RDONLY)
        try:
            with os.fdopen(fd, 'rb', closefd=False) as header:
                version = np.lib.format.read_magic(header)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(header)
                elif version == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(header)
                elif version == (3, 0):
                    # Version 3 differs in header encoding, not payload layout.
                    shape, fortran, dtype = np.lib.format._read_array_header(header, version)
                else:
                    raise ValueError(f'Unsupported NumPy descriptor version: {version}')
                offset = header.tell()
            if (len(shape) != 2 or any(not isinstance(x, int) or x <= 0 for x in shape)
                    or fortran or dtype != np.dtype(np.float32) or dtype.hasobject):
                raise ValueError('Descriptor file must be nonempty, C-contiguous, native FP32 N x D')
            size = os.fstat(fd).st_size
            if size < offset + int(shape[0]) * int(shape[1]) * dtype.itemsize:
                raise ValueError('Truncated NumPy descriptor payload')
            self.shape, self.dtype, self.ndim = tuple(shape), dtype, 2
            self._data_offset = offset
            self._row_bytes = int(shape[1]) * dtype.itemsize
            self._fd = fd
            self._advise(0, 0, 'POSIX_FADV_RANDOM')
        except BaseException:
            self._fd = None
            os.close(fd)
            raise

    def __len__(self):
        return self.shape[0]

    def _require_open(self):
        if self._fd is None:
            raise ValueError('Descriptor reader is closed')
        return self._fd

    def _advise(self, offset, length, name):
        """Ignore unsupported advice, but preserve actual descriptor errors."""
        if not hasattr(os, 'posix_fadvise') or not hasattr(os, name):
            return
        try:
            os.posix_fadvise(self._require_open(), offset, length, getattr(os, name))
        except OSError as error:
            if error.errno not in {errno.EINVAL, errno.ENOSYS, errno.ENOTSUP,
                                   getattr(errno, 'EOPNOTSUPP', errno.ENOTSUP)}:
                raise

    def _trim_range(self, offset, length):
        start = offset // self._page_size * self._page_size
        stop = (offset + length + self._page_size - 1) // self._page_size * self._page_size
        self._advise(start, stop - start, 'POSIX_FADV_DONTNEED')

    def trim_cache(self):
        """Suggest releasing cached pages of this file, for phase boundaries."""
        self._require_open()
        self._advise(0, 0, 'POSIX_FADV_DONTNEED')

    def _read_block(self, first, count):
        values = np.empty((count, self.shape[1]), dtype=self.dtype)
        view = memoryview(values).cast('B')
        offset = self._data_offset + int(first) * self._row_bytes
        done = 0
        try:
            while done < len(view):
                try:
                    if hasattr(os, 'preadv'):
                        received = os.preadv(self._require_open(), [view[done:]], offset + done)
                    else:
                        data = os.pread(self._require_open(), len(view) - done, offset + done)
                        received = len(data)
                        view[done:done + received] = data
                except InterruptedError:
                    continue
                if received <= 0:
                    raise EOFError(f'Short descriptor read at byte {offset + done}: {self.path}')
                done += received
            self._trim_range(offset, len(view))
            return values
        except BaseException:
            self.close()
            raise

    def __getitem__(self, key):
        self._require_open()
        scalar = isinstance(key, (int, np.integer)) and not isinstance(key, (bool, np.bool_))
        if isinstance(key, slice):
            if (key.start is not None and key.start < 0) or (key.stop is not None and key.stop < 0):
                raise IndexError('Negative descriptor indices are unsupported')
            rows = np.arange(*key.indices(len(self)), dtype=np.int64)
        elif scalar:
            rows = np.asarray([key])
        else:
            rows = np.asarray(key)
        if rows.ndim != 1 or rows.dtype.kind not in 'iu':
            raise TypeError('Descriptor rows must be a one-dimensional integer index')
        if rows.size and (np.any(rows < 0) or np.any(rows >= len(self))):
            raise IndexError('Descriptor row outside feature bank')
        output = np.empty((len(rows), self.shape[1]), dtype=self.dtype)
        if not len(rows):
            return output
        order = np.argsort(rows, kind='stable')
        sorted_rows = rows[order]
        unique = np.unique(sorted_rows)
        boundaries = np.flatnonzero(np.diff(unique) > 1) + 1
        runs = np.split(unique, boundaries)
        block_rows = max(1, self.READ_CHUNK_BYTES // self._row_bytes)
        for run in runs:
            for begin in range(int(run[0]), int(run[-1]) + 1, block_rows):
                end = min(begin + block_rows, int(run[-1]) + 1)
                block = self._read_block(begin, end - begin)
                first = int(np.searchsorted(sorted_rows, begin, side='left'))
                last = int(np.searchsorted(sorted_rows, end, side='left'))
                # Bound temporary reordering copies even for many duplicate rows.
                for destination in range(first, last, block_rows):
                    stop = min(destination + block_rows, last)
                    output[order[destination:stop]] = block[sorted_rows[destination:stop] - begin]
        return output[0].copy() if scalar else output

    def close(self):
        if self._fd is not None:
            fd, self._fd = self._fd, None
            os.close(fd)

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        self.close()
