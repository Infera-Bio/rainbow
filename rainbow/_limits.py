"""
An optional cap on the size of any single array rainbow sizes from a file.

The readers size arrays from values the file itself states: a scan count, a
point count, a start/end/step axis, a run's number of distinct m/z bins. A
damaged or hostile file can state values that ask for gigabytes from a few
kilobytes of input, and a process with a memory limit is killed before Python
can raise anything. With a cap set, every such allocation is checked first and
refused with :class:`AllocationLimitError`, which propagates out of the read
like any other parse failure.

No cap is set by default, so behaviour is unchanged unless one is asked for:
call :func:`set_max_array_bytes`, or set ``RAINBOW_MAX_ARRAY_BYTES`` (a whole
number of bytes) in the environment before rainbow is imported.

The cap applies to one array at a time. It bounds what a single crafted header
value can ask for; it is not a budget for the whole read.
"""

import math
import numbers
import os

import numpy as np


class AllocationLimitError(ValueError):
    """An array sized from file contents would exceed the configured cap."""


_ENV_VAR = "RAINBOW_MAX_ARRAY_BYTES"


def _validate(n):
    if n is None:
        return None
    if isinstance(n, bool) or not isinstance(n, numbers.Integral) or n < 0:
        raise ValueError(
            f"The array size cap must be a non-negative whole number of "
            f"bytes or None, got {n!r}.")
    return int(n)


def _from_environment():
    raw = os.environ.get(_ENV_VAR, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        value = raw
    try:
        return _validate(value)
    except ValueError:
        # A cap that was asked for and could not be read must not quietly
        # become no cap at all.
        raise ValueError(
            f"{_ENV_VAR} must be a non-negative whole number of bytes, "
            f"got {raw!r}.") from None


_max_array_bytes = _from_environment()


def set_max_array_bytes(n):
    """
    Sets the largest array, in bytes, rainbow will allocate from file contents.

    Args:
        n (int or None): The cap in bytes, or None for no cap (the default).
    """
    global _max_array_bytes
    _max_array_bytes = _validate(n)


def get_max_array_bytes():
    """Returns the current cap in bytes, or None if no cap is set."""
    return _max_array_bytes


def _refuse(what, detail, limit):
    raise AllocationLimitError(
        f"rainbow refused to allocate {detail} for {what}: the limit is "
        f"{limit:,} bytes (set by {_ENV_VAR} or rainbow.set_max_array_bytes).")


def check_array(shape, dtype, what):
    """
    Raises AllocationLimitError if an array of ``shape`` and ``dtype`` would
    exceed the cap. Does nothing when no cap is set.

    Args:
        shape (int or tuple): Element count, or the array's shape.
        dtype: Anything :class:`numpy.dtype` accepts.
        what (str): What the array holds, for the error message.
    """
    limit = _max_array_bytes
    if limit is None:
        return
    dims = shape if isinstance(shape, (tuple, list)) else (shape,)
    count = 1
    for dim in dims:
        # Python ints, so a product of file-stated sizes cannot wrap.
        count *= max(int(dim), 0)
    nbytes = count * np.dtype(dtype).itemsize
    if nbytes > limit:
        _refuse(what, f"{nbytes:,} bytes", limit)


def check_arange(start, stop, step, dtype, what):
    """
    :func:`check_array` for ``np.arange(start, stop, step)``, whose length is
    ``ceil((stop - start) / step)`` and so can be enormous for a tiny step.
    A zero or non-finite step, or a non-finite bound, is refused too: no
    finite length describes it.
    """
    limit = _max_array_bytes
    if limit is None:
        return
    try:
        ratio = (float(stop) - float(start)) / float(step)
    except (ZeroDivisionError, OverflowError):
        ratio = math.inf
    if math.isnan(ratio) or math.isinf(ratio):
        _refuse(what, f"an axis from {start!r} to {stop!r} in steps of "
                      f"{step!r}", limit)
    check_array(max(math.ceil(ratio), 0), dtype, what)
