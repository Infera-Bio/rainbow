"""
The optional cap on arrays sized from file contents (rainbow._limits).

Each crafted input is built here in a few kilobytes and states sizes that,
uncapped, ask for gigabytes. With a cap set the read must raise
AllocationLimitError before allocating; with no cap rainbow must behave as it
always has.
"""
import os
import struct
import zipfile

import numpy as np
import pytest

import rainbow as rb
from rainbow import _limits
from rainbow._binning import bin_datapairs
from rainbow.agilent import chemstation, masshunter
from rainbow.waters import masslynx

GIB = 1 << 30


@pytest.fixture
def cap():
    """Sets a cap for one test and restores whatever was set before."""
    before = rb.get_max_array_bytes()
    yield rb.set_max_array_bytes
    rb.set_max_array_bytes(before)


def _ch_179(path, start_time, end_time, values=(1.0, 2.0)):
    """A minimal Agilent .ch in the 179 container: a 0x1800-byte header and
    one little-endian double per point."""
    header = bytearray(0x1800)
    header[0:4] = b"\x03179"
    header[0x11A:0x122] = struct.pack(">ff", start_time, end_time)
    header[0x127C:0x1284] = struct.pack(">d", 1.0)
    with open(path, "wb") as f:
        f.write(bytes(header) + struct.pack(f"<{len(values)}d", *values))
    return str(path)


# --- the limit itself -------------------------------------------------------

def test_default_is_no_cap(monkeypatch):
    monkeypatch.delenv("RAINBOW_MAX_ARRAY_BYTES", raising=False)
    assert _limits._from_environment() is None


def test_setter_validates(cap):
    for bad in (-1, 1.5, True, "1024"):
        with pytest.raises(ValueError):
            cap(bad)
    cap(0)
    assert rb.get_max_array_bytes() == 0
    cap(None)
    assert rb.get_max_array_bytes() is None


def test_environment_value(monkeypatch):
    monkeypatch.setenv("RAINBOW_MAX_ARRAY_BYTES", " 1073741824 ")
    assert _limits._from_environment() == GIB
    monkeypatch.setenv("RAINBOW_MAX_ARRAY_BYTES", "")
    assert _limits._from_environment() is None
    # A cap that was asked for but cannot be read is an error, not no cap.
    monkeypatch.setenv("RAINBOW_MAX_ARRAY_BYTES", "1GiB")
    with pytest.raises(ValueError, match="RAINBOW_MAX_ARRAY_BYTES"):
        _limits._from_environment()


def test_error_is_a_value_error_naming_the_request(cap):
    cap(1000)
    with pytest.raises(ValueError) as info:
        _limits.check_array((10, 1000), np.float64, "a test grid")
    assert isinstance(info.value, rb.AllocationLimitError)
    assert "a test grid" in str(info.value)
    assert "80,000 bytes" in str(info.value)
    _limits.check_array((10, 100), np.uint8, "a test grid")  # 1000 fits


def test_shape_product_cannot_wrap(cap):
    cap(GIB)
    # 2**32 * 2**32 wraps to 0 in int64; Python ints do not.
    with pytest.raises(rb.AllocationLimitError):
        _limits.check_array((2 ** 32, 2 ** 32), np.uint8, "x")


def test_arange_counts_like_numpy(cap):
    cap(80)
    _limits.check_arange(0, 10, 1, np.float64, "x")       # 10 values fit
    with pytest.raises(rb.AllocationLimitError):
        _limits.check_arange(0, 11, 1, np.float64, "x")   # 11 do not
    _limits.check_arange(10, 0, 1, np.float64, "x")       # empty
    _limits.check_arange(10, 0, -1, np.float64, "x")      # counts down
    for step in (0, -0.0, float("nan")):
        with pytest.raises(rb.AllocationLimitError):
            _limits.check_arange(0, 1, step, np.float64, "x")
    with pytest.raises(rb.AllocationLimitError):
        _limits.check_arange(-1e308, 1e308, 1e-308, np.float64, "x")


def test_no_cap_checks_nothing(cap):
    cap(None)
    _limits.check_array((2 ** 40, 2 ** 40), np.float64, "x")
    _limits.check_arange(0, 1, 0, np.float64, "x")


# --- Agilent .ch: a tiny time step -----------------------------------------

def test_ch_with_a_tiny_time_step_is_refused(tmp_path, cap):
    # The reported case: 6,160 bytes whose header runs 0 to 1e-12 over two
    # points, which uncapped asks np.arange for 8,000,000,040 bytes.
    run = tmp_path / "crafted.D"
    run.mkdir()
    path = _ch_179(run / "FID1A.ch", 0.0, 1e-12)
    assert os.path.getsize(path) == 6160
    cap(GIB)
    with pytest.raises(rb.AllocationLimitError,
                       match="retention time axis") as info:
        chemstation.parse_ch(path)
    assert "8,000,000,0" in str(info.value)
    # And through the public entry point: the error is not swallowed.
    with pytest.raises(rb.AllocationLimitError):
        rb.read(str(run))


def test_ch_reads_the_same_with_and_without_a_cap(tmp_path, cap):
    path = _ch_179(tmp_path / "FID1A.ch", 0.0, 60000.0)
    cap(None)
    uncapped = chemstation.parse_ch(path)
    cap(GIB)
    capped = chemstation.parse_ch(path)
    np.testing.assert_array_equal(uncapped.xlabels, capped.xlabels)
    np.testing.assert_array_equal(uncapped.data, capped.data)


# --- the shared binning step ------------------------------------------------

def _scans_over_labels(num_times, num_labels):
    """One pair per scan, cycling through num_labels distinct keys: a matrix
    of num_times x num_labels from num_times pairs."""
    keys = (np.arange(num_times) % num_labels).astype(np.float64)
    values = np.ones(num_times, dtype=np.int64)
    return keys, values, np.ones(num_times, dtype=np.int64)


def test_binning_grid_is_refused(cap):
    # The reported Waters shape: 500,000 one-pair scans over 3,277 labels is a
    # 13,108,000,000-byte int64 grid from 500,000 pairs.
    keys, values, counts = _scans_over_labels(500_000, 3277)
    cap(GIB)
    with pytest.raises(rb.AllocationLimitError, match="13,108,000,000"):
        bin_datapairs(keys, values, counts, 1.0)


def test_binning_pair_counts_are_not_trusted(cap):
    # An index that claims far more pairs than the data holds.
    keys = np.array([1.0, 2.0])
    counts = np.array([2, 2 ** 40], dtype=np.int64)
    cap(GIB)
    with pytest.raises(rb.AllocationLimitError, match="row index"):
        bin_datapairs(keys, np.ones(2), counts, 1.0)


def test_binning_takes_the_sort_path_rather_than_refuse(cap):
    # A span whose dense histogram is over the cap still bins, by sorting.
    keys = np.array([0.0, 1_000_000.0])
    counts = np.array([1, 1])
    cap(None)
    expected = bin_datapairs(keys, np.ones(2, np.int64), counts, 1.0)
    cap(1_000_000)
    got = bin_datapairs(keys, np.ones(2, np.int64), counts, 1.0)
    np.testing.assert_array_equal(expected[0], got[0])
    np.testing.assert_array_equal(expected[1], got[1])


def test_binning_is_unchanged_under_a_cap_it_fits(cap):
    keys, values, counts = _scans_over_labels(1000, 37)
    cap(None)
    expected = bin_datapairs(keys, values, counts, 0.5, display_precision=1)
    cap(GIB)
    got = bin_datapairs(keys, values, counts, 0.5, display_precision=1)
    np.testing.assert_array_equal(expected[0], got[0])
    np.testing.assert_array_equal(expected[1], got[1])


# --- Waters .raw: real bytes through the reader -----------------------------

def _waters_raw(directory, num_times, num_labels):
    """A .raw with one 6-byte-format function: one pair per scan, the key
    cycling over num_labels whole m/z values, plus a 416-byte _FUNCTNS.INF."""
    raw = directory / "crafted.raw"
    raw.mkdir()
    keys = np.arange(num_times, dtype=np.uint32) % num_labels + 100
    pairs = np.zeros(num_times, dtype=[("val", "<i2"), ("raw", "<u4")])
    pairs["val"] = 1
    # key = (raw >> 9) * 2 ** (((raw & 0x1F0) >> 4) - 23): exponent 23 is 1.
    pairs["raw"] = (keys << 9) | (23 << 4)
    (raw / "_FUNC001.DAT").write_bytes(pairs.tobytes())
    idx = np.zeros(num_times, dtype=[("offset", "<u4"), ("count", "<u4"),
                                     ("pad", "V4"), ("time", "<f4"),
                                     ("tail", "V6")])
    idx["offset"] = np.arange(num_times) * 6
    idx["count"] = 1
    idx["time"] = np.arange(num_times) / 60.0
    (raw / "_FUNC001.IDX").write_bytes(idx.tobytes())
    (raw / "_FUNCTNS.INF").write_bytes(bytes(416))
    return raw


def test_waters_function_grid_is_refused(tmp_path, cap):
    raw = _waters_raw(tmp_path, 20_000, 500)        # 80,000,000-byte grid
    cap(None)
    datafile = masslynx.parse_spectrum(str(raw))[0]
    assert datafile.data.shape == (20_000, 500)
    cap(64 << 20)
    with pytest.raises(rb.AllocationLimitError, match="80,000,000"):
        masslynx.parse_spectrum(str(raw))


# --- Agilent MassHunter and OpenLab ----------------------------------------

def test_profile_flight_time_axis_is_refused(cap):
    # A delta_mz of 1e-12 with num_mz 2 is an axis of about a billion values.
    cap(GIB)
    with pytest.raises(rb.AllocationLimitError, match="flight-time"):
        masshunter._build_per_scan_profiles(
            np.zeros(1), [np.zeros(2, np.uint32)], [(2, 0.0, 1e-12)],
            np.zeros((1, 10)), [None], {})


def test_mass_calibration_polynomial_is_refused(cap):
    cap(1024)
    with pytest.raises(rb.AllocationLimitError, match="calibration"):
        masshunter.calibrate_mz(np.ones(3), np.ones(10), 1 << 10_000)


def test_dx_member_is_checked_before_it_is_inflated(tmp_path, cap):
    # 64 MiB of zeros deflates to a few dozen kilobytes.
    path = tmp_path / "crafted.dx"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("injection.acmd", b"<x/>")
        archive.writestr("00000000.CH", bytes(64 << 20))
    assert os.path.getsize(path) < 1 << 20
    cap(1 << 20)
    with pytest.raises(rb.AllocationLimitError, match="00000000.CH"):
        rb.read(str(path))
