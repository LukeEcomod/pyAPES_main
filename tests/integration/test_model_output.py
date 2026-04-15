import json
import numpy as np
import pytest
import xarray as xr
from pathlib import Path

RESULT_FILE = Path('results/CI_test.nc')
EXPECTED_FILE = Path(__file__).parent / 'expected_values.json'
RTOL = 0.1
ATOL = 5e-9

with open(EXPECTED_FILE) as _f:
    _EXPECTED = json.load(_f)

_PARAMS = [
    (var, stat)
    for var in _EXPECTED
    for stat in ('min', 'max', 'median')
]


@pytest.fixture(scope='module')
def ds():
    assert RESULT_FILE.exists(), f"Result file not found: {RESULT_FILE}"
    data = xr.open_dataset(RESULT_FILE)
    yield data
    data.close()


def _stats(data, var):
    vals = data[var].values.flatten()
    vals = vals[~np.isnan(vals)]
    return {
        'min': float(vals.min()),
        'max': float(vals.max()),
        'median': float(np.median(vals)),
    }


@pytest.mark.parametrize('var,stat', _PARAMS)
def test_stat_is_float(ds, var, stat):
    assert isinstance(_stats(ds, var)[stat], float)


@pytest.mark.parametrize('var,stat', _PARAMS)
def test_stat_in_range(ds, var, stat):
    actual = _stats(ds, var)[stat]
    ref = _EXPECTED[var][stat]
    assert np.isclose(actual, ref, rtol=RTOL, atol=ATOL), (
        f"{var}.{stat}: actual={actual:.6g}, expected={ref:.6g} "
        f"(allowed rtol={RTOL}, atol={ATOL})"
    )
