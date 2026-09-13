# -*- coding: utf-8 -*-
"""
.. module: pyAPES.utils.debug_capture
    :synopsis: Opt-in state capture for isolating Picard-loop non-convergence.

RESEARCH / DEBUG-ONLY MODULE. This lives on branch debug/tl-convergence-isolation
and is not intended to be merged into case_ranskalankorpi or main.

Enable by setting the PYAPES_CAPTURE_DIR environment variable to a directory
before running the driver. When unset, capture() is a no-op and there is no
behavioral or performance impact on normal runs.
"""

import os
import pickle
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

CAPTURE_DIR = os.environ.get('PYAPES_CAPTURE_DIR')
_counter = 0

# --- full-forcing recording (every timestep, not only non-convergent ones) ---
# Enable by setting PYAPES_FORCING_CAPTURE_DIR. Records accumulate in memory
# (one process, sequential timesteps -- see pyAPES_MLM.driver) and must be
# written out explicitly with flush_forcing_records() once the run is done.
FORCING_CAPTURE_DIR = os.environ.get('PYAPES_FORCING_CAPTURE_DIR')
_forcing_records: dict = {}


def enabled() -> bool:
    return CAPTURE_DIR is not None


def capture(tag: str, timestamp, payload: dict) -> None:
    """
    Pickle payload to CAPTURE_DIR/<tag>_<timestamp>_<counter>.pkl.
    No-op if PYAPES_CAPTURE_DIR is not set.

    Args:
        tag (str): identifies which non-convergence case this is
            (e.g. 'mlm_canopy', 'interception', 'planttype_sunlit')
        timestamp: the simulation datetime for this failure (parameters['date'])
        payload (dict): everything needed to replay the failing call in isolation
    """
    global _counter

    if not enabled():
        return

    outdir = Path(CAPTURE_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    _counter += 1
    ts_str = timestamp.strftime('%Y%m%dT%H%M') if hasattr(timestamp, 'strftime') else str(timestamp)
    fname = outdir / f"{tag}_{ts_str}_{_counter:05d}.pkl"

    with open(fname, 'wb') as f:
        pickle.dump(payload, f)

    logger.debug('debug_capture: wrote %s', fname)


def forcing_enabled() -> bool:
    return FORCING_CAPTURE_DIR is not None


def _snapshot_value(v):
    """Deep-ish copy of a forcing value: arrays/dicts/lists copied, scalars returned as-is."""
    if isinstance(v, np.ndarray):
        return v.copy()
    if isinstance(v, dict):
        return {k: _snapshot_value(vv) for k, vv in v.items()}
    if isinstance(v, (list, tuple)):
        return type(v)(_snapshot_value(vv) for vv in v)
    return v


def snapshot_dict(d: dict) -> dict:
    """Shallow-safe copy of a forcing dict, recursing into nested dicts/arrays."""
    return {k: _snapshot_value(v) for k, v in d.items()}


def record_forcing(tag: str, timestamp, outcome: str, data: dict) -> None:
    """
    Append one timestep's worth of forcing inputs for `tag` (e.g. 'mlm_canopy',
    'interception', 'planttype') to an in-memory list, tagged with the simulation
    timestamp and this timestep's Picard-loop outcome ('converged', 'tolerable',
    'switched_to_wma'). No-op if PYAPES_FORCING_CAPTURE_DIR is not set.

    Unlike capture(), this is meant to run on *every* timestep so forcing-input
    distributions can later be compared between converged and non-converged cases.
    Call flush_forcing_records() once the run is finished to write the results out.
    """
    if not forcing_enabled() or data is None:
        return

    _forcing_records.setdefault(tag, []).append({
        'timestamp': timestamp,
        'outcome': outcome,
        **data,
    })


def flush_forcing_records() -> None:
    """Write accumulated record_forcing() records to FORCING_CAPTURE_DIR/<tag>_forcing_samples.pkl."""
    if not forcing_enabled():
        return

    outdir = Path(FORCING_CAPTURE_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    for tag, records in _forcing_records.items():
        fname = outdir / f"{tag}_forcing_samples.pkl"
        with open(fname, 'wb') as f:
            pickle.dump(records, f)
        logger.debug('debug_capture: wrote %d forcing records to %s', len(records), fname)
