# -*- coding: utf-8 -*-
"""
debug/tl-convergence-isolation: capture-run script for forcing-distribution
exploration (Examples/debug_forcing_exploration.ipynb).

Unlike capture_case1.py (which only snapshots timesteps where the Picard loop
fails to converge), this records the forcing inputs going into mlm_canopy,
interception and planttype for EVERY timestep of the run, tagged with that
timestep's outcome ('converged' / 'tolerable' / 'switched_to_wma'). This lets
the exploration notebook compare input distributions between converged and
non-converged timesteps, matched by time of day.

Run from the repository root:
    PYTHONPATH=. .venv/bin/python Examples/capture_forcing_exploration.py

RESEARCH / DEBUG-ONLY SCRIPT. Lives on branch debug/tl-convergence-isolation,
not intended to be merged.
"""

import os

# must be set before importing pyAPES.canopy.mlm_canopy, which reads it at import time
os.environ.setdefault('PYAPES_FORCING_CAPTURE_DIR', 'Examples/debug_captures/forcing_exploration')

from dotenv import load_dotenv
from pyAPES.utils.iotools import read_forcing
from pyAPES.utils import debug_capture
from pyAPES.pyAPES_MLM import driver
from pyAPES.parameters.mlm_parameters_FI_Ran import gpara, cpara, spara

load_dotenv()
pyAPES_main_folder = os.getenv('pyAPES_main_folder')

# window requested for forcing-distribution exploration
gpara['start_time'] = '2022-06-01'
gpara['end_time'] = '2022-06-30'

forcing = read_forcing(
    forcing_file=gpara['forc_filename'],
    start_time=gpara['start_time'],
    end_time=gpara['end_time'],
    dt=gpara['dt'])

params = {
    'general': gpara,
    'canopy': cpara,
    'soil': spara,
    'forcing': forcing}

resultfile, _ = driver(parameters=params,
                        create_ncf=True,
                        result_file='capture_forcing_exploration.nc')

# forcing records accumulate in memory during the run (single process, sequential
# timesteps) -- write them out now that the run is finished.
debug_capture.flush_forcing_records()

print('Forcing capture directory:', os.environ['PYAPES_FORCING_CAPTURE_DIR'])
