# -*- coding: utf-8 -*-
"""
debug/tl-convergence-isolation: capture-run script for case 1
(mlm_canopy.py Picard-loop non-convergence: T/H2O/CO2/Tl/Ts profiles).

Runs a short, representative FI-Ran window and dumps a self-contained
snapshot (CanopyModel state + forcing/parameters + per-iteration
trajectory) to Examples/debug_captures/case1/ every time the model hits
the same non-convergence branches that were logged in
Examples/logs/ran_22_k7.log ("Maximum iterations reached but error
tolerable" / "Switched to WMA assumption").

Run from the repository root:
    PYAPES_CAPTURE_DIR=Examples/debug_captures/case1 python Examples/capture_case1.py

RESEARCH / DEBUG-ONLY SCRIPT. Lives on branch debug/tl-convergence-isolation,
not intended to be merged.
"""

import os

# must be set before importing pyAPES.canopy.mlm_canopy, which reads it at import time
os.environ.setdefault('PYAPES_CAPTURE_DIR', 'Examples/debug_captures/case1')

from dotenv import load_dotenv
from pyAPES.utils.iotools import read_forcing
from pyAPES.pyAPES_MLM import driver
from pyAPES.parameters.mlm_parameters_FI_Ran import gpara, cpara, spara

load_dotenv()
pyAPES_main_folder = os.getenv('pyAPES_main_folder')

# short window: known from ran_22_k7.log to contain both night-time (case 1)
# non-convergence and the daytime/morning cases used to isolate later
gpara['start_time'] = '2022-06-20'
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
                        result_file='capture_case1.nc')

print('Capture directory:', os.environ['PYAPES_CAPTURE_DIR'])
