# -*- coding: utf-8 -*-
"""
debug/tl-convergence-isolation: full-season validation run for
debug_convergence_meeting_summary.ipynb.

Unlike capture_case1.py / capture_forcing_exploration.py (both restricted to a
June 2022 window for fast iteration), this runs the FI-Ran default simulation
window (mlm_parameters_FI_Ran.gpara: 1.5.-30.9.2022, 7297 timesteps) against
the current, fixed source code, to check that the three convergence fixes
(mlm_canopy/interception/planttype) hold over a full growing season and not
just the June sample they were validated against.

No debug_capture instrumentation is enabled here (plain run) -- convergence
outcomes are read from the resulting DEBUG log, not from captured forcing.

Run from the repository root:
    PYTHONPATH=. python3 debug/run_ran22_full_season.py

RESEARCH / DEBUG-ONLY SCRIPT. Lives on branch debug/tl-convergence-isolation,
not intended to be merged.
"""

from dotenv import load_dotenv
from pyAPES.utils.iotools import read_forcing
from pyAPES.pyAPES_MLM import driver
from pyAPES.parameters.mlm_parameters_FI_Ran import gpara, cpara, spara

load_dotenv()

# gpara['start_time']/['end_time'] left at their FI-Ran defaults (2022-05-01 to
# 2022-09-30) -- the same window as the earlier results/ran_22_k7.nc run.
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

resultfile, _ = driver(parameters=params, create_ncf=True, result_file='ran_22_fixed.nc')

print('Result file:', resultfile)
print('start/end:', gpara['start_time'], gpara['end_time'])
