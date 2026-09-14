# -*- coding: utf-8 -*-
"""
debug/tl-convergence-isolation: baseline run for debug_convergence_meeting_summary.ipynb.

Reproduces the pre-fix baseline for the 1.5.-30.9.2022 FI-Ran window: same forcing
and window as results/ran_22_fixed.nc, but on branch `case_ranskalankorpi` (i.e.
before any of the mlm_canopy/interception/planttype relaxation fixes on this branch)
with that branch's own canopy/soil parameters left untouched.

This script cannot be run from this branch (debug/tl-convergence-isolation) as-is --
it must run against the case_ranskalankorpi checkout. Reproduce with a git worktree
so the current working tree is untouched:

    git worktree add /tmp/pyAPES_baseline_case_ranskalankorpi case_ranskalankorpi
    cp debug/run_baseline_ran22_case_ranskalankorpi.py /tmp/pyAPES_baseline_case_ranskalankorpi/
    cd /tmp/pyAPES_baseline_case_ranskalankorpi
    PYTHONPATH=. python3 run_baseline_ran22_case_ranskalankorpi.py
    # then copy back the resulting .log (see below) for the notebook to read, e.g.:
    cp ran_22_baseline.log <repo>/logs/ran_22_baseline.log
    cd <repo> && git worktree remove /tmp/pyAPES_baseline_case_ranskalankorpi

RESEARCH / DEBUG-ONLY SCRIPT. Lives on branch debug/tl-convergence-isolation,
not intended to be merged.
"""

from dotenv import load_dotenv
from pyAPES.utils.iotools import read_forcing
from pyAPES.pyAPES_MLM import driver
from pyAPES.parameters.mlm_parameters_FI_Ran import gpara, cpara, spara

load_dotenv()

# case_ranskalankorpi's own gpara default window is 2022-06-01 to 2022-06-15 --
# override to match results/ran_22_fixed.nc's window for a direct comparison.
gpara['start_time'] = '2022-05-01'
gpara['end_time'] = '2022-09-30'

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

resultfile, _ = driver(parameters=params, create_ncf=True, result_file='ran_22_baseline.nc')

print('Result file:', resultfile)
print('start/end:', gpara['start_time'], gpara['end_time'])
