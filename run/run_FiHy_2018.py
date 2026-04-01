# %%
from pyAPES.utils.iotools import read_forcing
from pyAPES.pyAPES_MLM import driver

from pyAPES.parameters.SmearII_parameters import gpara, cpara, spara

from dotenv import load_dotenv
import os

load_dotenv()
# set pyAPES to path
pyAPES_main_folder = os.getenv('pyAPES_main_folder')

# Load forcing
forcing = read_forcing(
    forcing_file=gpara['forc_filename'],
    start_time=gpara['start_time'],
    end_time=gpara['end_time'],
    dt=gpara['dt'])

# Generate model parameters
params = {
    'general': gpara,
    'canopy': cpara,
    'soil': spara,
    'forcing': forcing}

params['general']['start_time'] = '2018-06-01'
params['general']['end_time'] = '2018-06-07'
resultfile, _ = driver(parameters=params,
                       create_ncf=True,
                       result_file= 'FiHy2018_run_test.nc')
