# %%
from pyAPES.utils.iotools import read_forcing
from pyAPES.pyAPES_MLM import driver

from pyAPES.parameters.SmearII_parameters import gpara, cpara, spara

from dotenv import load_dotenv
import os

load_dotenv()
# set pyAPES to path
pyAPES_main_folder = os.getenv('pyAPES_main_folder')

gpara['start_time'] = '2017-01-01'
gpara['end_time'] = '2018-12-31'
gpara['forc_filename']
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

print(forcing.head())

resultfile, _ = driver(parameters=params,
                       create_ncf=True,
                       result_file='FiHy_2017_2018.nc')
