

import sys
import os
from dotenv import load_dotenv
import pathlib

load_dotenv()
pyAPES_main_folder = os.getenv('pyAPES_main_folder')
sys.path.append(pyAPES_main_folder)


pyAPES_main_folder = pathlib.Path(os.getenv("pyAPES_main_folder")).resolve()
sys.path.insert(0, str(pyAPES_main_folder))


from pyAPES.utils.iotools import read_forcing
from pyAPES.pyAPES_MLM import driver

# from pyAPES.parameters.SmearII_parameters import gpara, cpara, spara
from pyAPES.parameters.FiRan_parameters import gpara, cpara, spara


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

resultfile, _ = driver(parameters=params,
                       create_ncf=True,
                       result_file='FiRan_2022_test.nc')
