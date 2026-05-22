# debugging mlm, case ränskälänkorpi
# this script just runs the mlm for ränskis 2022
# it's just nicer to run a .py in debugging mode than a notebook
# 
# SS may 22 2026

import warnings
# warnings.simplefilter("always", RuntimeWarning)
warnings.simplefilter("always")
import logging
logging.captureWarnings(True)

import sys
import os
from dotenv import load_dotenv

load_dotenv()
pyAPES_main_folder = os.getenv('pyAPES_main_folder')

sys.path.append(pyAPES_main_folder)

from pyAPES.utils.iotools import read_forcing
from pyAPES.pyAPES_MLM import driver

# import parameter dictionaries
from pyAPES.parameters.mlm_parameters_FI_Ran import gpara, cpara, spara # model configuration, canopy parameters, soil parameters

# forc_filename should lead to parent folder where forcing folder is
forcing_file_path = '../'+gpara['forc_filename']

forcing_file_path = os.path.join(pyAPES_main_folder, gpara['forc_filename'])

forcing = read_forcing(
    forcing_file=forcing_file_path,
    start_time=gpara['start_time'],
    end_time=gpara['end_time'],
    dt=gpara['dt']
)

params = {
    'general': gpara,   # model configuration
    'canopy': cpara,    # planttype, micromet, canopy, bottomlayer parameters
    'soil': spara,      # soil heat and water flow parameters
    'forcing': forcing  # forcing data
}


resultfile, Model = driver(parameters=params,
                           create_ncf=True,
                           result_file= 'ran_22_debug15.nc' # 
                          )

# resultfile  = f'{pyAPES_main_folder}\\results\\ran_22_debug1.nc'

from pyAPES.utils.iotools import read_results, read_data

# read simulation restuls to xarray dataset
results = read_results(resultfile)
print(results)