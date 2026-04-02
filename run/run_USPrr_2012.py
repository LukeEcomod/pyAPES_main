import sys
import os
from dotenv import load_dotenv
# function to read forcing data. See 'forcing/forcing_info.txt' for model forcing variable names and units!
from pyAPES.utils.iotools import read_forcing
# import the multi-layer model (mlm) driver
from pyAPES.pyAPES_MLM import driver
# Read model parameter dictionaries
from pyAPES.parameters.mlm_parameters_US_Prr import gpara, cpara, spara

# Set pyAPES to path
load_dotenv()
pyAPES_main_folder = os.getenv('pyAPES_main_folder')
sys.path.append(pyAPES_main_folder)

# Edit the start and end time of the simulation

gpara['start_time'] = '2012-06-01'
gpara['end_time'] = '2013-12-31'

# create forcing
forcing = read_forcing(
    forcing_file=gpara['forc_filename'],
    start_time=gpara['start_time'],
    end_time=gpara['end_time'],
    dt=gpara['dt']
)

params = {
    'general': gpara,   # model configuration
    'canopy': cpara,    # planttype, micromet, canopy, bottomlayer parameters
    'soil': spara,      # soil heat and water flow parameters
    'forcing': forcing  # forging data
}

# Run the model

resultfile, Model = driver(parameters=params,
                           create_ncf=True,
                           result_file= 'USPrr_2011_2012.nc'
                          )