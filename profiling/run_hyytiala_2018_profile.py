from pyAPES.utils.iotools import read_forcing
from pyAPES.pyAPES_MLM import driver

from pyAPES.parameters.SmearII_parameters import gpara, cpara, spara

from dotenv import load_dotenv
import cProfile

gpara['start_time'] = '2018-06-01'
gpara['end_time'] = '2018-06-02'
load_dotenv()
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

cProfile.run('driver(parameters=params, create_ncf=True, result_file= "FiHy2018_profiling.nc")')