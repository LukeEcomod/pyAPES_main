from pyAPES.pyAPES_MLM import driver
from pyAPES.parameters.parameter_tools import get_parameter_list
from pyAPES.parameters.SmearII_parameters import gpara, cpara, spara
from pyAPES.utils.iotools import read_forcing

# gpara['end_time'] = "2018-06-05"

#  wrap parameters in dictionary
params = {
    'general': gpara,
    'canopy': cpara,
    'soil': spara
}

forcing = read_forcing(
    forcing_file=gpara['forc_filename'],
    start_time=gpara['start_time'],
    end_time=gpara['end_time'],
    dt=gpara['dt']
)

params['forcing'] = forcing

# run model
outputfile, Model = driver(parameters=params, create_ncf=True, result_file= 'FiHy2018.nc')

