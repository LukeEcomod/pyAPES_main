from pyAPES.pyAPES_MLM import driver
from pyAPES.parameters.mlm_parameters import gpara, cpara, spara
from pyAPES.utils.iotools import read_forcing

gpara['start_time'] = "2007-06-01"
gpara['end_time'] = "2007-06-10"

forcing = read_forcing(
    forcing_file=gpara['forc_filename'],
    start_time=gpara['start_time'],
    end_time=gpara['end_time'],
    dt=gpara['dt'])

#  wrap parameters in dictionary
params = {
    'general': gpara,
    'canopy': cpara,
    'soil': spara,
    'forcing': forcing
}

# run model
outputfile, Model = driver(parameters=params, create_ncf=True, result_file= 'testrun_test_june.nc')
