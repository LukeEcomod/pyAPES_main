import cProfile
from pyAPES.pyAPES_MLM import driver
from pyAPES.parameters.mlm_parameters import gpara, cpara, spara
from pyAPES.parameters.parameter_tools import get_parameter_list
from pyAPES.utils.iotools import read_forcing

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

# parameters simulation(s)
params = get_parameter_list(params, 'hyytiala_2018_lad')
print(params[0]['general'])

# run model

cProfile.run("driver(parameters=params, create_ncf=True, result_file= 'testrun_fullsoil_profile_moss_iteration_hackathon_day1.nc')","lad_run_profile_long_fullsoil_moss_iteration_hackathon_day1.prof")

# # run model
# outputfile, Model = driver(parameters=params, create_ncf=True, result_file= 'testrun_fullsoil_profile.nc')

