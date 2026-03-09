from pyAPES.pyAPES_MLM import driver
from pyAPES.parameters.parameter_tools import get_parameter_list
from pyAPES.parameters.SmearII_parameters import gpara, cpara, spara
import cProfile

#  wrap parameters in dictionary
params = {
    'general': gpara,
    'canopy': cpara,
    'soil': spara
}

# parameters simulation(s)
params = get_parameter_list(params, 'hyytiala_2018_lad')
print(params[0]['general'])
# run model

cProfile.run("driver(parameters=params, create_ncf=True, result_file='FiHy2018_lad_profile2.nc')","lad_run_profile_long")
