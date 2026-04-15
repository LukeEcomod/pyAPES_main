import time
from pyAPES.parallelAPES import driver as parallel_driver
from pyAPES.parameters.mlm_outputs import output_variables, parallel_logging_configuration
from pyAPES.parameters.parameter_tools import get_parameter_list
from pyAPES.parameters.SmearII_parameters import gpara, cpara, spara

if __name__ == '__main__':
    params = {
        'general': gpara,
        'canopy': cpara,
        'soil': spara
    }

    tasks = get_parameter_list(params, 'hyytiala_2018_lad')
    print(tasks[0]['general'])

    ncf_params = {
        'variables': output_variables['variables'],
        'Nsim': len(tasks),
        'Nsoil_nodes': len(tasks[0]['soil']['grid']['dz']),
        'Ncanopy_nodes': tasks[0]['canopy']['grid']['Nlayers'],
        'Nplant_types': len(tasks[0]['canopy']['planttypes']),
        'Nground_types': 1,  # This is tricky if it varies between simulations!!!!!
        'time_index': tasks[0]['forcing'].index,
        'filename': time.strftime('%Y%m%d%H%M_') + 'FiHy2018_lad.nc',
        'filepath': tasks[0]['general']['results_directory'],
    }

    N_workers = 2

    outputfile = parallel_driver(
        tasks=tasks,
        ncf_params=ncf_params,
        logging_configuration=parallel_logging_configuration,
        N_workers=N_workers)

    print(outputfile)
