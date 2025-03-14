# -*- coding: utf-8 -*-
"""
.. module: pyAPES.models.mlm
    :synopsis: pyAPES component. Multi-layer, multi-species SVAT model
.. moduleauthor:: Samuli Launiainen, Kersti Leppä & Antti-Jussi Kieloaho

Multi-layer, multi-species 1D soil-vegetation-atmosphere transfer model.

Key references:

Launiainen, S., Katul, G.G., Lauren, A. and Kolari, P., 2015. Coupling boreal
forest CO2, H2O and energy flows by a vertically structured forest canopy –
soil model with separate bryophyte layer. Ecological Modelling, 312, pp.385-405.

Leppä, K. et al. 2020. Agric. For. Meteorology

***
Demo & primitive user-guide in /Examples/mlm_demo.ipynb 

To call model and run single simulation and read results: see example in sandbox.pyfrom tools.iotools import read_results
    from pyAPES import driver
    # for NetCDF-outputs
    outputfile = driver(create_ncf=True, result_file='test.nc')
    results = read_results(outputfile) # opens NetCDF-file using xarray

    # for returning results directly
    results = driver(create_ncf=False) # returns dict with integer keys
    results = results[0] # first simulation

LAST EDIT: 15.1.2020 Samuli Launiainen
    * new forestfloor and altered outputs
Todo:

    - make minimal example of handling and plotting outputs using xarray -tools;
      now see tools.iotools.read_forcing for documentation!

"""
import numpy as np
import time
import logging
from pandas import date_range
from typing import List, Tuple, Dict

from pyAPES.utils.iotools import initialize_netcdf,  write_ncf
from pyAPES.canopy.mlm_canopy import CanopyModel
from pyAPES.soil.soil import Soil_1D

from pyAPES.utils.constants import WATER_DENSITY

def driver(parameters,
           create_ncf=False, 
           result_file=None):
    """
    Driver for pyAPES_MLM multi-layer model. Gets parameters and forcing as argument, prepares output files, runs model.
    
    Args:
        parameters (dict/list): either single parameter dictionary or list of parameter dictionaries

        create_ncf (bool): results saved to netCDF4 file
        result_file (str): name of result file
    Returns:
        output_file (str): file path
        tasks[0] (object): model object from task[0], state corrensponds to end of simulation
    """

    # --- CONFIGURATION PARAMETERS of LOGGING and NetCDF -outputs read
    from pyAPES.parameters.mlm_outputs import output_variables, logging_configuration
    #from logging.config import dictConfig

    # --- Config logger
    logging.config.dictConfig(logging_configuration)
    logger = logging.getLogger(__name__)

    # --- Check parameters

    if isinstance(parameters, dict):
        Nsim = 1 # number of simulations
        parameters = [parameters]
    elif isinstance(parameters, list):
        Nsim = len(parameters)
    else:
        raise TypeError('Parameters should be either dict or list.')

    logger.info('pyAPES_MLM simulation started. Number of simulations: {}'.format(Nsim))

    # --- Run simulations

    tasks = []

    for k in range(Nsim):
        tasks.append(
                MLM_model(
                parameters[k]['general']['dt'],
                parameters[k]['canopy'],
                parameters[k]['soil'],
                parameters[k]['forcing'],
                output_variables['variables'],
                nsim=k
            )
        )

    if create_ncf: # outputs to NetCDF-file, returns filename
        gpara = parameters[0]['general'] # same for all tasks
        timestr = time.strftime('%Y%m%d%H%M')
        if result_file:
            filename = result_file
        else:
            filename = timestr + '_pyAPES_results.nc'

        time_index = parameters[0]['forcing'].index

        ncf, _ = initialize_netcdf(
                output_variables['variables'],
                Nsim,
                tasks[k].Nsoil_nodes,
                tasks[k].Ncanopy_nodes,
                tasks[k].Nplant_types,
                tasks[k].Nground_types,
                time_index=time_index,
                filepath=gpara['results_directory'],
                filename=filename)

        for task in tasks:
            logger.info('Running simulation number (start time %s): %s' % (
                        time.strftime('%Y-%m-%d %H:%M'), task.Nsim))
            running_time = time.time()
            results = task.run()
            logger.info('Running time %.2f seconds' % (time.time() - running_time))
            write_ncf(nsim=task.Nsim, results=results, ncf=ncf)

            del results

        output_file = gpara['results_directory'] + filename
        logger.info('Ready! Results are in: ' + output_file)

        ncf.close()

        return output_file, tasks[0]

    else: # returns dictionary of outputs
        running_time = time.time()
        results = {task.Nsim: task.run() for task in tasks}

        logger.info('Running time %.2f seconds' % (time.time() - running_time))

        return results, tasks[0] # this would return also 1st Model instance


class MLM_model(object):
    """
    pyAPES_MLM -model.

    Combines submodels 'CanopyModel' and 'Soil' and handles data-transfer
    between these components and writing results.

    """
    def __init__(self, 
                 dt: float,
                 canopy_para: Dict,
                 soil_para: Dict,
                 forcing: Dict, # or pd.DataFrame
                 outputs: Dict,
                 nsim: int=0
                ):

        self.dt = dt

        self.Nsteps = len(forcing)
        self.forcing = forcing
        self.Nsim = nsim

        self.Nsoil_nodes = len(soil_para['grid']['dz'])
        self.Ncanopy_nodes = canopy_para['grid']['Nlayers']

        # create soil model instance
        self.soil = Soil_1D(soil_para)

        # create canopy model instance
        
        # initial delayed temperature and degreedaysum for pheno & LAI-models
        if canopy_para['ctr']['pheno_cycle'] and 'X' in forcing:
            for pt in list(canopy_para['planttypes'].keys()):
                canopy_para['planttypes'][pt]['phenop'].update({'Xo': forcing['X'].iloc[0]})
        
        if canopy_para['ctr']['seasonal_LAI'] and 'DDsum' in forcing:
            for pt in list(canopy_para['planttypes'].keys()):
                canopy_para['planttypes'][pt]['laip'].update({'DDsum0': forcing['DDsum'].iloc[0]})

        self.canopy_model = CanopyModel(canopy_para, self.soil.grid['dz'])

        self.Nplant_types = len(self.canopy_model.planttypes)
        self.Nground_types = len(self.canopy_model.forestfloor.bottomlayer_types)

        # initialize temorary structure to save results
        self.results = _initialize_results(outputs,
                                       self.Nsteps,
                                       self.Nsoil_nodes,
                                       self.Ncanopy_nodes,
                                       self.Nplant_types,
                                       self.Nground_types)

    def run(self):
        """
        Loops through self.forcing timesteps and appends to self.results.

        self.forcing variables and units correspond to uppermost gridpoint:
            precipitation [kg m-2 s-1]
            air_pressure [Pa]
            air_temperature [degC]
            wind_speed [m s-1]
            friction_velocity [m s-1]
            h2o [mol mol-1]
            co2 [ppm]
            zenith_angle [rad]
            lw_in: Downwelling long wave radiation [W m-2]
            diffPar: Diffuse PAR [W m-2]
            dirPar: Direct PAR [W m-2]
            diffNir: Diffuse NIR [W m- 2]
            dirNir: Direct NIR [W m-2]
        
        Args:
            self (object)
        
        Returns:
            self.results (dict)
        """

        logger = logging.getLogger(__name__)
        logger.info('Running simulation {}'.format(self.Nsim))
        time0 = time.time()

        #print('RUNNING')
        k_steps=np.arange(0, self.Nsteps, int(self.Nsteps/10))

        for k in range(0, self.Nsteps):
            #print(k)
            # --- print progress on screen
            if k in k_steps[:-1]:
                s = str(np.where(k_steps==k)[0][0]*10) + '%'
                print('{0}..'.format(s), end=' ')

            # --- CanopyModel ---
            # run daily loop: updates LAI, phenology and moisture stress ---
            if self.forcing['doy'].iloc[k] != self.forcing['doy'].iloc[k-1] or k == 0:
                self.canopy_model.run_daily(
                        self.forcing['doy'].iloc[k],
                        self.forcing['Tdaily'].iloc[k])

            # compile forcing dict for canopy model: soil_ refers to state of soil model
            canopy_forcing = {
                'wind_speed': self.forcing['U'].iloc[k],            # [m s-1]
                'friction_velocity': self.forcing['Ustar'].iloc[k], # [m s-1]
                'air_temperature': self.forcing['Tair'].iloc[k],    # [deg C]
                'precipitation': self.forcing['Prec'].iloc[k],      # [kg m-2 s-1]
                'h2o': self.forcing['H2O'].iloc[k],                 # [mol mol-1]
                'co2': self.forcing['CO2'].iloc[k],                 # [ppm]
                'PAR': {'direct': self.forcing['dirPar'].iloc[k],   # [W m-2]
                        'diffuse': self.forcing['diffPar'].iloc[k]},
                'NIR': {'direct': self.forcing['dirNir'].iloc[k],   # [W m-2]
                        'diffuse': self.forcing['diffNir'].iloc[k]},
                'lw_in': self.forcing['LWin'].iloc[k],              # [W m-2]
                'air_pressure': self.forcing['P'].iloc[k],          # [Pa]
                'zenith_angle': self.forcing['Zen'].iloc[k],        # [rad]

                # from soil model
                'soil_temperature': self.soil.heat.T,         # [deg C]
                'soil_water_potential': self.soil.water.h,    # [m]
                'soil_volumetric_water': self.soil.heat.Wliq, # total water content [m3 m-3]
                'soil_volumetric_ice': self.soil.heat.Wice,
                'soil_volumetric_air': self.soil.heat.Wair,   # [m3 m-3]
                'soil_pond_storage': self.soil.water.h_pond * WATER_DENSITY,      
                  
                # 'soil_temperature': self.soil.heat.T[self.canopy_model.ix_roots],         # [deg C]
                # 'soil_water_potential': self.soil.water.h[self.canopy_model.ix_roots],    # [m]
                # 'soil_volumetric_water': self.soil.heat.Wliq[self.canopy_model.ix_roots], # total water content [m3 m-3]
                # 'soil_volumetric_ice': self.soil.heat.Wice[self.canopy_model.ix_roots],
                # 'soil_volumetric_air': self.soil.heat.Wair[self.canopy_model.ix_roots],   # [m3 m-3]
                # 'soil_pond_storage': self.soil.water.h_pond * WATER_DENSITY,              # [kg m-2]
            }

            canopy_parameters = {
                'soil_depth': self.soil.grid['z'][0],   # [m]
                'soil_porosity': self.soil.heat.porosity, #[m3 m-3]
                'soil_hydraulic_conductivity': self.soil.water.Kv, # [m s-1]
                #'soil_hydraulic_conductivity': self.soil.water.Kv[self.canopy_model.ix_roots], # [m s-1]
                'soil_thermal_conductivity': self.soil.heat.thermal_conductivity[0],        # [W m-1 K-1]
                # SINGLE SOIL LAYER
                # 'state_water':{'volumetric_water_content': self.forcing['Wliq'].iloc[k]},
                #                'state_heat':{'temperature': self.forcing['Tsoil'].iloc[k]}
                'date': self.forcing.index[k]   # pd.datetime
            }

            # call self.canopy_model.run to solve above-ground part
            out_canopy, out_planttype, out_ffloor, out_groundtype = self.canopy_model.run(
                dt=self.dt,
                forcing=canopy_forcing,
                parameters=canopy_parameters
            )

            # --- Soil model  ---
            # compile forcing for Soil: potential infiltration and evaporation are at from ground surface
            # water fluxes must be in [m s-1]
            soil_forcing = {
                'potential_infiltration': out_ffloor['throughfall'] / WATER_DENSITY,
                'potential_evaporation': ((out_ffloor['soil_evaporation'] +
                                           out_ffloor['capillary_rise']) / WATER_DENSITY),
                'pond_recharge': out_ffloor['pond_recharge'] / WATER_DENSITY,
                'atmospheric_pressure_head': -1.0E6,  # set to large value, because potential_evaporation already account for h_soil
                'ground_heat_flux': -out_ffloor['ground_heat'],
                'date': self.forcing.index[k]}

            # call self.soil to solve below-ground water and heat flow
            soil_flux, soil_state = self.soil.run(
                    dt=self.dt,
                    forcing=soil_forcing,
                    water_sink=out_canopy['root_sink'])

            # --- append results and copy of forcing to self.results
            forcing_output = {
                    'wind_speed': self.forcing['U'].iloc[k],
                    'friction_velocity': self.forcing['Ustar'].iloc[k],
                    'air_temperature': self.forcing['Tair'].iloc[k],
                    'precipitation': self.forcing['Prec'].iloc[k],
                    'h2o': self.forcing['H2O'].iloc[k],
                    'co2': self.forcing['CO2'].iloc[k],
                    'pressure': self.forcing['P'].iloc[k],
                    'par':  self.forcing['dirPar'].iloc[k] + self.forcing['diffPar'].iloc[k],
                    'nir':  self.forcing['dirNir'].iloc[k] + self.forcing['diffNir'].iloc[k],
                    'lw_in': self.forcing['LWin'].iloc[k]
                    }

            soil_state.update(soil_flux)

            self.results = _append_results('forcing', k, forcing_output, self.results)
            self.results = _append_results('canopy', k, out_canopy, self.results)
            self.results = _append_results('ffloor', k, out_ffloor, self.results)
            self.results = _append_results('soil', k, soil_state, self.results)
            self.results = _append_results('pt', k, out_planttype, self.results)
            self.results = _append_results('gt', k, out_groundtype, self.results)

        print('100%')

        # append plantype, groundtype and grid information
        ptnames = [pt.name for pt in self.canopy_model.planttypes]

        self.results = _append_results('canopy', None, {'z': self.canopy_model.z,
                                                        'planttypes': np.array(ptnames)}, self.results)

        gtnames = [gt.name for gt in self.canopy_model.forestfloor.bottomlayer_types]

        self.results = _append_results('ffloor', None, {'groundtypes': np.array(gtnames)}, self.results)

        self.results = _append_results('soil', None, {'z': self.soil.grid['z']}, self.results)

        logger.info('Finished simulation %.0f, running time %.2f seconds' % (self.Nsim, time.time() - time0))

        return self.results


def _initialize_results(variables: Dict, 
                        Nstep: int, 
                        Nsoil_nodes: int, 
                        Ncanopy_nodes: int, 
                        Nplant_types: int,
                        Nground_types: int):
    """
    Creates temporary dictionary to accumulate simulation results
    Args:
        see arguments
    Returns:
        results (dict)
    """

    results = {}

    for var in variables:

        var_name = var[0]
        dimensions = var[2]

        if 'canopy' in dimensions:
            if 'planttype' in dimensions:
                var_shape = [Nstep, Nplant_types, Ncanopy_nodes]
            else:
                var_shape = [Nstep, Ncanopy_nodes]

        elif 'soil' in dimensions:
            var_shape = [Nstep, Nsoil_nodes]

        elif 'planttype' in dimensions and 'canopy' not in dimensions:
            var_shape = [Nstep, Nplant_types]

        elif 'groundtype' in dimensions:
            if 'date' not in dimensions:
                var_shape = [Nground_types]
            else:
                var_shape = [Nstep, Nground_types]

        else:
            var_shape = [Nstep]

        results[var_name] = np.full(var_shape, np.NAN)
        # print(var_name, var_shape, dimensions)

    return results


def _append_results(group: str, step: int, step_results: Dict, results: Dict):
    """
    Adds results from timestep to temporary dictionary
    
    Args:
        group (str): group_id
        step (int): timestep index
        step_results (dict): results from 'step'
        results (dict): simulation results to where step_results are appended

    Returns:
        results (dict): updated results
    """

    results_keys = results.keys()
    step_results_keys = step_results.keys()

    for key in step_results_keys:
        variable = group + '_' + key
        if variable in results_keys:
            if key == 'z' or key == 'planttypes' or key == 'groundtypes':
                results[variable] = step_results[key]
            else:
                #print(variable, key, np.shape(results[variable][step]), np.shape(step_results[key]))
                results[variable][step] = step_results[key]

    return results

if __name__ == '__main__':

    # setting path
    import sys
    #sys.path.append('c:\\Repositories\\pyAPES_main')
    import os
    from dotenv import load_dotenv

    load_dotenv()
    pyAPES_main_folder = os.getenv('pyAPES_main_folder')

    sys.path.append(pyAPES_main_folder)
    #print(sys.path)

    #import argparse
    from pyAPES.parameters.mlm_outputs import output_variables
    from pyAPES.parameters.mlm_parameters import gpara, cpara, spara
    from pyAPES.utils.iotools import read_forcing

    # read forcing data
    forcing = read_forcing(
        forcing_file=gpara['forc_filename'],
        start_time=gpara['start_time'],
        end_time=gpara['end_time'],
        dt=gpara['dt']
    )

    # combine parameterset
    params = {
        'general': gpara,   # model configuration
        'canopy': cpara,    # planttype, micromet, canopy, bottomlayer parameters
        'soil': spara,      # soil heat and water flow parameters
        'forcing': forcing  # forging data
    }
 
    # run the model
    resultfile, Model = driver(parameters=params,
                           create_ncf=True,
                           result_file= 'testrun.nc'
                          )
# EOF