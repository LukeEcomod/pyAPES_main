# -*- coding: utf-8 -*-
"""
.. module: utils.iotools
    :synopsis: pyAPES component for data input/outputs
.. moduleauthor:: Kersti Leppä, Antti-Jussi Kieloaho & Samuli Launiainen

"""

import sys
import os
from pathlib import Path
import pandas as pd
import xarray as xr
import numpy as np
import json
import yaml
import logging
from typing import Dict, Tuple, List

def get_interval_slices(forcing_index, dt: float, write_interval: str) -> list:
    """
    Computes (t_start, t_end) integer index pairs for chunked NetCDF writes.

    Args:
        forcing_index: pandas DatetimeIndex of the simulation forcing
        dt (float): model timestep in seconds
        write_interval (str): pandas offset string (e.g. '1D', '1M', '1Y')
    Returns:
        list of (t_start, t_end) pairs; always at least one entry covering
        the whole simulation when write_interval >= simulation length.
    """
    idx = forcing_index
    dt_td = pd.Timedelta(seconds=dt)
    boundaries = pd.date_range(start=idx[0], end=idx[-1] + dt_td, freq=write_interval)

    slices = []
    t_start = 0
    for boundary in boundaries[1:]:
        t_end = int((idx < boundary).sum())
        if t_end > t_start:
            slices.append((t_start, t_end))
            t_start = t_end

    if t_start < len(idx):
        slices.append((t_start, len(idx)))

    return slices


def update_logging_configuration(logging_configuration, general_parameters, ncf_filename, handler='file'):
    """
    Updates the log file path in logging_configuration.

    The log filename is derived from ncf_filename by replacing the .nc
    extension with .log.  The directory is read from
    general_parameters['logging_directory'] (optional; defaults to the
    same directory as the NCF file when omitted).

    Args:
        logging_configuration (dict): logging configuration dict
        general_parameters (dict): gpara dict from simulation parameters
        ncf_filename (str): NetCDF4 output filename (basename, e.g. '20240101_run.nc')
        handler (str): name of the file handler to update (default 'file');
            use 'parallelAPES_file' for parallel runs
    Returns:
        logging_configuration (dict): updated configuration
    """
    log_dir_str = general_parameters.get('logging_directory', '')
    log_dir = Path(log_dir_str) if log_dir_str else Path()

    if log_dir_str:
        log_dir.mkdir(parents=True, exist_ok=True)

    log_basename = Path(ncf_filename).stem + '.log'
    logfile = log_dir / log_basename

    logging_configuration['handlers'][handler]['filename'] = str(logfile)

    return logging_configuration


def initialize_netcdf(variables,
                      sim,
                      soil_nodes,
                      canopy_nodes,
                      planttypes,
                      groundtypes,
                      time_index,
                      filepath='results/',
                      filename='test.nc',
                      description='pyAPES_MLM results'):
    r"""
    Creates pyAPES_MLM NetCDF4 format output file

    Args:
        variables (list): list of variables to be saved in netCDF4
        sim (int): number of simulations
        soil_nodes (int): number of soil calculation nodes
        canopy_nodes (int): number of canopy calculation nodes
        time_index (np.datetimeindex): time_index of default forcing data (pd.DataSeries)
        filepath (str): path for saving results
        filename (str): filename
        description (str): info
    """
    from netCDF4 import Dataset, date2num
    from datetime import datetime
    from dotenv import load_dotenv
    load_dotenv()
    pyapes_main_folder = os.getenv('pyAPES_main_folder')

    if not pyapes_main_folder:
        pyapes_main_folder = os.getcwd()
    filepath = os.path.join(pyapes_main_folder, filepath)

    if not os.path.exists(filepath):
        os.makedirs(filepath)

    ff = os.path.join(filepath, filename)

    # create dataset and dimensions
    ncf = Dataset(ff, 'w')
    ncf.description = description
    ncf.history = 'created ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ncf.source = 'pyAPES_1.0'

    ncf.createDimension('date', None)
    ncf.createDimension('simulation', sim)
    ncf.createDimension('soil', soil_nodes)
    ncf.createDimension('canopy', canopy_nodes)
    ncf.createDimension('planttype', planttypes)
    ncf.createDimension('groundtype', groundtypes)

    time = ncf.createVariable('date', 'f8', ('date',))
    time.units = 'days since 0001-01-01 00:00:00.0'
    time.calendar = 'standard'
#    tvec = [k.to_datetime() for k in forcing.index] is depricated
    tvec = [pd.to_datetime(k) for k in time_index]
    time[:] = date2num(tvec, units=time.units, calendar=time.calendar)

    for var in variables:

        var_name = var[0]
        var_unit = var[1]
        var_dim = var[2]

        if var_name == 'canopy_planttypes' or var_name == 'ffloor_groundtypes':
            variable = ncf.createVariable(
                var_name, 'S10', var_dim)
        else:
            variable = ncf.createVariable(
                    var_name, 'f4', var_dim)

        variable.units = var_unit

    return ncf, ff

def write_ncf(nsim=None, results=None, ncf=None, t_start=0):
    r"""
    Writes pyAPES_MLM results into NetCDF4-file

    Args:
        nsim (int): simulation index
        results (dict): calculation results from group
        ncf (object): netCDF4-file handle
        t_start (int): starting time index for writing; enables partial/chunked
            writes. Default 0 writes from the beginning (full-simulation write).
    """

    # Variables without a time dimension written only once
    static_keys = {'soil_z', 'canopy_z', 'canopy_planttypes', 'ffloor_groundtypes'}

    keys = results.keys()
    variables = ncf.variables.keys()

    # Determine chunk length from the first time-varying result array.
    # Needed to compute the write slice [t_start : t_start + chunk_len].
    # For a full-simulation write chunk_len == Nsteps; for periodic writes
    # it equals the number of timesteps in the current interval.
    chunk_len = None
    for key in keys:
        if key in variables and key != 'date' and key not in static_keys:
            arr = np.asarray(results[key])
            if arr.ndim >= 1:
                chunk_len = arr.shape[0]
                break

    for key in keys:
        if key in variables and key != 'date':
            arr = np.asarray(results[key])
            if key in static_keys:
                # Static grid/metadata arrays have no time dimension.
                # Write only once, from simulation 0's results.
                # Static keys appear only in the last
                # chunk of a chunked run (or the single full-sim results dict),
                # so there is no risk of writing them more than once.
                if nsim == 0:
                    ncf[key][:] = arr

            elif arr.ndim > 1:
                # Time-varying arrays with spatial/type dimensions:
                #   (date, simulation, canopy/soil/planttype/groundtype)   ndim=2
                #   (date, simulation, planttype, canopy)                  ndim=3
                # nsim selects the simulation column; trailing ':' covers
                # all remaining spatial/type dimensions regardless of count.
                ncf[key][t_start:t_start + chunk_len, nsim, :] = arr

            else:
                # Time-varying scalars stored as (date, simulation) in NCF
                # but as 1-D [Nsteps] arrays in results.
                ncf[key][t_start:t_start + chunk_len, nsim] = arr


def read_forcing(forcing_file: str, start_time: str, end_time: str,
                 dt: float=1800.0, na_values: str='NaN', sep: str=';') -> pd.DataFrame:
    """
    Reads model forcing data from csv-file to pd.DataFrame

    Args:
        forc_fp (str): forcing file path
        start_time (str): starting time [yyyy-mm-dd], if None first date in
            file used
        end_time (str): ending time [yyyy-mm-dd], if None last date
            in file used
        dt (float): time step [s], if given checks
            that dt in file is equal to this
        na_values (str|float): nan value representation in file
        sep (str): field separator
    Returns:
        Forc (pd.DataFrame): dataframe with datetimeindex and columns read from file
    """

    # filepath
    #forc_fp = "forcing/" + forc_filename
    dat = pd.read_csv(forcing_file, header='infer', na_values=na_values, sep=sep)

    # set to dataframe index
    tvec = pd.to_datetime(dat[['year', 'month', 'day', 'hour', 'minute']])
    tvec = pd.DatetimeIndex(tvec)
    dat.index = tvec

    dat = dat[(dat.index >= start_time) & (dat.index <= end_time)]

    # convert: H2O mmol / mol --> mol / mol; Prec kg m-2 in dt --> kg m-2 s-1
    dat['H2O'] = 1e-3 * dat['H2O']
    dat['Prec'] = dat['Prec'] / dt

    cols = ['doy', 'Prec', 'P', 'Tair', 'Tdaily', 'U', 'Ustar', 'H2O', 'CO2', 'Zen',
            'LWin', 'diffPar', 'dirPar', 'diffNir', 'dirNir']
    
    # these needed for phenology model initialization
    if 'X' in dat:
        cols.append('X')
    if 'DDsum' in dat:
        cols.append('DDsum')

    # for case for bypassing soil computations
    if 'Tsa' in dat:
        cols.append('Tsa')
    if 'Wa' in dat:
        cols.append('Wa')
    if 'Rew' in dat:
        cols.append('Rew')  

    # Create dataframe from specified columns
    Forc = dat[cols].copy()

    # Check time step if specified
    if len(set(Forc.index[1:]-Forc.index[:-1])) > 1:
        sys.exit("Forcing file does not have constant time step")
    if (Forc.index[1] - Forc.index[0]).total_seconds() != dt:
        sys.exit("Forcing file time step differs from dt given in general parameters")

    return Forc

def read_data(ffile: str, start_time: str=None, end_time: str=None, na_values: str='NaN', sep: str=';') -> pd.DataFrame:
    r"""
    Reads csv-datafile into pd.DataFrame
    Args:
        ffile (str): filepath
        start_time (str): starting time [yyyy-mm-dd], if None first date in
            file used
        end_time (str): ending time [yyyy-mm-dd], if None last date
            in file used
        na_values (str|float): nan value representation in file
        sep (str): field separator
    Returns:
        dat (pd.DataFrame): dataframe with datetimeindex and columns read from file
    """
    
    dat = pd.read_csv(ffile, header='infer', na_values=na_values, sep=sep)
    
    # set to dataframe index
    tvec = pd.to_datetime(dat[['year', 'month', 'day', 'hour', 'minute']])
    tvec = pd.DatetimeIndex(tvec)
    dat.index = tvec

    # select time period
    if start_time == None:
        start_time = dat.index[0]
    if end_time == None:
        end_time = dat.index[-1]

    dat = dat[(dat.index >= start_time) & (dat.index <= end_time)]

    return dat

def read_results(outputfiles):
    """
    Opens simulation results from NetCDF4 dataset(s) in xr dataset(s)
    Args:
        outputfiles (str|list):
    Returns:
        results (xarray|list of xarrays):
            simulation results from given outputfile(s)
    """
    if type(outputfiles) != list:
        outputfiles = [outputfiles]

    results = []
    for outputfile in outputfiles:
        fp = outputfile
        result = xr.open_dataset(fp)
        result.coords['simulation'] = result.simulation.values
        result.coords['soil'] = result.soil_z.values
        result.coords['canopy'] = result.canopy_z.values
#        result.coords['planttype'] = ['pine','spruce','decid','shrubs']
        results.append(result)

    if len(results) == 1:
        return results[0]
    else:
        return results


def _sanitize_for_yaml(obj):
    if isinstance(obj, dict):
        return {k: _sanitize_for_yaml(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_yaml(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return '<excluded>'
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_parameters_yaml(parameters, stem: str, directory: Path):
    """
    Saves simulation parameters as a YAML file.

    Args:
        parameters (list | dict): list of parameter dicts, or a dict mapping
            simulation keys to parameter dicts. Forcing keys are excluded.
        stem (str): filename stem (without extension) shared with the NCF output
        directory (Path): directory where the YAML file is written
    """
    logger = logging.getLogger(__name__)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    yaml_path = directory / (stem + '.yml')

    if isinstance(parameters, dict):
        sanitized = {k: _sanitize_for_yaml(v) for k, v in parameters.items()}
    else:
        sanitized = []
        for p in parameters:
            entry = {k: _sanitize_for_yaml(v) for k, v in p.items() if k != 'forcing'}
            sanitized.append(entry)
        if len(sanitized) == 1:
            sanitized = sanitized[0]

    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(sanitized, f, default_flow_style=False, allow_unicode=True)

    logger.info('Parameters saved to: ' + str(yaml_path))


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def jsonify(params, file_name='parameter_space.json'):
    """ Dumps simulation parameters into json format
    """

    os.path.join(file_name)

    with open(file_name, 'w+') as fp:
        json.dump(params, fp, cls=NumpyEncoder)


def open_json(file_path):
    """ Opens a json file
    """
    os.path.join(file_path)
    with open(file_path) as json_data:
        data = json.load(json_data)

    return data

