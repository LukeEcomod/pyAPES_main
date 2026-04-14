#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 11:07:09 2018
TODO:
    - dump parameter space to file
    - check if filter/adapeter can be used for configure loggers Formatter
    (There is need for add nsim, process id can be added somwhere directily)
@author: ajkieloaho
"""

import os
import sys
from threading import Thread
from multiprocessing import Process, Queue, Pool  # , cpu_count
#from psutil import cpu_count
from copy import deepcopy

from pyAPES.utils.iotools import initialize_netcdf, write_ncf, update_logging_configuration, get_interval_slices
from pyAPES.pyAPES_MLM import MLM_model

import time

import logging
import logging.handlers
import logging.config

# Imported at module level so that spawned worker processes can access these names
from pyAPES.parameters.mlm_outputs import parallel_logging_configuration, output_variables


def _result_writer(ncf, writing_queue):
    """
    Args:
        ncf: NetCDF4 file handle
        writing_queue (Queue): queue from which write messages are consumed
    """

    logger = logging.getLogger()
    logger.info("Writer is ready!")

    while True:
        # Message is a 4-tuple (nsim, data, t_start, date_info).
        # t_start enables partial/chunked writes for the write_interval feature.
        # date_info is a human-readable string for logging the written date range.
        msg = writing_queue.get()

        if msg is None:
            ncf.close()
            logger.info("NetCDF4 file is closed. and Writer closes.")
            break

        nsim, data, t_start, date_info = msg
        logger.info("Writing simulation {} ({})".format(nsim, date_info))
        write_ncf(nsim=nsim, results=data, ncf=ncf, t_start=t_start)

# logging to a single file from multiple processes
# https://docs.python.org/dev/howto/logging-cookbook.html#logging-to-a-single-file-from-multiple-processes

def _logger_listener(logging_queue):
    """
    Args:
        logging_queue (Queue): logging queue
    """

    while True:
        record = logging_queue.get()

        if record is None:
            # print('logger done')
            break

        logger = logging.getLogger(record.name)
        logger.handle(record)


def _worker(task_queue, writing_queue, logging_queue):
    """
    Args:
        task_queue (Queue): queue of task initializing parameters
        writing_queue (Queue): queue of model calculation results
        logging_queue (Queue): queue for model loggers
    """

    # --- LOGGING ---
    qh = logging.handlers.QueueHandler(logging_queue)
    root = logging.getLogger()

    # !!! root level set should be in configuration dictionary!!!
    root.handlers = []
    root.setLevel(logging.INFO)
    root.addHandler(qh)

    # --- TASK QUEUE LISTENER ---
    while True:
        task = task_queue.get()

        if task is None:
            root.info('Worker done')
            break

        root.info("Creating simulation {}".format(task['nsim']))

        try:
            model = MLM_model(
                dt=task['general']['dt'],
                canopy_para=task['canopy'],
                soil_para=task['soil'],
                forcing=task['forcing'],
                outputs=output_variables['variables'],
                nsim=task['nsim'],
            )

            write_interval = task['general'].get('write_interval', None)

            # Use chunked writes only when write_interval produces more than
            # one chunk; if it spans the whole simulation, fall back to a
            # single full-simulation write (write_interval treated as None).
            if write_interval is not None:
                slices = get_interval_slices(
                    task['forcing'].index, task['general']['dt'], write_interval)
                if len(slices) <= 1:
                    write_interval = None

            
            if write_interval is None:
                # Write the full simulation results at once
                result = model.run()
                date_info = '{} to {}'.format(
                    task['forcing'].index[0].strftime('%Y-%m-%d'),
                    task['forcing'].index[-1].strftime('%Y-%m-%d'))
                writing_queue.put((task['nsim'], result, 0, date_info))
            else:
                # Periodic writes: generator yields (t_start, t_end, chunk) at interval boundaries
                for t_start, t_end, chunk in model.run(write_interval=write_interval):
                    date_info = '{} to {}'.format(
                        task['forcing'].index[t_start].strftime('%Y-%m-%d'),
                        task['forcing'].index[t_end - 1].strftime('%Y-%m-%d'))
                    writing_queue.put((task['nsim'], chunk, t_start, date_info))

        except:
            message = 'FAILED: simulation {}'.format(task['nsim'])
            # root.info(message + '_' + sys.exc_info()[0])
        # can return something if everything went right


def driver(tasks,
           ncf_params,
           logging_configuration,
           N_workers):
    """
    Args:
        tasks (list): list of task parameter dicts (output of get_parameter_list)
        ncf_params (dict): netCDF4 parameters
        logging_configuration (dict): parallel logging configuration
        N_workers (int): number of worker processes
    """
    task_queue = Queue()
    logging_queue = Queue()
    writing_queue = Queue()

    # Fill the task queue before starting workers
    for task in tasks:
        task_queue.put(deepcopy(task))

    # --- PROCESSES ---
    running_time = time.time()

    workers = []
    for k in range(N_workers):
        workers.append(
            Process(
                target=_worker,
                args=(task_queue, writing_queue, logging_queue),
            )
        )
        workers[k].start()

    # Send one termination sentinel per worker after all tasks are enqueued
    for _ in workers:
        task_queue.put(None)

    # --- NETCDF4 ---
    ncf, _ = initialize_netcdf(
        variables=ncf_params['variables'],
        sim=ncf_params['Nsim'],
        soil_nodes=ncf_params['Nsoil_nodes'],
        canopy_nodes=ncf_params['Ncanopy_nodes'],
        planttypes=ncf_params['Nplant_types'],
        groundtypes=ncf_params['Nground_types'],
        time_index=ncf_params['time_index'],
        filepath=ncf_params['filepath'],
        filename=ncf_params['filename'])

    writing_thread = Thread(
        target=_result_writer,
        args=(ncf, writing_queue),
    )

    writing_thread.start()

    # --- LOGGING ---
    logging_configuration = update_logging_configuration(
        logging_configuration, tasks[0]['general'], ncf_params['filename'],
        handler='parallelAPES_file')
    logging.config.dictConfig(logging_configuration)

    logging_thread = Thread(
        target=_logger_listener,
        args=(logging_queue,),
    )

    logging_thread.start()

    # --- USER INFO ---

    logger = logging.getLogger()
    logger.info('Number of worker processes is {}, number of simulations: {}'.format(N_workers, ncf_params['Nsim']))

    # --- CLOSE ---

    # join worker processes
    for w in workers:
        w.join()

    logger.info('Worker processes have joined.')
    logger.info('Running time %.2f seconds' % (time.time() - running_time))

    # end logging queue and join
    logging_queue.put_nowait(None)
    logging_thread.join()

    # end writing queue and join
    writing_queue.put_nowait(None)
    writing_thread.join()

    logger.info('Results are in path: ' + ncf_params['filepath'])

    return ncf_params['filepath']

if __name__ == '__main__':
    import argparse
    from pyAPES.parameters.mlm_parameters import gpara, cpara, spara
    from pyAPES.parameters.parameter_tools import get_parameter_list

    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', help='number of cpus to be used', type=int)
    parser.add_argument('--scenario', help='scenario name', type=str)
    args = parser.parse_args()

    # --- TASKS ---
    scen = args.scenario

    # list of parameters
    parameters = {
        'general': gpara,
        'canopy': cpara,
        'soil': spara
        }

    tasks = get_parameter_list(parameters, scen)

    # ncf parameters
    ncf_params = {
        'variables': output_variables['variables'],
        'Nsim': len(tasks),
        'Nsoil_nodes': len(tasks[0]['soil']['grid']['dz']),
        'Ncanopy_nodes': tasks[0]['canopy']['grid']['Nlayers'],
        'Nplant_types': len(tasks[0]['canopy']['planttypes']),
        'Nground_types': 1,  # This is tricky if it varies between simulations!!!!!
        'time_index': tasks[0]['forcing'].index,
        'filename': time.strftime('%Y%m%d%H%M_') + scen + '_pyAPES_results.nc',
        'filepath': tasks[0]['general']['results_directory'],
    }

    # --- Number of workers ---
    Ncpu = args.cpu

    N_workers = Ncpu - 1

    # --- DRIVER CALL ---
    outputfile = driver(
        tasks=tasks,
        ncf_params=ncf_params,
        logging_configuration=parallel_logging_configuration,
        N_workers=N_workers)

    print(outputfile)