# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 11:00:09 2024

@author: Kersti Leppa
"""

# function to read forcing data. See 'forcing/forcing_info.txt' for model forcing variable names and units!
from pyAPES.utils.iotools import read_forcing

# import the multi-layer model (mlm) driver
from pyAPES.pyAPES_MLM import driver



eps = 1e-16

# import parameter dictionaries
from pyAPES.parameters.mlm_parameters import gpara, cpara, spara # model configuration, canopy parameters, soil parameters

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

# run model
resultfile, Model = driver(parameters=params,
                           create_ncf=True,
                           result_file= 'testrun.nc'
                          )

from pyAPES.utils.iotools import read_results
import matplotlib.pyplot as plt

# read simulation restuls to xarray dataset
results = read_results(resultfile)

sim = 0  # in this demo, we have only one simulation (i.e. only one parameter set was used)

# grid variables for plotting
t = results.date  # time
zc = results.canopy_z  # height above ground [m]
zs = results.soil_z  # depth of soil; shown negative [m]

import numpy as np
var = ['soil_temperature', 'soil_volumetric_water_content']

lyrs = [0, 1, 2, 3, 4] # five top layers
#depths = np.array2string(np.asarray(zs[lyrs]), precision=1, separator=', ')
depths = ['{:.2f} m'.format(k) for k in zs[lyrs]]

fig, ax = plt.subplots(2, 1, figsize=(10,7))

k = 0
for v in var:
    ax[k].plot(t, results[v][:,sim,lyrs], label=depths)
    ax[k].set_ylabel(results[v].attrs['units'])
    ax[k].tick_params(axis='x', labelrotation = 20)
    ax[k].legend(fontsize=8)
    k += 1

# vertical profile at last timestep
fig, ax = plt.subplots(1, 2) #figsize=(10,5))

k = 0
for v in var:
    ax[k].plot(results[v][-1,sim,:], zs)
    ax[k].set_xlabel(results[v].attrs['units'])
    ax[0].set_ylabel('depth (m)')
    k += 1