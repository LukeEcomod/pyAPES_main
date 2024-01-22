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
from pyAPES.parameters.mlm_parameters import gpara, cpara, spara, isopara # model configuration, canopy parameters, soil parameters

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
    'isotopes': isopara,  # isotope parameters 
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

var = ['pt_d13c_leaf_sugar','pt_d18o_leaf_water','pt_d18o_leaf_sugar']

fig, ax = plt.subplots(1,3, figsize=(10,3.5), sharey='all')
for i, v in enumerate(var):
    results[v+'_sunlit'][-23,0,0,:].plot(y='canopy',ax=ax[i])
    results[v+'_shaded'][-23,0,0,:].plot(y='canopy',ax=ax[i])
    ax[i].set_ylabel('')
    ax[i].set_xlabel(results[v+'_shaded'].attrs['units'].split(',')[0] + ' (‰)')
    if i != 1:
        ax[i].set_title('')
ax[0].set_ylabel('z (m)')
ax[0].legend(['sunlit', 'shaded'])
fig.tight_layout()

var = [['pt_d13c_leaf_sugar'],['pt_d18o_leaf_sugar','pt_d18o_leaf_water']]
var2 = ['pt_d13c_C_pool','pt_d18o_C_pool','pt_C_pool']
var3 = ['pt_d13c_treering_celluose','pt_d18o_treering_celluose']

fig, ax = plt.subplots(3,1, figsize=(10,10), sharex='all')
for i, vv in enumerate(var):
    for v in vv:
        results[v+'_sunlit'][:,0,0,50].plot(x='date',ax=ax[i],label=results[v+'_sunlit'].attrs['units'])
        results[v+'_shaded'][:,0,0,50].plot(x='date',ax=ax[i],label=results[v+'_shaded'].attrs['units'])
for i, v in enumerate(var2):
    results[v][:,0,0].plot(x='date',ax=ax[i],label=results[v].attrs['units'])
for i, v in enumerate(var3):
    results[v][:,0,0].plot(x='date',ax=ax[i],label=results[v].attrs['units'], marker='o', linestyle='')
    ax[i].set_xlabel('')
    ax[i].set_ylabel('')
    ax[i].set_title('')
    ax[i].legend()
fig.tight_layout()

results.close()