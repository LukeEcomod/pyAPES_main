# -*- coding: utf-8 -*-
"""
Created on Tue Oct 09 16:31:25 2018

@author: Samuli Launiainen
"""

# to show figures in qt (pop-up's!)
# write in console:
# import matplotlib.pyplot
# %matplotlib qt

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from pyAPES import driver
from parameters.parameter_tools import get_parameter_list
from tools.iotools import read_results
from tools.iotools import read_forcing, read_data
from pyAPES_utilities.plotting import plot_fluxes

# Get parameters and forcing for SMEAR II -site

from parameters.Lettosuo import gpara, cpara, spara

#  wrap parameters in dictionary
params = {
    'general': gpara,
    'canopy': cpara,
    'soil': spara
    }

# parameters simulation(s)
params = get_parameter_list(params, 'test')

# run model
outputfile, Model = driver(parameters=params, create_ncf=True)

# read results
results = read_results(outputfile)

# import fluxdata
flxdata = read_data('forcing/Lettosuo/Lettosuo_EC_2010_2018.csv', sep=',',
                     start_time=results.date[0].values, end_time=results.date[-1].values)
flxdata.index = flxdata.index - pd.Timedelta(hours=0.5)  # period end
for trmt in ['control_','partial_']:
    flxdata[trmt+'NEE'] *= 1.0 / 44.01e-3
    flxdata[trmt+'Reco'] *= 1.0 / 44.01e-3
    flxdata[trmt+'GPP'] = flxdata[trmt+'Reco'] - flxdata[trmt+'NEE']

results['ground_heat_flux'] = results['soil_heat_flux'].isel(soil=6).copy()

plot_fluxes(results, flxdata, norain=True,
            res_var=['canopy_Rnet','canopy_SWnet','canopy_SH','canopy_LE','ground_heat_flux',
                      'canopy_NEE','canopy_GPP','canopy_Reco'],
            Data_var=['control_NRAD','control_NSWRAD','control_SH','control_LE','control_GHF',
                      'control_NEE','control_GPP','control_Reco'],
            fmonth=5, lmonth=9, sim_idx=0)

plt.figure()
results['soil_ground_water_level'].plot.line(x='date')