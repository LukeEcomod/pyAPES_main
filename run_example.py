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

from parameters.SmearII import gpara, cpara, spara

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
flxdata = read_data("forcing/Hyytiala/FIHy_flx_2005-2010.dat", sep=';',
                       start_time=results.date[0].values, end_time=results.date[-1].values)

plot_fluxes(results, flxdata, norain=True,
            res_var=['canopy_Rnet','canopy_SH','canopy_LE',
                      'canopy_NEE','canopy_GPP','canopy_Reco'],
            Data_var=['Rnet','H','LE','NEE','GPP','Reco'],
            fmonth=5, lmonth=9, sim_idx=0)