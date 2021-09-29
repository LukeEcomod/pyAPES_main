#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 12:39:09 2018

@author: ajkieloaho
"""
from pyAPES_utilities.parameter_utilities import single_lad_profiles
from parameters.Lettosuo import grid

hs = 0.5  # height of understory shrubs [m]
control = single_lad_profiles(grid, 'pyAPES_utilities/runkolukusarjat/letto2014.txt',
                              hs, plot=False, biomass_function='aleksis_combination')
partial = single_lad_profiles(grid, 'pyAPES_utilities/runkolukusarjat/letto2016_partial.txt',
                              hs, plot=False, biomass_function='aleksis_combination')

# Modifications to some parameters

def get_parameters(scenario):
    # spefify as one values (same for all simulations) or tuple of length 'count'
    if scenario.upper() == 'TEST':
        parameters = {
            'count': 1,
            'scenario': 'test',
            'general':{
                'start_time' : "2010-06-01",
                'end_time' : "2010-06-10",
            },
            'canopy': {
                'planttypes': {
                    'pine': {
                        'LAImax': (control['lai']['pine'], partial['lai']['pine']),
                        'lad': (control['lad']['pine'], partial['lad']['pine']),
                    },
                    'spruce': {
                        'LAImax': (control['lai']['spruce'], partial['lai']['spruce']),
                        'lad': (control['lad']['spruce'], partial['lad']['spruce']),
                    },
                    'decid': {
                        'LAImax': (control['lai']['decid'], partial['lai']['decid']),
                        'lad': (control['lad']['decid'], partial['lad']['decid']),
                    },
                    'shrubs': {
                        'LAImax': (1.0, 0.8),
                        'lad': (control['lad']['shrubs'], partial['lad']['shrubs']),
                    },
                },
            },
        }

        return parameters

    else:
        raise ValueError("Unknown parameterset!")

# EOF