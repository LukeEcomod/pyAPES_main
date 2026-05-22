#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 12:39:09 2018

@author: ajkieloaho
"""

# Modifications to some parameters

def get_parameters(scenario):
    # spefify as one values (same for all simulations) or tuple of length 'count'
    if scenario.upper() == 'TEST':
        parameters = {
            'count': 2,
            'scenario': 'test',
            'general':{
                'start_time' : "2005-06-01",
                'end_time' : "2005-06-10",
                'forc_filename' : "Hyytiala/FIHy_forcing_2005-2010.dat"
            },
            'canopy': {
                'planttypes': {
                    'decid': {
                        'LAImax': (1.2, 3.0),
                    },
                },
            },
        }
        return parameters

    elif scenario.upper() == 'HYYTIALA_2018_LAD':
        from pyAPES.parameters.hyytiala_2018_lad import hyytiala_2018_lad_parameters
        return hyytiala_2018_lad_parameters
    
    elif scenario.upper() == 'DEGERO_SNOW':
        from pyAPES.parameters.degero_snow import degero_snow_parameters
        return degero_snow_parameters

    else:
        raise ValueError("Unknown parameterset!")

# EOF