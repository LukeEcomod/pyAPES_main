import pandas as pd
import numpy as np

Nsim = 2 # Number of simulations

snow_model = tuple(['fsm2', 'degreeday'])

degero_snow_parameters = {
    'count': Nsim, # Number of simulations
    'scenario': 'degero_snow',
    'canopy': {
        'forestfloor': {
            'snowpack': {
                'snow_model': snow_model
            }
        }
    }
}