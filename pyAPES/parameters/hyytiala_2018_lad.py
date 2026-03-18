import pandas as pd
import numpy as np

Nsim = 10 # Number of simulations

# The original LAImax are pine=2.1; spruce=1.2; decid=1.2; shrubs=0.7
# Keep the same ratio 1.75:1 for pine:spruce or decid and 3:1 for pine:shrubs.
# i.e., ca. 40% pine 23% spruce and decid and 13% shrubs

LAI_original = np.array([2.1, 1.2, 1.2, 0.7])
LAI_original_total = np.sum(LAI_original)
LAI_ratio = LAI_original/LAI_original_total

# Create an array of increasing total LAI from 0.5 to 5
LAI_total = np.linspace(0.5, 5, Nsim)
LAI_pine = tuple(LAI_total * LAI_ratio[0])
LAI_spruce = tuple(LAI_total * LAI_ratio[1])
LAI_decid = tuple(LAI_total * LAI_ratio[2])
LAI_shrubs = tuple(LAI_total * LAI_ratio[3])

hyytiala_2018_lad_parameters = {
    'count': Nsim, # Number of simulations
    'scenario': 'hyytiala_2018_lad',
    'canopy': {
        'planttypes': {
            'pine': {
                'LAImax': LAI_pine,
            },
            'spruce': {
                'LAImax': LAI_spruce
            },
            'decid': {
                'LAImax': LAI_decid
            },
            'shrubs': {
                'LAImax': LAI_shrubs
            },
        }
    }
}