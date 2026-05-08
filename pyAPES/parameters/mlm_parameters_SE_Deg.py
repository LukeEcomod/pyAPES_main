# -*- coding: utf-8 -*-
"""
.. module: pyAPES.parameters.mlm_parameters_SE_Deg
    :synopsis: pyAPES_MLM PARAMETERIZATION FOR FluxNet site SE-Deg (Degerö Stormyr, Sweden)
.. moduleauthor:: Samuli Launiainen

Define pyAPES_MLM parameters and forcing file here.

Define pyAPES_MLM output variables and logger config in: parameters.mlm_outputs

"""

import numpy as np
import os
import pathlib
from pyAPES.utils.utilities import lad_weibul, lad_constant

from dotenv import load_dotenv
load_dotenv()
pyAPES_main_folder = os.getenv('pyAPES_main_folder')
# model forcing: see Demo_creating_model_forcing_Degero.ipynb
forcing_file = pathlib.Path(fr'{pyAPES_main_folder}/forcing/Degero/Degero_forcing_2014-2016.dat')

#**************** PARAMETER DICTIONARIES ****************************

gpara = {'dt' : 1800.0,  # timestep in forcing data file [s]
         'start_time' : "2016-06-01",  # start time of simulation [yyyy-mm-dd]
         'end_time' : "2016-06-30",  # end time of simulation [yyyy-mm-dd]
         'start_doy': 150,
         'forc_filename' : forcing_file,  # forcing data file
         'results_directory': 'results/',  # This is given relative to pyAPES main folder or if not in .env then current working directory
         'logging_directory': 'logs/',  # This is also given similar to results_directory
         'parameters_directory': 'inputs/'  # This is also given similar to results_directory
         }

# --- Model control flags
ctr = {'Eflow': True,  # use ensemble flow statistics; i.e fixed ratio of Utop/ustar.
       'WMA': False,  # assume air-space scalar profiles well-mixed
       'Ebal': True,  # computes leaf and surface temperature by solving energy balance
       }

# site location
loc = {
    'lat': 64.11,  # latitude
    'lon': 19.33  # longitude
}

# grid
grid = {'zmax': 10.0,  # heigth [m] of grid from ground surface
        'Nlayers': 20  # number of layers [-]
        }

z = np.linspace(0, grid['zmax'], grid['Nlayers'])  # grid [m] above ground

# --- Turbulent flow & scalar transport in air-space: pyAPES.microclimate.micromet.Micromet
micromet = {
    'zos': 0.01,  # forest floor roughness length [m]  -- not used?
    'dPdx': 0.0,  # horizontal pressure gradient
    'Cd': 0.15,  # drag coefficient
    'Utop': 9.0,  # ensemble U/ustar
    'Ubot': 0.01,  # lower boundary
            'Sc': {'T': 2.0, 'H2O': 2.0, 'CO2': 2.0}  # turbulent Schmidt numbers in canopy flow
            }

# --- Short- and long-wave radiation: pyAPES.microclimate.radiation.Radiation
radiation = {'SWmodel': 'ZHAOQUALLS',
             'LWmodel': 'ZHAOQUALLS',
             'clump': 0.9,  # clumping index [-]
             'leaf_angle': 1.0,  # leaf-angle distribution [-]
             'Par_alb': 0.12,  # shoot Par-albedo [-]
             'Nir_alb': 0.55,  # shoot NIR-albedo [-]
             'leaf_emi': 0.98  # leaf emissivity [-]
             }

# --- Rainfall and snow interception: pyAPES.microclimate.interception.Interception
interception = {'wmax': 0.2,  # maximum interception storage capacity for rain [kg m-2 (leaf), i.e. mm H2O m-2 (leaf)]
                'wmaxsnow': 0.8,  # maximum interception storage capacity for snow [kg m-2 (leaf), i.e. mm H2O m-2 (leaf)]
                'w_ini': 0.0,  # initial canopy storage, fraction of maximum rain storage [-]
                'Tmin': 0.0,  # temperature below which all is snow [degC]
                'Tmax': 2.0,  # temperature above which all is water [degC]
                'leaf_orientation': 0.5, # leaf orientation factor for randomdly oriented leaves
                }

# --- PlantTypes and their leaf-scale properties: pyAPES.planttype.PlantType
#       Define here pt's and then append to list.
#       Note: tree height h must be less than uppermost model gridpoint: grid['zref']

pt1 = {
    'name': 'sedges',
    'ctr': {
        'WaterStress': 'Rew',  # How soil water limitations are accounted for: 'Rew' |'PsiL' | None
        'seasonal_LAI': True,  # account for seasonal LAI dynamics
        'pheno_cycle': None #'deciduous',  # account for seasonal Vcmax25, Jmax25 dynamics
        },
    'LAImax': 0.6, # maximum annual LAI m2m-2
    'lad': lad_constant(z, LAI=1.0, h=0.5),  # leaf-area density m2m-3
    # seasonal cycle of photosynthetic activity: pyAPES.planttype.phenology.Photo_cycle

    # seasonal cycle of LAI: pyAPES.planttype.phenology.LAI_cycle
    'laip': {
        'lai_min': 0.8,
        'lai_ini': None,
        'DDsum0': 0.0,
        'Tbase': 5.0,
        'ddo': 45.0,
        'ddmat': 250.0,
        'sdl': 12.0,
        'sdur': 30.0
    },
    # A-gs model: pyAPES.leaf.photo
    'photop': {
        'Vcmax': 45.0,
        'Jmax': 72.0,  # 1.6*Vcmax (Kattge and Knorr, 2007)
        'Rd': 0.7,  # 0.015*Vcmax
        'tresp': { # temperature response parameters (Kattge and Knorr, 2007)
            'Vcmax': [72., 200., 649.],
            'Jmax': [50., 200., 646.],
            'Rd': [33.0]
        },
        'alpha': 0.2,   # quantum efficiency parameter [-]
        'theta': 0.7,   # curvature parameter [-]
        'beta': 0.95,   # co-limitation parameter [-]
        'g1': 4.0,      # USO-model stomatal slope kPa^(0.5)
        'g0': 5.0e-3,   # residual conductance for CO2 [mol m-2 s-1]
        'kn': 0.5,      # nitrogen attenuation coefficient [-]
        'drp': [0.39, 0.83, 0.31, 3.0], # Rew-based drought response parameters
        # growth respiration: Rg25 = construction cost [umol CO2 m-2 leaf]
        'Rg25': 1.5e6,   # [umol CO2 m-2 leaf]
        'Q10g': 2.0,  # temperature sensitivity of growth respiration [-]
    },
    'leafp': {
        'lt': 0.02,     # leaf length scale [m]
    },
    # root zone: pyAPES.planttype.rootzone.RootUptake
    'rootp': {
        'root_depth': 0.2, # rooting depth [m]
        'beta': 0.943, # root distribution shape parameter [-]
        'root_to_leaf_ratio': 2.0, # fine-root to leaf-area ratio [-]
        'root_radius': 2.0e-3, # [m]
        'root_conductance': 5.0e8, # [s]
    }
}

pt2 = { 'name': 'pine',
        'ctr': {
            'WaterStress': 'Rew',  # How soil water limitations are accounted for: 'Rew' |'PsiL' | None
            'seasonal_LAI': False,  # account for seasonal LAI dynamics
            'pheno_cycle': 'conifer',  # account for seasonal Vcmax25, Jmax25 dynamics
            },
        'LAImax': 0.2, # maximum annual LAI m2m-2
        'lad': lad_weibul(z, LAI=1.0, h=5.0, hb=1.0, species='pine'),  # leaf-area density m2m-3
        # seasonal cycle of photosynthetic activity: pyAPES.planttype.phenology.Photo_cycle
        'phenop': {
            'Xo': 0.0,
            'fmin': 0.1,
            'Tbase': -4.67,  # Kolari 2007
            'tau': 8.33,  # Kolari 2007
            'smax': 18.0  # Kolari 2014
            },
        # seasonal cycle of LAI: pyAPES.planttype.phenology.LAI_cycle
        'laip': {
            'lai_min': 0.8,
            'lai_ini': None,
            'DDsum0': 0.0,
            'Tbase': 5.0,
            'ddo': 45.0,
            'ddmat': 250.0,
            'sdl': 12.0,
            'sdur': 30.0
            },
        # A-gs model: pyAPES.leaf.photo
        'photop': {
            'Vcmax': 40.0,
            'Jmax': 64.0,  
            'Rd': 0.6,  
            'tresp': { # temperature response parameters (Kattge and Knorr, 2007)
                'Vcmax': [78., 200., 649.],
                'Jmax': [56., 200., 646.],
                'Rd': [33.0]
                },
            'alpha': 0.2,   # quantum efficiency parameter [-]
            'theta': 0.7,   # curvature parameter [-]
            'beta': 0.95,   # co-limitation parameter [-]
            'g1': 2.5,      # USO-model stomatal slope kPa^(0.5)
            'g0': 1.0e-3,   # residual conductance for CO2 [mol m-2 s-1]
            'kn': 0.5,      # nitrogen attenuation coefficient [-]
            'drp': [0.39, 0.83, 0.31, 3.0], # Rew-based drought response parameters
            # growth respiration: Rg25 = construction cost [umol CO2 m-2 leaf]
            'Rg25': 1.5e6,   # [umol CO2 m-2 leaf]
            'Q10g': 2.0,  # temperature sensitivity of growth respiration [-]
            },
        'leafp': {
            'lt': 0.02,     # leaf length scale [m]
            },
        # root zone: pyAPES.planttype.rootzone.RootUptake
        'rootp': {
            'root_depth': 0.5, # rooting depth [m]
            'beta': 0.943, # root distribution shape parameter [-]
            'root_to_leaf_ratio': 2.0, # fine-root to leaf-area ratio [-]
            'root_radius': 2.0e-3, # [m]
            'root_conductance': 5.0e8, # [s]
            }
        }


# --- forestfloor: pyAPES.canopy.forestfloor.ForestFloor combines snowpack, soil, and organiclayer types.

# --- pyAPES.snow.snowpack.DegreeDaySnow
snowpack = {
    'kmelt': 2.31e-5,  # Melting coefficient [kg m-2 s-1 degC-1] (=2.0 mm/C/d)
    'kfreeze': 5.79e-6,  # Freezing  coefficient [kg m-2 s-1 degC-1] (=0.5 mm/C/d)
    'retention': 0.2,  # max fraction of liquid water in snow [-]
    'Tmelt': 0.0,  # temperature when melting starts [degC]
    'optical_properties': {
        'emissivity': 0.97,
        'albedo': {'PAR': 0.8, 'NIR': 0.8}
    },
    'initial_conditions': {
        'temperature': 0.0,
        'snow_water_equivalent': 0.0}
}

# --- pyAPES.bottomlayer.carbon.SoilRespiration
soil_respiration = {
    'r10': 2.5, # base rate (bulk heterotrophic + autotrophic) [umol m-2 (ground) s-1]
    'q10': 2.0, # temperature sensitivity [-]
    'moisture_coeff': [3.11, 2.42],  # moisture response; Moyano et al. 2013 Eq. 1 "generic"
    'beta': 1.0 # exponential decay for potential soil respiration depth profile
}

# Note: renewed bryophyte parameters


# this is general Sphagnum parametrisation based on literature review
Sphagnum = {
    'name': 'Sphagnum sp.',
    'layer_type': 'bryophyte',
    'coverage': 1.0,
    'height': 0.06, # [0.044, 0.076]
    'roughness_height': 0.02,
    'bulk_density': 35.1,  # [9.28, 46.7]
    'max_water_content': 17.8,  # [15.6, 24.4]
    'water_content_ratio': 0.43,  # max_symplast_water_content:max_water_content -ratio
    'min_water_content': 0.1,
    'porosity': 0.98,

    'photosynthesis': { # farquhar-parameters
        'Vcmax': 45.0, 'Jmax': 85.5, 'Rd': 1.35, # umolm-2s-1
        'alpha': 0.3, 'theta': 0.8, 'beta': 0.9, # quantum yield, curvature, co-limitation
        'gref': 0.04, 'wref': 7.0, 'a0': 0.7, 'a1': -0.263, 'CAP_desic': [0.58, 10.0],
        'tresp': {
            'Vcmax': [69.83, 200.0, 27.56],
            'Jmax': [100.28, 147.92, 19.8],
            'Rd': [33.0]
        }
    },
    'optical_properties': { # moisture responses are hard-coded
        'emissivity': 0.98,
        'albedo': {'PAR': 0.10, 'NIR': 0.27} # albedos when fully hydrated [-]
    },
    'water_retention': {
        'alpha': 0.381,  # based on fitted value
        'n': 1.781,  # based on fitted value
        'saturated_conductivity': 2.88e-4,  # [m s-1], based on fitted value
        'pore_connectivity': -2.27  # based on fitted value
    },
    'initial_conditions': {
        'temperature': 10.0,
        'water_content': 10.0
    }
}
# --- compile forestfloor parameter dictionary

forestfloor = {
    'bottom_layer_types': {
        'Sphagnum': Sphagnum,
    },
    'snowpack': snowpack,
    'soil_respiration': soil_respiration
}

# --- compile canopy-model parameter dictionary

cpara = {'loc': loc,
         'ctr': ctr,
         'grid': grid,
         'radiation': radiation,
         'micromet': micromet,
         'interception': interception,
         'planttypes': {'sedges': pt1, 'pine': pt2},
         'forestfloor': forestfloor
         }

# --- Soil water & heat: pyAPES.soil.Soil

# grid and soil properties: pF and conductivity values for Degerö Stormyr peat profile
soil_grid = {  # thickness of computational layers [m]: 0.01 m until 0.1m, 0.02m until 0.3m, 0.05m until 1.0m, 0.1m until 2m depth
            'dz': [0.01] * 10 + [0.02] * 10 + [0.05] * 14 + [0.1] * 10,
            # bottom depth of layers with different characteristics [m]
            'zh': [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1., -1.5, -2.0]
            }

soil_properties = {
    'pF': {  # vanGenuchten water retention parameters
           'ThetaS': [0.945, 0.945, 0.945, 0.918, 0.918, 0.918, 0.918, 0.918, 0.918, 0.918, 0.918, 0.918],  # [m3m-3]
           'ThetaR': [0.098, 0.098, 0.098, 0.098, 0.098, 0.098, 0.098, 0.098, 0.098, 0.098, 0.098, 0.098],  # [m3m-3]
           'alpha': [0.338, 0.338, 0.338, 0.072, 0.072, 0.072, 0.072, 0.072, 0.072, 0.072, 0.072, 0.072],  # [cm-1]
           'n': [1.402, 1.402, 1.402, 1.371, 1.371, 1.371, 1.371, 1.371, 1.371, 1.371, 1.371, 1.371]  # [-]
    },
    'saturated_conductivity_vertical': [9E-05, 3E-05, 1E-05, 3E-06, 1E-06, 3E-07, 1E-07, 1E-07, 1E-07, 1E-07, 1E-07, 1E-07],  # [m s-1]
    'saturated_conductivity_horizontal': [9E-05, 3E-05, 1E-05, 3E-06, 1E-06, 3E-07, 1E-07, 1E-07, 1E-07, 1E-07, 1E-07, 1E-07],  # [m s-1]
    'solid_heat_capacity': None,  # [J m-3 (solid) K-1] - if None, estimated from organic/mineral composition
    'solid_composition': {  # fraction of solids
        'organic': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        'sand':    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'silt':    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'clay':    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
    'freezing_curve': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # freezing curve parameter
    'bedrock': {
        'solid_heat_capacity': 2.16e6,  # [J m-3 (solid) K-1]
        'thermal_conductivity': 3.0  # thermal conductivity of non-porous bedrock [W m-1 K-1]
    }
}

# --- water model: pyAPES.soil.water.Water
water_model = {'solve': True,
               'type': 'Equilibrium',  # solution approach 'Richards' | 'Equilibrium'
               'pond_storage_max': 0.002,  #  maximum pond depth [m]
               'initial_condition': {
                       'ground_water_level': -0.05,  # groundwater depth [m], <=0
                       'pond_storage': 0.0  # pond depth at surface [m]
                       },
               'lower_boundary': {
                       'type': 'impermeable',
                       'value': None,
                       'depth': -2.0
                       },
               'drainage_equation': {
                       'type': 'Hooghoudt',
                       'depth': 0.1,  # drain depth [m]
                       'spacing': 100.0,  # drain spacing [m]
                       'width': 1.0,  # drain width [m]
                       }
                }

# --- heat model: pyAPES.soil.heat.Heat
T_ini = 2.0
heat_model = {'solve': True,
              'initial_condition': {
                      'temperature': T_ini,  # initial soil temperature [degC], assumed constant with depth - can also be array of correct length
                      },
              'lower_boundary': {  # lower boundary condition (type, value)
                      'type': 'temperature',
                      'value': 2.0
                      },
              }

# --- compile soil model parameter dictionary
spara = {'grid': soil_grid,
         'soil_properties': soil_properties,
         'water_model': water_model,
         'heat_model': heat_model}

