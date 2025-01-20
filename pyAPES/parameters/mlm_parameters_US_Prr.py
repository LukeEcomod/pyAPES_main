# -*- coding: utf-8 -*-
"""
.. module: pyAPES.parameters.mlm_parameters
    :synopsis: pyAPES_MLM PARAMETERIZATION FOR FluxNet site US-Prr
.. moduleauthor:: Samuli Launiainen

Define pyAPES_MLM parameters and forcing file here.

Define pyAPES_MLM output variables and logger config in: parameters.mlm_outputs

"""

import numpy as np
import pandas as pd
from pyAPES.utils.utilities import lad_weibul, lad_constant

# model forcing: see Ex1_creating_model_forcing.ipynb
forcing_file = 'c:/Repositories/pyAPES_main/forcing/US-Prr/US-Prr_forcing_2011_2015.dat'
# vertical leaf-area density profiles for small and large Black Spruce: see Demo0_ecosystem_structure_US-Prr.ipynb
lad_file = 'c:/Repositories/pyAPES_main/forcing/US-Prr/BlackSpruce_relative_lad_two_cohorts.csv'

#**************** PARAMETER DICTIONARIES ****************************

gpara = {'dt' : 1800.0,  # timestep in forcing data file [s]
         'start_time' : "2011-06-01",  # start time of simulation [yyyy-mm-dd]
         'end_time' : "2011-06-15",  # end time of simulation [yyyy-mm-dd]
         'forc_filename' : 'c:/Repositories/pyAPES_main/forcing/US-Prr/US-Prr_forcing_2011_2015.dat', # forcing data file
         'results_directory':'results/'
         }

# --- Model control flags
ctr = {'Eflow': True,  # use ensemble flow statistics; i.e fixed ratio of Utop/ustar.
       'WMA': False,  # assume air-space scalar profiles well-mixed
       'Ebal': True,  # computes leaf temperature by solving energy balance
       'WaterStress': 'Rew',  # How soil water limitations are accounted for: 'Rew' |'PsiL' | None
       'seasonal_LAI': True,  # account for seasonal LAI dynamics
       'pheno_cycle': True  # account for phenological cycle
       }

# site location
loc = {'lat': 65.12,  # latitude, decimal deg
       'lon': -147.49  # longitude, decimal deg; negative lon west 
       }

# grid
grid = {'zmax': 11.0,  # heigth [m] of grid from ground surface. Corresponds to height of forcing data, here EC-setup
        'Nlayers': 50  # number of layers [-]
        }

z = np.linspace(0, grid['zmax'], grid['Nlayers'])  # grid [m] above ground

# --- Turbulent flow & scalar transport in air-space: pyAPES.microclimate.micromet.Micromet
micromet = {'zos': 0.01,  # forest floor roughness length [m]  -- not used?
            'dPdx': 0.0,  # horizontal pressure gradient [units]
            'Cd': 0.15,  # drag coefficient [-]
            'Utop': 5.0,  # ensemble U/ustar [-]
            'Ubot': 0.01,  # flow at lower boundary [m s-1 or -]
            'Sc': {'T': 2.0, 'H2O': 2.0, 'CO2': 2.0}  # turbulent Schmidt numbers in canopy flow
            }

# --- Short- and long-wave radiation: pyAPES.microclimate.radiation.Radiation
radiation = {'clump': 0.7,  # clumping index [-]
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

# now read lad-file. # large spruce contributes 1/3 and small (<3.5m) spruce 2/3 to total overstory LAI
spruce_lad = pd.read_csv(lad_file, sep=';') # columns: z, small_spruce, large_spruce, total
z0 = spruce_lad['z'].values
lad_sp1 = np.interp(z, z0, spruce_lad['large_spruce'].values)
lad_sp2 = np.interp(z, z0, spruce_lad['small_spruce'].values)

pt1 = { 'name': 'Spruce large',
        'LAImax': 0.4, # maximum annual LAI m2m-2
        'lad': lad_sp1, # normalized leaf area density profile [m2m-3] (sum(lad) * dz = 1)       
        
        # Weibul-distribution library for different speceis: Teske, M.E., and H.W. Thistle, 2004, A library of forest canopy structure for 
        # use in interception modeling. Forest Ecology and Management, 198, 341-350. 
        #'lad': lad_weibul(z, LAI=1.0, h=6.0, hb=3.0, species='pine'),  # leaf-area density m2m-3

        # seasonal cycle of photosynthetic activity: pyAPES.planttype.phenology.Photo_cycle
        'phenop': {
            'Xo': 0.0, # initial delayed temperature [degC]
            'fmin': 0.1, # minimum relative photosynthetic capacity
            'Tbase': -4.7,  # base temperature [degC]
            'tau': 8.33,  # time constant [d]
            'smax': 18.0  # threshold for full acclimation [degC]
            },
        # seasonal cycle of LAI: #  pyAPES.planttype.phenology.LAI_cycle
        'laip': {
            'lai_min': 0.8, # minimum LAI, fraction of annual maximum [-]
            'lai_ini': None, # initial LAI, if None lai_ini = Lai_min * LAImax
            'DDsum0': 0.0, # initial degree-day sum [degC]
            'Tbase': 5.0, # base temperature for degree-day sy
            'ddo': 45.0, # degree-days at bud burst [days]
            'ddmat': 250.0, #degreedays at full maturation [days]
            'sdl': 12.0, # day length [h] for starting autumn senecence
            'sdur': 30.0 # duration [d] of senescence
            },
        # A-gs model: pyAPES.leaf.photo
        'photop': {
            'Vcmax': 35.0, # maximum carboxylation rate [umol m-2 (leaf) s-1] at 25 degC
            'Jmax': 60.0,  # maximum electron transport rate[umol m-2 (leaf) s-1] at 25 degC1.97*Vcmax (Kattge and Knorr, 2007)
            'Rd': 0.8, # dark respiration rate [umol m-2 (leaf) s-1] at 25 degC
            'tresp': { # temperature response (Kattge and Knorr, 2007)
                'Vcmax': [78., 200., 649.], # [activation energy [kJ mol-1], deactivation energy [kJ mol-1]
                                 #             entropy factor [kJ mol-1]]
                'Jmax': [56., 200., 646.],
                'Rd': [33.0]
                },
            'alpha': 0.2,   # quantum efficiency parameter [-]
            'theta': 0.7,   # curvature parameter [-]
            'beta': 0.95,   # co-limitation parameter [-]
            'g1': 2.5,      # USO-model stomatal slope kPa^(0.5)
            'g0': 5.0e-3,   # residual conductance for CO2 [mol m-2 s-1]
            'kn': 0.5,      # nitrogen attenuation coefficient; affects Vcmax, Jmax, Rd profile in PlantType [-]
            'drp': [0.39, 0.83, 0.31, 3.0] # Rew-based drought response parameters.
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

pt2 = { 'name': 'Spruce small',
        'LAImax': 0.8, # maximum annual LAI m2m-2
        'lad': lad_sp2, # normalized leaf area density profile [m2m-3] (sum(lad) * dz = 1)       
        # lad_weibul provides species lad-profiles from Teske & Thistle
        #'lad': lad_weibul(z, LAI=1.0, h=6.0, hb=3.0, species='pine'),  # leaf-area density m2m-3

        # seasonal cycle of photosynthetic activity: pyAPES.planttype.phenology.Photo_cycle
        'phenop': {
            'Xo': 0.0, # initial delayed temperature [degC]
            'fmin': 0.1, # minimum relative photosynthetic capacity
            'Tbase': -4.7,  # base temperature [degC]
            'tau': 8.33,  # time constant [d]
            'smax': 18.0  # threshold for full acclimation [degC]
            },
        # seasonal cycle of LAI: #  pyAPES.planttype.phenology.LAI_cycle
        'laip': {
            'lai_min': 0.8, # minimum LAI, fraction of annual maximum [-]
            'lai_ini': None, # initial LAI, if None lai_ini = Lai_min * LAImax
            'DDsum0': 0.0, # initial degree-day sum [degC]
            'Tbase': 5.0, # base temperature for degree-day sy
            'ddo': 45.0, # degree-days at bud burst [days]
            'ddmat': 250.0, #degreedays at full maturation [days]
            'sdl': 12.0, # day length [h] for starting autumn senecence
            'sdur': 30.0 # duration [d] of senescence
            },
        # A-gs model: pyAPES.leaf.photo
        'photop': {
            'Vcmax': 35.0, # maximum carboxylation rate [umol m-2 (leaf) s-1] at 25 degC
            'Jmax': 60.0,  # maximum electron transport rate[umol m-2 (leaf) s-1] at 25 degC1.97*Vcmax (Kattge and Knorr, 2007)
            'Rd': 0.8, # dark respiration rate [umol m-2 (leaf) s-1] at 25 degC
            'tresp': { # temperature response (Kattge and Knorr, 2007)
                'Vcmax': [78., 200., 649.], # [activation energy [kJ mol-1], deactivation energy [kJ mol-1]
                                 #             entropy factor [kJ mol-1]]
                'Jmax': [56., 200., 646.],
                'Rd': [33.0]
                },
            'alpha': 0.2,   # quantum efficiency parameter [-]
            'theta': 0.7,   # curvature parameter [-]
            'beta': 0.95,   # co-limitation parameter [-]
            'g1': 2.5,      # USO-model stomatal slope kPa^(0.5)
            'g0': 5.0e-3,   # residual conductance for CO2 [mol m-2 s-1]
            'kn': 0.5,      # nitrogen attenuation coefficient; affects Vcmax, Jmax, Rd profile in PlantType [-]
            'drp': [0.39, 0.83, 0.31, 3.0] # Rew-based drought response parameters.
            },
        'leafp': {
            'lt': 0.02,     # leaf length scale [m]
            },

        # root zone: pyAPES.planttype.rootzone.RootUptake
        'rootp': {
            'root_depth': 0.3, # rooting depth [m]
            'beta': 0.943, # root distribution shape parameter [-]
            'root_to_leaf_ratio': 2.0, # fine-root to leaf-area ratio [-]
            'root_radius': 2.0e-3, # [m]
            'root_conductance': 5.0e8, # [s]
            }
        }

pt3 = { 'name': 'Understory',
        'LAImax': 0.8, # maximum annual LAI m2m-2
        'lad': lad_constant(z, LAI=1.0, h=0.6, hb=0.0),  # leaf-area density [m2 m-3]
        # seasonal cycle of photosynthetic activity: pyAPES.planttype.phenology.Photo_cycle
        'phenop': {
            'Xo': 0.0, # initial delayed temperature [degC]
            'fmin': 0.1, # minimum relative photosynthetic capacity
            'Tbase': -4.7,  # base temperature [degC]
            'tau': 8.33,  # time constant [d]
            'smax': 18.0  # threshold for full acclimation [degC]
            },
        # seasonal cycle of LAI: #  pyAPES.planttype.phenology.LAI_cycle
        'laip': {
            'lai_min': 0.1, # minimum LAI, fraction of annual maximum [-]
            'lai_ini': None, # initial LAI, if None lai_ini = Lai_min * LAImax
            'DDsum0': 0.0, # initial degree-day sum [degC]
            'Tbase': 5.0, # base temperature for degree-day sy
            'ddo': 45.0, # degree-days at bud burst [days]
            'ddmat': 250.0, #degreedays at full maturation [days]
            'sdl': 12.0, # day length [h] for starting autumn senecence
            'sdur': 30.0 # duration [d] of senescence
            },
        # A-gs model: pyAPES.leaf.photo
        'photop': {
            'Vcmax': 50.0, # maximum carboxylation rate [umol m-2 (leaf) s-1] at 25 degC
            'Jmax': 85.0,  # maximum electron transport rate[umol m-2 (leaf) s-1] at 25 degC
            'Rd': 0.8, # dark respiration rate [umol m-2 (leaf) s-1] at 25 degC
            'tresp': { # temperature response (Kattge and Knorr, 2007)
                'Vcmax': [78., 200., 649.], # [activation energy [kJ mol-1], deactivation energy [kJ mol-1]
                                 #             entropy factor [kJ mol-1]]
                'Jmax': [56., 200., 646.],
                'Rd': [33.0]
                },
            'alpha': 0.2,   # quantum efficiency parameter [-]
            'theta': 0.7,   # curvature parameter [-]
            'beta': 0.95,   # co-limitation parameter [-]
            'g1': 2.3,      # USO-model stomatal slope kPa^(0.5)
            'g0': 5.0e-3,   # residual conductance for CO2 [mol m-2 s-1]
            'kn': 0.5,      # nitrogen attenuation coefficient; affects Vcmax, Jmax, Rd profile in PlantType [-]
            'drp': [0.39, 0.83, 0.31, 3.0] # Rew-based drought response parameters.
            },
        'leafp': {
            'lt': 0.02,     # leaf length scale [m]
            },

        # root zone: pyAPES.planttype.rootzone.RootUptake
        'rootp': {
            'root_depth': 0.3, # rooting depth [m]
            'beta': 0.943, # root distribution shape parameter [-]
            'root_to_leaf_ratio': 2.0, # fine-root to leaf-area ratio [-]
            'root_radius': 2.0e-3, # [m]
            'root_conductance': 5.0e8, # [s]
            }
        }

# --- forestfloor: pyAPES.canopy.forestfloor.ForestFloor combines snowpack, soil, and organiclayer types.

# --- pyAPES.snow.snowpack.DegreeDaySnow
snowpack = {
        'kmelt': 2.31e-5,  # Melting coefficient [kg m-2 s-1 degC-1]; (= 2.0 mm degC d-1)
        'kfreeze': 5.79e-6,  # Freezing  coefficient [kg m-2 s-1 degC-1] (=0.5 mm degC d-1)
        'retention': 0.2,  # max fraction of liquid water in snow [-]
        'Tmelt': 0.0,  # temperature when melting starts [degC]
        'optical_properties': {
                'emissivity': 0.97,
                'albedo': {'PAR': 0.8, 'NIR': 0.8}
                },
        'initial_conditions': {'temperature': 0.0,
                               'snow_water_equivalent': 0.0}
        }

# --- pyAPES.bottomlayer.carbon.SoilRespiration
soil_respiration = {
        'r10': 2.5, # base rate (bulk heterotrophic + autotrophic) [umol m-2 (ground) s-1]
        'q10': 2.0, # temperature sensitivity [-]
        'moisture_coeff': [0.1, -0.28, 4.325, -3.65], # [minimum_relative_respiration,  p[0], p[1], p[2]]. Moyano et al. 2012 data f = p[0] + p[1]*Sat + p[2]*Sat**2
        #'moisture_coeff': [3.83, 4.43, 1.25, 0.854]  # moisture response; Skopp moisture function param [a ,b, d, g]}
        'beta': 6.0 # expontential decay factor for potential soil respiration. f = np.exp(beta*z). With beta=3.0, ca 80% of respiration is from top 50cm 
        }

# --- pyAPES.bottomlayer.OrganicLayer
#   Moss or litter layer properties. Define groundtypes

Forest_moss = {
    'name': 'Feather moss',  # Based on literature review of Pleurozium schreberi and Hylocomium splendens
    'layer_type': 'bryophyte',
    'coverage': 0.6, # fractional coverage [-]
    'height': 0.03,  # range (min, max): [0.021, 0.10]
    'roughness_height': 0.01, # [m]
    'bulk_density': 14.3, # kg m-3 range: [7.3, 28.74]
    'max_water_content': 9.7, # g H2O g-1 DM,  range: [7.91, 11.8]
    'water_content_ratio': 0.25,  # [-] max_symplast_water_content:max_water_content -ratio
    'min_water_content': 0.1, # g H2O g-1 DM
    'porosity': 0.98, # macroporosity

    # --- pyAPES.bottomlayer.carbon.BryophyteFarquhar
    'photosynthesis': {
        'Vcmax': 10.0, 'Jmax': 17.0, 'Rd': 0.5, # [umol m-2 (ground) s-1] at 25 degC
        'alpha': 0.3, 'theta': 0.8, 'beta': 0.9, # quantum yield [-], curvature [-], co-limitation[-]
        'gref': 0.02, 'wref': 7.0, 'a0': 0.7, 'a1': -0.263, 'CAP_desic': [0.44, 7.0],
        'tresp': { # temperature response 
                'Vcmax': [78., 200., 649.], # [activation energy, deactivation energy, entropy factor [kJ mol-1]]
                'Jmax': [56., 200., 646.],
                'Rd': [33.0]
                },
    },
    'optical_properties': {
        'emissivity': 0.98,
        'albedo': {'PAR': 0.11, 'NIR': 0.29} # albedo when moss is fully hydrated [-]
    },

    'water_retention': {
        # 'theta_s': 0.526,  
        # 'theta_r': 0.07,
        'alpha': 0.17, # air-entry potential [cm-1]  
        'n': 1.68,  # [-], pore connectivity
        'saturated_conductivity': 1.17e-8,  # [m s-1], based on fitted value of other mosses than Sphagnum
        'pore_connectivity': -2.30, # based on fitted value of other mosses than Sphagnum
    },

    'initial_conditions': {
        'temperature': 10.0, # degC
        'water_content': 10.0 # g H2O g-1 DM
    }
}

# this is general Sphagnum parametrisation based on literature review
Sphagnum = {
    'name': 'Sphagnum',
    'layer_type': 'bryophyte',
    'coverage': 0.3,
    'height': 0.04,  # range: [0.044, 0.076]
    'roughness_height': 0.02, # [m]
    'bulk_density': 35.1,  # [kg m-3], range: [9.28, 46.7]
    'max_water_content': 17.8,  # [g g-1 DM], range: [15.6, 24.4]
    'water_content_ratio': 0.43,  # max_symplast_water_content:max_water_content -ratio
    'min_water_content': 0.1, # [g g-1 DM]
    'porosity': 0.98, # macroporosity [-]

    'photosynthesis': {
        'Vcmax': 15.0, 'Jmax': 28.0, 'Rd': 0.75, # [umol m-2 (ground) s-1] at 25 degC
        'alpha': 0.3, 'theta': 0.8, 'beta': 0.9, # quantum yield, curvature, co-limitation
        'gref': 0.04, 'wref': 7.0, 'a0': 0.7, 'a1': -0.263, 'CAP_desic': [0.58, 10.0],
        'tresp': { # temperature response 
                'Vcmax': [78., 200., 649.], # [activation energy, deactivation energy, entropy factor [kJ mol-1]]
                'Jmax': [56., 200., 646.],
                'Rd': [33.0]
                },
    },
    'optical_properties': { # moisture responses are hard-coded
        'emissivity': 0.98,
        'albedo': {'PAR': 0.10, 'NIR': 0.27} # albedos when fully hydrated [-]
    },
    'water_retention': {
        # 'theta_s': 0.679,  # based on fitted value
        # 'theta_r': 0.176,  # based on fitted value
        'alpha': 0.381,  # air-entry potential [cm-1] 
        'n': 1.781,  # pore-size distribution [-]
        'saturated_conductivity': 3.4e-4,  # [m s-1]
        'pore_connectivity': -2.1  # [-]
    },
    'initial_conditions': {
        'temperature': 10.0, # degC
        'water_content': 20.0 # g g-1 DM
    }
}

Litter = {
    'name': 'Litter',
    'layer_type': 'litter',
    'coverage': 0.1,
    'height': 0.02,  # [m]
    'roughness_height': 0.01,  # [m]
    'bulk_density': 45.0,  # [kg m-3]
    'max_water_content': 4.0, # [g g-1 DM]
    'water_content_ratio': 0.25,  # max_symplast_water_content:max_water_content -ratio
    
    'min_water_content': 0.1, # [g g-1 DM]
    'porosity': 0.95,  # macroporosity[-]

    # -- pyAPES.bottomlayer.organiclayer.carbon.OrganicRespiration
    'respiration': {
        'q10': 1.6,  # base heterotrophic respiration rate at 10 degC [umol m-2(ground) s-1]
        'r10': 2.0,  # temperature sensitivity [-]
        #'moisture_coeff': [add here]
    },
    'optical_properties': {
        'emissivity': 0.98,  # [-]
        'albedo': {'PAR': 0.11, 'NIR': 0.29} # albedos when fully hydrated [-]
    },
    'water_retention': {#'theta_s': 0.95,  # max_water_content / WATER_DENSITY * bulk_density
                        #'theta_r': 0.01,  # min_water_content /WATER_DENSITY * bulk_density
        'alpha': 0.13,
        'n': 2.17,
        'saturated_conductivity': 1.16e-8,  # [m s-1]
        'pore_connectivity': -2.37,
    },
    'initial_conditions': {
        'temperature': 10.0,
        'water_content': 4.0
    }
}

# --- compile forestfloor parameter dictionary

forestfloor = {
    'bottom_layer_types': {'Feather_moss': Forest_moss,
                           'Sphagnum': Sphagnum,
                           'Litter': Litter
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
         'planttypes': {'Spruce large': pt1, 'Spruce small': pt2, 'Understory': pt3},
         'forestfloor': forestfloor
         }

# --- Soil water & heat: pyAPES.soil.Soil

# grid and soil properties: pF and conductivity values from Launiainen et al. (2015), Hyytiala

soil_grid = {#thickness of computational layers [m]: 0.01 m until 0.1m, 0.02m until 0.5m and 0.05 m until 2m depth
            'dz': [0.01] * 10 + [0.02] * 20 + [0.05] * 30,
            # bottom depth of layers with different characteristics [m]
            'zh': [-0.05, -0.1, -0.35, -10.0]
            }

soil_properties = {
    # Liu & Lennartz 2019 Sphagnum type II, III, V & Launiainen et al. 2022 cluster C1: silty soil with high org. content
    'pF': {  # vanGenuchten water retention parameters
           'ThetaS': [0.90, 0.8, 0.75, 0.6], # [m3m-3]
           'ThetaR': [0.001, 0.001, 0.001, 0.001], # [m3m-3]
           'alpha': [0.16, 0.22, 0.02, 1.12], # [cm-1]
           'n': [1.25, 1.21, 1.28, 4.45] # [-]
    },
    'saturated_conductivity_vertical': [1e-4, 1e-5, 1e-6, 1e6],  # saturated vertical hydraulic conductivity [m s-1]
    'saturated_conductivity_horizontal': [1e-4, 1e-5, 1e-6, 1e6],  # saturated horizontal hydraulic conductivity [m s-1]
    'solid_heat_capacity': None,  # [J m-3 (solid) K-1] - if None, estimated from organic/mineral composition
    'solid_composition': { # fraction of solids
        'organic': [1.0, 1.0, 1.0, 0.2],
        'sand': [0.0, 0.0, 0.0, 0.0],
        'silt': [0.0, 0.0, 0.0, 0.4],
        'clay': [0.0, 0.0, 0.0, 0.2],
    },
    'freezing_curve': [0.2, 0.2, 0.2, 0.4],  # freezing curve parameter, for peat see Nagare et al. 2012 HESS https://doi.org/10.5194/hess-16-501-2012
    'bedrock': {
        'solid_heat_capacity': 2.16e6,  # [J m-3 (solid) K-1]
        'thermal_conductivity': 3.0  # thermal conductivity of non-porous bedrock [W m-1 K-1]
    }
}

# --- water model: pyAPES.soil.water.Water

water_model = {'solve': True,
               'type': 'Equilibrium',  # solution approach 'Richards' | 'Equilibrium'
               'pond_storage_max': 0.05,  #  maximum pond depth [m]
               'initial_condition': {
                       'ground_water_level': -0.05,  # groundwater depth [m], <=0
                       'pond_storage': 0.0  # pond depth at surface [m]
                       },
               'lower_boundary': {
#                       'type': 'head_oneway',
#                       'value': -0.0,
                       'type': 'impermeable',
                       'value': None,
                       'depth': -5.0
                       },
               'drainage_equation': {
                       'type': None,  #
#                       'type': 'Hooghoudt',  #
#                       'depth': 1.0,  # drain depth [m]
#                       'spacing': 45.0,  # drain spacing [m]
#                       'width': 1.0,  # drain width [m]
                       }
                }

# --- heat model: pyAPES.soil.heat.Heat
heat_model = {'solve': True,
              'initial_condition': {
                      'temperature': -2.0,  # initial soil temperature [degC], assumed constant with dept - can also be array of correct length.
                      },
              'lower_boundary': {  # lower boundary condition (type, value)
                      'type': 'temperature',
                      'value': -2.0
                      },
              }

# --- compile soil model parameter dictionary
spara = {'grid': soil_grid,
         'soil_properties': soil_properties,
         'water_model': water_model,
         'heat_model': heat_model}
