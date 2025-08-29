# -*- coding: utf-8 -*-
"""
.. module: pyAPES.parameters.mlm_parameters
    :synopsis: pyAPES_MLM PARAMETERIZATION
.. moduleauthor:: Samuli Launiainen

Define pyAPES_MLM parameters and forcing file here.

Define pyAPES_MLM output variables and logger config in: parameters.mlm_outputs

"""

import numpy as np
from pyAPES.utils.utilities import lad_weibul, lad_constant


gpara = {'dt' : 1800.0,  # timestep in forcing data file [s]
         'start_time' : "2006-06-01",  # start time of simulation [yyyy-mm-dd]
         'end_time' : "2006-06-10",  # end time of simulation [yyyy-mm-dd]
         'forc_filename' : '/Users/jpnousu/pyAPES_main/forcing/FIHy_forcing_2006_2008.dat', # forcing data file
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
loc = {'lat': 61.51,  # latitude, decimal deg
       'lon': 24.0  # longitude, decimal deg
       }

# grid
grid = {'zmax': 25.0,  # heigth of grid from ground surface. Corresponds to height of forcing data [m]
        'Nlayers': 101  # number of layers [-]
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
#       Note: tree height h must be less than grid['zref']

pt1 = { 'name': 'pine',
        'LAImax': 2.1, # maximum annual LAI m2m-2
        'lad': lad_weibul(z, LAI=1.0, h=15.0, hb=3.0, species='pine'),  # leaf-area density m2m-3
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
            'Vcmax': 55.0, # maximum carboxylation rate [umol m-2 (leaf) s-1] at 25 degC
            'Jmax': 105.0,  # maximum electron transport rate[umol m-2 (leaf) s-1] at 25 degC1.97*Vcmax (Kattge and Knorr, 2007)
            'Rd': 1.3, # dark respiration rate [umol m-2 (leaf) s-1] at 25 degC
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
            'root_depth': 0.5, # rooting depth [m]
            'beta': 0.943, # root distribution shape parameter [-]
            'root_to_leaf_ratio': 2.0, # fine-root to leaf-area ratio [-]
            'root_radius': 2.0e-3, # [m]
            'root_conductance': 5.0e8, # [s]
            }
        }
pt2 = { 'name': 'shrubs',
        'LAImax': 0.5, # maximum annual LAI m2m-2
        'lad': lad_constant(z, LAI=1.0, h=0.5, hb=0.0),  # leaf-area density [m2 m-3]
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
            'Vcmax': 40.0, # maximum carboxylation rate [umol m-2 (leaf) s-1] at 25 degC
            'Jmax': 76.0,  # maximum electron transport rate[umol m-2 (leaf) s-1] at 25 degC
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
            'root_depth': 0.5, # rooting depth [m]
            'beta': 0.943, # root distribution shape parameter [-]
            'root_to_leaf_ratio': 2.0, # fine-root to leaf-area ratio [-]
            'root_radius': 2.0e-3, # [m]
            'root_conductance': 5.0e8, # [s]
            }
        }

# --- forestfloor: pyAPES.canopy.forestfloor.ForestFloor combines snowpack, soil, and organiclayer types.

# --- pyAPES.snowpack
snow_model = {'type': 'fsm2'} # snowpack model either 'degreeday' or 'fsm2'

# --- pyAPES.snow.degreeday.degreeday.DegreeDaySnow
degreeday = {
        'kmelt': 2.31e-5,  # Melting coefficient [kg m-2 s-1 degC-1]; (= 2.0 mm degC d-1)
        'kfreeze': 5.79e-6,  # Freezing  coefficient [kg m-2 s-1 degC-1] (=0.5 mm degC d-1)
        'retention': 0.2,  # max fraction of liquid water in snow [-]
        'Tmelt': 0.0,  # temperature when melting starts [degC]
        'optical_properties': {
                'emissivity': 0.97,
                'albedo': {'PAR': 0.8, 'NIR': 0.8}
                },
        'initial_conditions': {'temperature': 0.0,
                               'snow_water_equivalent': 0.0,
                               }
        }

# --- pyAPES.snow.energybalance.fsm2.FSM2
fsm2 = {'physics_options': {
            'DENSTY': 1,
            'HYDRL': 1,
            'CONDCT': 1,
            'ZOFFST': 0,
            'CANMOD': 0,
            'EXCHNG': 0,
            'ALBEDO': 2,
            'SNFRAC': 1,
            'SWPART': 0,
        },
        'params': {
            'asmn': 0.5,            # Minimum albedo for melting snow
            'asmx': 0.85,           # Maximum albedo for fresh snow
            'eta0': 3.7e7,          # Reference snow viscosity (Pa s)
            'hfsn': 0.1,            # Snowcover fraction depth scale (m)
            'kfix': 0.24,           # Fixed thermal conductivity of snow (W/m/K)
            'rcld': 300,            # Maximum density for cold snow (kg/m^3)
            'rfix': 300,            # Fixed snow density (kg/m^3)
            'rgr0': 5e-5,           # Fresh snow grain radius (m)
            'rhof': 100,            # Fresh snow density (kg/m^3)
            'rhow': 300,            # Wind-packed snow density (kg/m^3)
            'rmlt': 500,            # Maximum density for melting snow (kg/m^3)
            'Salb': 10,             # Snowfall to refresh albedo (kg/m^2)
            'snda': 2.8e-6,         # Thermal metamorphism parameter (1/s)
            'Talb': -2,             # Snow albedo decay temperature threshold (C)
            'tcld': 3.6e6,          # Cold snow albedo decay time scale (s)
            'tmlt': 3.6e5,          # Melting snow albedo decay time scale (s)
            'trho': 200*3600,       # Snow compaction timescale (s)
            'Wirr': 0.03,           # Irreducible liquid water content of snow
            'gsnf': 0.01,           # Snow-free vegetation moisture conductance (m/s)
            'hbas': 2.0,            # Canopy base height (m)
            'kext': 0.5,            # Vegetation light extinction coefficient
            'leaf': 20,             # Leaf boundary resistance (s/m)^(1/2)
            'wcan': 2.5,            # Canopy wind decay coefficient
            'svai': 4.4,            # Intercepted snow capacity per unit VAI (kg/m^2)
            'tunl': 240 * 3600,     # Canopy snow unloading time scale (s)
            'z0sf': 0.1,           # Snow-free surface roughness length (m)
            'z0sn': 0.001,          # Snow roughness length (m)
            'VAI': 0.0,             # Vegetation area index
            'vegh': 0.0,            # Canopy height (m)
            'zT': 18.,              # Temperature measurement height with offset (m)
            'zU': 18.,              # Wind measurement height with offset (m)
            'hfsn': 0.1,            # Snowcover fraction depth scale (m)
            'acn0': 0.1,            # Snow-free dense canopy albedo
            'acns': 0.4,            # Snow-covered dense canopy albedo
            #'lveg': 0.0,
            #'elev': 100.0
        },
        'layers': {
            'Nsmax': 3,                 # Maximum number of snow layers
            'Ncnpy': 0,                 # Number of canopy layers
            'Dzsnow': np.array([0.1, 0.2, 0.4]),  # Minimum snow layer thicknesses (m)
            'fvg1': [],                 # Fraction of vegetation in the upper canopy layer
            'zsub': 2.0,                # Subcanopy wind speed diagnostic height (m)
            'Nsoil': 4,                 # Soil layers
            'Dzsoil': np.array([0.1, 0.2, 0.4, 0.8]), # Soil layer thicknesses
        },
        'initial_conditions': {
            'Nsnow': 0,             # Number of snow layers
            'Dsnw': np.array([0.0, 0.0, 0.0]),      # Snow layer thicknesses (m)
            'Rgrn': np.array([0.0, 0.0, 0.0]),      # Snow layer grain radius (m)
            'Sice': np.array([0.0, 0.0, 0.0]),      # Ice content of snow layers (kg/m^2)
            'Sliq': np.array([0.0, 0.0, 0.0]),      # Liquid content of snow layers (kg/m^2)
            'Tsnow': np.array([273., 273., 273.]),   # Snow layer temperatures (K)
            'Tsoil': np.array([285., 285., 285., 285.]),   # Soil layer temperatures (K)
            'Wflx': np.array([0.0, 0.0, 0.0]),      # Water flux into snow layer (kg/m^2/s)
            'Tsrf': 285.,         # Snow/ground surface temperature (K)
            'fsnow': 0.0,           # Snow cover fraction
            'fcans': 0.0,
            'Vsmc': np.array([0.3, 0.3, 0.3, 0.3])  # Volumetric water content in soil
        },
        'soilprops': {
            'fcly': 0.3,    # Fraction of clay
            'fsnd': 0.6,    # Fraction of sand
            'gsat': 0.01,   # Surface conductance for saturated soil (m/s)
            'z0sf': 0.1,    # Surface roughness length
            'alb0': 0.2     # Snow-free surface albedo
        }
    }


# --- pyAPES.bottomlayer.carbon.SoilRespiration
soil_respiration = {
        'r10': 2.5, # base rate (bulk heterotrophic + autotrophic) [umol m-2 (ground) s-1]
        'q10': 2.0, # temperature sensitivity [-]
        'moisture_coeff': [3.83, 4.43, 1.25, 0.854]  # moisture response; Skopp moisture function param [a ,b, d, g]}
        }

# --- pyAPES.bottomlayer.OrganicLayer
#   Moss or litter layer properties. Define groundtypes

Forest_moss = {
    'name': 'forest mosses',  # Based on literature review of Pleurozium schreberi and Hylocomium splendens
    'layer_type': 'bryophyte',
    'coverage': 1.0, # fractional coverage [-]
    'height': 0.057,  # range (min, max): [0.021, 0.10]
    'roughness_height': 0.01, # [m]
    'bulk_density': 14.3, # kg m-3 range: [7.3, 28.74]
    'max_water_content': 9.7, # g H2O g-1 DM,  range: [7.91, 11.8]
    'water_content_ratio': 0.25,  # [-] max_symplast_water_content:max_water_content -ratio
    'min_water_content': 0.1, # g H2O g-1 DM
    'porosity': 0.98, # macroporosity

    # --- pyAPES.bottomlayer.carbon.BryophyteFarquhar
    'photosynthesis': {
        'Vcmax': 15.0, 'Jmax': 28.5, 'Rd': 0.75, # [umol m-2 (ground) s-1] at 25 degC
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
    'name': 'Sphagnum sp.',
    'layer_type': 'bryophyte',
    'coverage': 0.0, # Note - now no sphagnum!
    'height': 0.06,  # range: [0.044, 0.076]
    'roughness_height': 0.02, # [m]
    'bulk_density': 35.1,  # [kg m-3], range: [9.28, 46.7]
    'max_water_content': 17.8,  # [g g-1 DM], range: [15.6, 24.4]
    'water_content_ratio': 0.43,  # max_symplast_water_content:max_water_content -ratio
    'min_water_content': 0.1, # [g g-1 DM]
    'porosity': 0.98, # macroporosity [-]

    'photosynthesis': {
        'Vcmax': 45.0, 'Jmax': 85.5, 'Rd': 1.35, # [umol m-2 (ground) s-1] at 25 degC
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
    'coverage': 0.0,  # [-], Note - now no litter!
    'height': 0.03,  # [m]
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
    'bottom_layer_types': {
        #'litter': Litter,
        'forest_moss': Forest_moss,
        #'sphagnum': Sphagnum,
    },
    'snow_model': snow_model,
    'snowpack': {'degreeday': degreeday, 'fsm2': fsm2}.get(snow_model.get('type')),
    'soil_respiration': soil_respiration
}

# --- compile canopy-model parameter dictionary

cpara = {'loc': loc,
         'ctr': ctr,
         'grid': grid,
         'radiation': radiation,
         'micromet': micromet,
         'interception': interception,
         'planttypes': {'pine': pt1, 'shrubs': pt2},
         'forestfloor': forestfloor
         }

# --- Soil water & heat: pyAPES.soil.Soil

# grid and soil properties: pF and conductivity values from Launiainen et al. (2015), Hyytiala

soil_grid = {#thickness of computational layers [m]
            'dz': [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                   0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
                   0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                   0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            # bottom depth of layers with different characteristics [m]
            'zh': [-0.05, -0.11, -0.35, -10.0]
            }

soil_properties = {'pF': {  # vanGenuchten water retention parameters
                        'ThetaS': [0.80, 0.50, 0.50, 0.41],
                        'ThetaR': [0.01, 0.08, 0.08, 0.03],
                        'alpha': [0.70, 0.06, 0.06, 0.05],
                        'n': [1.25, 1.35, 1.35, 1.21]
                        },
                  'saturated_conductivity_vertical': [2.42E-05, 2.08e-06, 3.06e-06, 4.17e-06],  # saturated vertical hydraulic conductivity [m s-1]
                  'saturated_conductivity_horizontal': [2.42E-05, 2.08e-06, 3.06e-06, 4.17e-06],  # saturated horizontal hydraulic conductivity [m s-1]
                  'solid_heat_capacity': None,  # [J m-3 (solid) K-1] - if None, estimated from organic/mineral composition
                  'solid_composition': {
                         'organic': [0.1611, 0.0714, 0.1091, 0.028],
                         'sand': [0.4743, 0.525, 0.5037, 0.5495],
                         'silt': [0.3429, 0.3796, 0.3641, 0.3973],
                         'clay': [0.0217, 0.0241, 0.0231, 0.0252]
                         },
                  'freezing_curve': [0.2, 0.5, 0.5, 0.5],  # freezing curve parameter
                  'bedrock': {
                              'solid_heat_capacity': 2.16e6,  # [J m-3 (solid) K-1]
                              'thermal_conductivity': 3.0  # thermal conductivity of non-porous bedrock [W m-1 K-1]
                              }
                  }

# --- water model: pyAPES.soil.water.Water

water_model = {'solve': True,
               'type': 'Richards',  # solution approach 'Richards' | 'Equilibrium'
               'pond_storage_max': 0.05,  #  maximum pond depth [m]
               'initial_condition': {
                       'ground_water_level': -2.0,  # groundwater depth [m], <=0
                       'pond_storage': 0.0  # pond depth at surface [m]
                       },
               'lower_boundary': {
                       'type': 'head_oneway',
                       'value': -0.0,
#                       'type': 'impermeable',
#                       'value': None,
#                       'depth': -2.0
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
                      'temperature': 4.0,  # initial soil temperature [degC], assumed constant with dept - can also be array of correct length.
                      },
              'lower_boundary': {  # lower boundary condition (type, value)
                      'type': 'temperature',
                      'value': 4.0
                      },
              }

# --- compile soil model parameter dictionary
spara = {'grid': soil_grid,
         'soil_properties': soil_properties,
         'water_model': water_model,
         'heat_model': heat_model}
