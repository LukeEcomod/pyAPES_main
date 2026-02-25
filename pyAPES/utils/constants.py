#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
.. module: pyAPES.utils.constants
    :synopsis: constants used in pyAPES 
.. moduleauthor:: Kersti Leppä, Samuli Launiainen, Antti-Jussi Kieloaho

"""

import numpy as np

#: machine epsilon
EPS = np.finfo(float).eps

#: [J mol-1], latent heat of vaporization at 20 degC
LATENT_HEAT = 44100.0
#: [kg mol-1], molar mass of H2O
MOLAR_MASS_H2O = 18.015e-3
#: [kg mol-1], molar mass of CO2
MOLAR_MASS_CO2 = 44.01e-3
#: [kg mol-1], molar mass of C
MOLAR_MASS_C = 12.01e-3
#: [kg mol-1], molar mass of dry air
MOLAR_MASS_AIR = 29.0e-3
#: [J kg-1 K-1], specific heat of H2O
SPECIFIC_HEAT_H2O = 4.18e3
#: [J kg-1 K-1], specific heat of ice
SPECIFIC_HEAT_ICE = 2.09e3
#: [J kg-1 K-1], specific heat of organic matter
SPECIFIC_HEAT_ORGANIC_MATTER = 1.92e3
#: [J kg-1 K-1], heat capacity of dry air at constant pressure
SPECIFIC_HEAT_AIR = 29.3
#: [W m-2 K-4], Stefan-Boltzmann constant
STEFAN_BOLTZMANN = 5.6697e-8
#: [-], von Karman constant
VON_KARMAN = 0.41
#: [K], zero degrees celsius in Kelvin
DEG_TO_KELVIN = 273.15
#: [K], zero degrees celsius in Kelvin
NORMAL_TEMPERATURE = 273.15
#: [mol m-3], molar density of dry air at 20 degC
AIR_DENSITY = 41.6
#: [m2 s-1], kinematic viscosity of air at 20 degC
AIR_VISCOSITY = 15.1e-6
#: [m2 s-1], thermal diffusivity of air at 20 degC
THERMAL_DIFFUSIVITY_AIR = 21.4e-6
#: [m2 s-1], molecular diffusvity of CO2 at 20 degC
MOLECULAR_DIFFUSIVITY_CO2 = 15.7e-6
#: [m2 s-1], molecular diffusvity of H2O at 20 degC
MOLECULAR_DIFFUSIVITY_H2O = 24.0e-6
#: [J mol-1 K-1], universal gas constant
GAS_CONSTANT = 8.314
#: [K] reference temperature for photosynthetic parameters, 283.15 [K]
TN = 25.0 + DEG_TO_KELVIN  
#: [J mol-1] universal gas constant times 25C reference temperature in Kelvin
TN_GAS_CONSTANT = TN*GAS_CONSTANT
#: [kg m2 s-1], standard gravity
GRAVITY = 9.81
#: [kg m-3], water density
WATER_DENSITY = 1.0e3
#: [1 W m-2 in umol m2 s-1], conversion from watts to micromol
PAR_TO_UMOL = 4.56
#: [1 ged in rad], conversion from deg to rad
DEG_TO_RAD = 3.14159 / 180.0
#: [umol m-3], molar O2 concentration in air
O2_IN_AIR = 2.10e5
#: [-] H2O to CO2 diffusivity ratio
H2O_CO2_RATIO = 1.6  

#%% for pyAPES.soil

#: [J kg-1], latent heat of freezing
LATENT_HEAT_FREEZING = 333700.0
#: [degC], freezing point of water
FREEZING_POINT_H2O = 0.0
#: [kg m-3], densities
ICE_DENSITY = 917.0

#: [J m-1 K-1], thermal condutivities of soil constituents
K_WATER = 0.57 # liquid phase
K_ICE = 2.2 # ice-phase
K_AIR = 0.025 # air
K_ORG = 0.25 # organic matter
K_SAND = 7.7  # Tian et al. 2016
K_SILT = 2.74  # Tian et al. 2016
K_CLAY = 1.93  # Tian et al. 2016

#: volumetric heat capacities  [J m-3 -1]
CV_AIR = 1297.0  # air at 101kPa
CV_WATER = 4.18e6  # water
CV_ICE = 1.93e6  # ice
CV_ORGANIC = 2.50e6  # dry organic matter
CV_MINERAL = 2.31e6  # soil minerals

# H2O molar density [mol m-3]
H2O_MOLARDENSITY = 55.5e3
# H2 18O diffusivity [m2 s-1]
H2_18O_DIFFYSIVITY = 2.66e-9
# EOF