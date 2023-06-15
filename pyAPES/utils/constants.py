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

#: [J mol\ :sup:`-1`\ ], latent heat of vaporization at 20\ :math:`^{\circ}`\ C
LATENT_HEAT = 44100.0
#: [kg mol\ :sup:`-1`\ ], molar mass of H\ :sub:`2`\ O
MOLAR_MASS_H2O = 18.015e-3
#: [kg mol\ :sup:`-1`\ ], molar mass of CO\ :sub:`2`\
MOLAR_MASS_CO2 = 44.01e-3
#: [kg mol\ :sup:`-1`\ ], molar mass of C
MOLAR_MASS_C = 12.01e-3
#: [kg mol\ :sup:`-1`\ ], molar mass of air
MOLAR_MASS_AIR = 29.0e-3
#: [J kg\ :sup:`-1` K\ :sup:`-1`\ ], specific heat of H\ :sub:`2`\ O
SPECIFIC_HEAT_H2O = 4.18e3
#: [J kg\ :sup:`-1` K\ :sup:`-1`\ ], specific heat of organic matter
SPECIFIC_HEAT_ORGANIC_MATTER = 1.92e3
#: [J mol\ :sup:`-1` K\ :sup:`-1`\ ], heat capacity of air at constant pressure
SPECIFIC_HEAT_AIR = 29.3
#: [W m\ :sup:`-2` K\ :sup:`-4`\ ], Stefan-Boltzmann constant
STEFAN_BOLTZMANN = 5.6697e-8
#: [-], von Karman constant
VON_KARMAN = 0.41
#: [K], zero degrees celsius in Kelvin
DEG_TO_KELVIN = 273.15
#: [K], zero degrees celsius in Kelvin
NORMAL_TEMPERATURE = 273.15
#: [mol m\ :sup:`-3`\ ], density of air at 20\ :math:`^{\circ}`\ C
AIR_DENSITY = 41.6
#: [m\ :sup:`2` s\ :sup:`-1`\ ], kinematic viscosity of air at 20\ :math:`^{\circ}`\ C
AIR_VISCOSITY = 15.1e-6
#: [m\ :sup:`2` s\ :sup:`-1`\ ], thermal diffusivity of air at 20\ :math:`^{\circ}`\ C
THERMAL_DIFFUSIVITY_AIR = 21.4e-6
#: [m\ :sup:`2` s\ :sup:`-1`\ ], molecular diffusvity of CO\ :sub:`2` at 20\ :math:`^{\circ}`\ C
MOLECULAR_DIFFUSIVITY_CO2 = 15.7e-6
#: [m\ :sup:`2` s\ :sup:`-1`\ ], molecular diffusvity of H\ :sub:`2`\ at 20\ :math:`^{\circ}`\ C
MOLECULAR_DIFFUSIVITY_H2O = 24.0e-6
#: [J mol\ :sup:`-1` K\ :sup:``-1], universal gas constant
GAS_CONSTANT = 8.314
#: [kg m\ :sup:`2` s\ :sup:`-1`\ ], standard gravity
GRAVITY = 9.81
#: [kg m\ :sup:`-3`\ ], water density
WATER_DENSITY = 1.0e3
#: [umol m\ :sup:`2` s\ :sup:`-1`\ ], conversion from watts to micromol
PAR_TO_UMOL = 4.56
#: [rad], conversion from deg to rad
DEG_TO_RAD = 3.14159 / 180.0
#: [umol m\ :sup:`-1`], O2 concentration in air
O2_IN_AIR = 2.10e5

#%% for pyAPES.soil

#: [J kg-1], latent heat of freezing
LATENT_HEAT_FREEZING = 333700.0
#: [degC], freezing point of water
FREEZING_POINT_H2O = 0.0
#: [kg m-3], densities
ICE_DENSITY = 917.0

#: [J m-1 K-1], thermal condutivities
K_WATER = 0.57
K_ICE = 2.2
K_AIR = 0.025
K_ORG = 0.25
K_SAND = 7.7  # Tian et al. 2016
K_SILT = 2.74  # Tian et al. 2016
K_CLAY = 1.93  # Tian et al. 2016

#: volumetric heat capacieties  [J m-3 -1]
CV_AIR = 1297.0  # air at 101kPa
CV_WATER = 4.18e6  # water
CV_ICE = 1.93e6  # ice
CV_ORGANIC = 2.50e6  # dry organic matter
CV_MINERAL = 2.31e6  # soil minerals

# EOF