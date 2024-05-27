# -*- coding: utf-8 -*-
"""
.. module: photo
    :synopsis: pyAPES leaf component. Defines functions for leaf boundary-layer thickness and boundary-layer conductance-
.. moduleauthor:: Samuli Launiainen & Kersti Leppä

Key references:
    Launiainen et al. 2015 Ecol. Mod.
    Campbell, S.C., and J.M. Norman (1998), An introduction to Environmental Biophysics, 
    Springer, 2nd edition, Ch. 7
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import List, Dict, Tuple

from pyAPES.utils.constants import EPS, GRAVITY, MOLECULAR_DIFFUSIVITY_CO2, MOLECULAR_DIFFUSIVITY_H2O, \
THERMAL_DIFFUSIVITY_AIR, AIR_VISCOSITY

def leaf_boundary_layer_conductance(u: np.ndarray, d: float, Ta: np.ndarray, 
                                    dT:np.ndarray, P: float=101300.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes 2-sided leaf boundary layer conductance assuming mixed forced and free
    convection form two parallel transport mechanisma through the leaf boundary layer.

    Reference: 
        Campbell, S.C., and J.M. Norman (1998), An introduction to Environmental Biophysics, 
        Springer, 2nd edition, Ch. 7
    
    Note: the factor of 1.4 is adopted for outdoor environment, see Campbell and Norman, 1998.

    Args: 
        u (float|array): [m s-1], mean velocity
        d (float|array): [m], characteristic dimension of the leaf
        Ta (float|array): [degC], ambient temperature
        dT (float|array): [degC], leaf-air temperature difference, for free convection
        P (float): [Pa], air pressure

    Returns: 
        (tuple):
            gb_h (float|array): [mol m-2 (leaf) s-1], bl conductance for heat
            gb_c (float|array): [mol m-2 (leaf) s-1], bl conductance for CO2
            gb_v (float|array): [mol m-2 (leaf) s-1], bl conductance for H2O

    """

    u = np.maximum(u, EPS)

    factor1 = 1.4*2  # forced conv. both sides, 1.4 is correction for turbulent flow
    factor2 = 1.5  # free conv.; 0.5 comes from cooler surface up or warmer down

    # -- Adjust diffusivity, viscosity, and air density to current pressure/temp.
    t_adj = (101300.0 / P)*((Ta + 273.15) / 293.16)**1.75

    Da_v = MOLECULAR_DIFFUSIVITY_H2O*t_adj
    Da_c = MOLECULAR_DIFFUSIVITY_CO2*t_adj
    Da_T = THERMAL_DIFFUSIVITY_AIR*t_adj
    va = AIR_VISCOSITY*t_adj
    rho_air = 44.6*(P / 101300.0)*(273.15 / (Ta + 273.13))  # [mol/m3]

    # ----- Compute the leaf-level dimensionless groups
    Re = u*d / va  # Reynolds number
    Sc_v = va / Da_v  # Schmid numbers for water
    Sc_c = va / Da_c  # Schmid numbers for CO2
    Pr = va / Da_T  # Prandtl number
    Gr = GRAVITY*(d**3)*abs(dT) / (Ta + 273.15) / (va**2)  # Grashoff number

    #r = Gr / (Re**2)  # ratio of free/forced convection
    
    # ----- aerodynamic conductance for "forced convection"
    gb_T = (0.664*rho_air*Da_T*Re**0.5*(Pr)**0.33) / d  # [mol/m2/s]
    gb_c=(0.664*rho_air*Da_c*Re**0.5*(Sc_c)**0.33) / d  # [mol/m2/s]
    gb_v=(0.664*rho_air*Da_v*Re**0.5*(Sc_v)**0.33) / d  # [mol/m2/s]

    # ----- Compute the aerodynamic conductance for "free convection"
    gbf_T = (0.54*rho_air*Da_T*(Gr*Pr)**0.25) / d  # [mol/m2/s]
    gbf_c = 0.75*gbf_T  # [mol/m2/s]
    gbf_v = 1.09*gbf_T  # [mol/m2/s]

    # --- aerodynamic conductance: "forced convection"+"free convection"
    gb_h = factor1*gb_T + factor2*gbf_T
    gb_c = factor1*gb_c + factor2*gbf_c
    gb_v = factor1*gb_v + factor2*gbf_v

    return gb_h, gb_c, gb_v#, r