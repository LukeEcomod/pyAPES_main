# -*- coding: utf-8 -*-
r"""
.. module: rootzone
    :synopsis: pyAPES-model planttype component
.. moduleauthor:: Kersti Leppä

Describes root water uptake of planttype.

Adapted from: 

    Volpe, V., Marani, M., Albertson, J.D. and Katul, G., 2013. Root controls
    on water redistribution and carbon uptake in the soil - plant system under current and future climate. 
    Advances in Water resources, 60, pp.110-120.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

from pyAPES.utils.constants import EPS

class RootUptake(object):
    r""" 
    Describes roots & water uptake of planttype.
    """
    def __init__(self, p: Dict, dz_soil: np.ndarray, LAImax: float):
        r"""
        Initializes rootuptake object.

        Args:
            p (dict):
                'root_depth': depth of rooting zone [m]
                'beta': shape parameter for root distribution model [-]
                'RAI_LAI_multiplier': multiplier for total fine root area index (RAI = 2*LAImax)
                'fine_radius': fine root radius [m]
                'radial_K': maximum bulk root membrane conductance in radial direction [s-1]
            dz_soil (array): thickness of soilprofile layers from top to bottom [m]
            LAImax (float): maximum leaf area index [m2 m-2]
        Returns:
            self (object)
        """
        # parameters
        self.root_depth = p['root_depth']
        self.fine_radius = p['fine_radius']  # fine root radius [m]
        self.root_cond = p['root_cond']  # [s]
        self.RAI = p['RAI_LAI_multiplier']*LAImax  # total fine root area index (m2/m2)
        self.rad = self.RAI * RootDistribution(p['beta'], dz_soil, p['root_depth'])
        self.ix = np.where(np.isfinite(self.rad))
        self.dz = dz_soil[self.ix]

        # state variables
        self.h_root = 0.0

    def wateruptake(self, transpiration_rate: float, h_soil: np.ndarray, kh_soil: np.ndarray) -> np.ndarray:
        r""" 
        Distributes root wateruptake to layers based on relative bulk-soil to root zylem conductance

        Agrs:
            self (object)
            transpiration_rate (float): whole plant [m s-1]
            h_soil (array): soil water potential [m]
            kh_soil (array): soil hydraulic conductivity [m s-1]
        Returns:
            rootsink (array)
        """

        # conductance from soil to root-soil interface [s-1]
        alpha = np.sqrt(self.root_depth/(self.RAI + EPS)) / np.sqrt(2.0 * self.fine_radius)
        ks = alpha * kh_soil[self.ix] * self.rad

        # conductance from soil-root interface to base of xylem [s-1]
        kr = self.rad * self.dz / self.root_cond

        # soil to xylem conductance [s-1]
        g_sr = ks * kr / (ks + kr + EPS)

        # assume total root uptake equals transpiration rate and solve uniform root pressure [m]
        self.h_root = -(transpiration_rate - sum(g_sr * h_soil[self.ix])) / sum(g_sr)

        # root uptake [m s-1]
        rootsink = g_sr * (h_soil[self.ix] - self.h_root)

        return rootsink

def RootDistribution(beta: float, dz: np.ndarray, root_depth: float) -> np.ndarray:
    r"""
    Computes normalized root area density distribution
    Args:
        beta (float): shape parameter for root distribution model [-]
        dz (array):  thickness soil layers from top to bottom [m]
        root_depth: depth of rooting zone [m]
    Returns:
        R (array): normalized root area density distribution with depth,
            extends only to depth of rooting zone. Integrates to unity: sum(dz*R) = 1

    Reference:
        Gale and Grigal, 1987 Can. J. For.Res., 17, 829 - 834.
    """
    z = np.concatenate([[0.0], np.cumsum(dz)])
    root_depth = np.minimum(root_depth, z[-1])
    z = np.concatenate([z[z < root_depth], [root_depth]])
    d = abs(z * 100.0)  # depth in cm

    Y = 1.0 - beta**d  # cumulative distribution (Gale & Grigal 1987)
    R = Y[1:] - Y[:-1]  # root area density distribution
    
    # Test: SET FIRST LAYER WITH NO ROOTS
    if len(R) > 1:
        R[0] = 0.0

    # addjust distribution to match soil profile depth
    R = R / sum(R) / dz[:len(R)]

    return R

# EOF