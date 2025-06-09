"""
.. module: snow
    :synopsis: APES-model component
.. moduleauthor:: Jari-Pekka Nousu

*Thermal properties of snow and soil (based on FSM2)*

"""

import numpy as np
from typing import Dict, List, Tuple
from pyAPES.utils.utilities import tridiag
from pyAPES.utils.constants import GRAVITY, SPECIFIC_HEAT_ICE, SPECIFIC_HEAT_WATER, \
    LATENT_HEAT_FUSION, LATENT_HEAT_SUBMILATION, \
    WATER_VISCOCITY, ICE_DENSITY, WATER_DENSITY, \
    T_MELT, K_AIR, K_ICE, K_WATER, K_SAND, K_CLAY

EPS = np.finfo(float).eps  # machine epsilon


class Thermal:
    def __init__(self, properties: Dict) -> object:
        """
        Thermal properties module for snow and soil.

        Args:
            properties (dict):
                'layers' (dict):
                    'Nsmax' (int): Maximum number of snow layers
                'physics_options' (dict):
                    'HYDROL' (int): Hydrology scheme selection
                    'CONDCT' (int): Soil thermal conductivity scheme selection
                    'DENSTY' (int): Snow density scheme selection

        Returns:
            self (object)
        """

        self.kfix = properties['params']['kfix']
        self.rhof = properties['params']['rhof']
        self.Nsmax = properties['layers']['Nsmax']
        self.HYDRL = properties['physics_options']['HYDRL']
        self.CONDCT = properties['physics_options']['CONDCT']
        self.DENSTY = properties['physics_options']['DENSTY']

    def run(self, forcing: Dict, solve_soil=True) -> Tuple:
        """
        Calculates one timestep and updates thermal state.

        
        Args:
            forcing (dict):
                'Nsnow' (int): Number of snow layers
                'Dsnw' (np.ndarray): Snow layer thicknesses (m)
                'Sice' (np.ndarray): Ice content of snow layers (kg/m^2)
                'Sliq' (np.ndarray): Liquid content of snow layers (kg/m^2)
                'Tsnow' (np.ndarray): Snow layer temperatures (K)
                'Tsoil' (np.ndarray): Soil layer temperatures (K)
                'Dzsoil' (np.ndarray): Soil layer thickness (m)
 
        Returns:
            (tuple):
            fluxes (dict):
            states (dict):
                Ds1 (float): Surface layer thickness (m)
                gs1 (float): Surface moisture conductance (m/s)
                ks1 (float): Surface layer thermal conductivity (W/m/K)
                Ts1 (float): Surface layer temperature (K)
                csoil (np.ndarray): Areal heat capacity of soil layers (J/K/m^2)
                ksnow (np.ndarray): Thermal conductivity of snow layers (W/m/K)
                ksoil (np.ndarray): Thermal conductivity of soil layers (W/m/K)
        """
        
        Nsnow = forcing['Nsnow']
        Dsnw = forcing['Dsnw']
        Sice = forcing['Sice']
        Sliq = forcing['Sliq']
        Tsnow = forcing['Tsnow']
        Tsoil = forcing['Tsoil']
        ksoil = forcing['ksoil']
        gs1 = forcing['gs1']
        Dzsoil = forcing['Dzsoil']

        ksnow = np.zeros(int(self.Nsmax))
        ksnow[:] = self.kfix

        if self.CONDCT == 1:
            for k in range(Nsnow):
                rhos = self.rhof
                if self.DENSTY != 0:
                    if (Dsnw[k] > EPS):
                        rhos = (Sice[k] + Sliq[k]) / Dsnw[k]
                ksnow[k] = 2.224 * (rhos / WATER_DENSITY)**1.885

        # Surface layer
        Ds1 = np.maximum(Dzsoil, Dsnw[0]) # thickness
        Ts1 = Tsoil + (Tsnow[0] - Tsoil)*Dsnw[0]/Dzsoil # temperature
        ks1 = Dzsoil/(2*Dsnw[0]/ksnow[0] +
                              (Dzsoil - 2*Dsnw[0])/ksoil) # thermal conductivity
        hs = np.sum(Dsnw) # snow depth
        if (hs > 0.5*Dzsoil):
            ks1 = ksnow[0]
        if (hs > Dzsoil):
            Ts1 = Tsnow[0]

        fluxes = {}

        states = {'Ds1': Ds1,
                  'gs1': gs1,
                  'ks1': ks1,
                  'Ts1': Ts1,
                  'ksnow': ksnow,
                  'ksoil': ksoil,
                  }

        return fluxes, states
