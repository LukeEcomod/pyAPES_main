"""
.. module: snow
    :synopsis: APES-model component
.. moduleauthor:: Jari-Pekka Nousu

*Thermal properties of snow and its lower boundary (based on FSM2)*

"""

import numpy as np
from typing import Dict, List, Tuple
from pyAPES.utils.utilities import tridiag
from pyAPES.utils.constants import WATER_DENSITY
EPS = np.finfo(float).eps  # machine epsilon

class Thermal:
    def __init__(self, properties: Dict) -> object:
        """
        Thermal properties module for snow and its lower boundary.

        Args:
            properties (dict):
                'params' (dict):
                    'kfix' (float): Fixed thermal conductivity for snow (W m-1 K-1)
                    'rhof' (float): Reference snow density for thermal conductivity (kg m-3)
                'layers' (dict):
                    'Nsmax' (int): Maximum number of snow layers
                'physics_options' (dict):
                    'CONDCT' (int): Snow thermal conductivity scheme selection
                    'DENSTY' (int): Snow density scheme selection
        Returns:
            self (object)
        """

        self.kfix = properties['params']['kfix']
        self.rhof = properties['params']['rhof']
        self.Nsmax = properties['layers']['Nsmax']
        self.CONDCT = properties['physics_options']['CONDCT']
        self.DENSTY = properties['physics_options']['DENSTY']

    def run(self, forcing: Dict) -> Tuple:
        """
        Calculates one timestep.
    
        Args:
            forcing (dict):
                'Nsnow' (int): Number of snow layers
                'Dsnw' (np.ndarray): Snow layer thicknesses (m)
                'Sice' (np.ndarray): Ice content of snow layers (kg m-2)
                'Sliq' (np.ndarray): Liquid content of snow layers (kg m-2)
                'Tsnow' (np.ndarray): Snow layer temperatures (K)
                'Tlb' (float): Lower-boundary organic-layer temperature (K)
                'klb' (float): Lower-boundary organic-layer thermal conductivity (W m-1 K-1)
                'gs1' (float): Surface moisture conductance (m/s)
                'Dzlb' (float): Lower-boundary organic-layer thickness (m)
 
        Returns:
            (tuple):
            fluxes (dict):
            states (dict):
                'Ds1' (float): Surface layer thickness (m)
                'gs1' (float): Surface moisture conductance (m/s)
                'ks1' (float): Surface layer thermal conductivity (W m-1 K-1)
                'Ts1' (float): Surface layer temperature (K)
                'ksnow' (np.ndarray): Thermal conductivity of snow layers (W m-1 K-1)
        """
        
        # read forcings
        Nsnow = forcing['Nsnow']
        Dsnw = forcing['Dsnw']
        Sice = forcing['Sice']
        Sliq = forcing['Sliq']
        Tsnow = forcing['Tsnow']
        Tlb = forcing['Tlb']
        klb = forcing['klb']
        gs1 = forcing['gs1']
        Dzlb = forcing['Dzlb']

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
        Ds1 = np.maximum(Dzlb, Dsnw[0]) # thickness
        Ts1 = Tlb + (Tsnow[0] - Tlb)*Dsnw[0]/Dzlb # temperature
        ks1 = Dzlb/(2*Dsnw[0]/ksnow[0] +
                              (Dzlb - 2*Dsnw[0])/klb) # thermal conductivity
        
        hs = np.sum(Dsnw) # snow depth

        if (hs > 0.5*Dzlb):
            ks1 = ksnow[0]
        if (hs > Dzlb):
            Ts1 = Tsnow[0]

        fluxes = {}

        states = {'Ds1': Ds1,
                  'gs1': gs1,
                  'ks1': ks1,
                  'Ts1': Ts1,
                  'ksnow': ksnow,
                  }

        return fluxes, states
