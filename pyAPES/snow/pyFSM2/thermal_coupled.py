"""
.. module: snow
    :synopsis: APES-model component
.. moduleauthor:: Jari-Pekka Nousu

*Thermal properties of snow and soil (based on FSM2)*

"""

import numpy as np
from typing import Dict, List, Tuple
from pyAPES.utils.utilities import tridiag
from pyAPES.utils.constants import WATER_DENSITY
EPS = np.finfo(float).eps  # machine epsilon

class Thermal:
    def __init__(self, properties: Dict) -> object:
        """
        Thermal properties module for snow and soil.

        Args:
            properties (dict):
                'params' (dict):
                    'kfix' (float): Fixed thermal conductivity for snow (W/m/K)
                    'rhof' (float): Reference snow density for thermal conductivity (kg/m^3
                'layers' (dict):
                    'Nsmax' (int): Maximum number of snow layers
                'physics_options' (dict):
                    'CONDCT' (int): Soil thermal conductivity scheme selection
                    'DENSTY' (int): Snow density scheme selection
        Returns:
            self (object)
        """

        self.kfix = properties['params']['kfix']
        self.rhof = properties['params']['rhof']
        self.Nsmax = properties['layers']['Nsmax']
        self.CONDCT = properties['physics_options']['CONDCT']
        self.DENSTY = properties['physics_options']['DENSTY']

    def run(self, forcing: Dict, solve_soil=True) -> Tuple:
        """
        Calculates one timestep.
    
        Args:
            forcing (dict):
                'Nsnow' (int): Number of snow layers
                'Dsnw' (np.ndarray): Snow layer thicknesses (m)
                'Sice' (np.ndarray): Ice content of snow layers (kg/m^2)
                'Sliq' (np.ndarray): Liquid content of snow layers (kg/m^2)
                'Tsnow' (np.ndarray): Snow layer temperatures (K)
                'Tsoil' (np.ndarray): Soil layer temperatures (K)
                'ksoil' (np.ndarray): Soil layer thermal conductivities (W/m/K)
                'kbt' (float): Thermal conductivity of organic layer (W/m/K)
                'gs1' (float): Surface moisture conductance (m/s)
                'Dzsoil' (float): Soil layer thickness (m)
                'Dzbt' (float): Thickness of organic layer (m)
 
        Returns:
            (tuple):
            fluxes (dict):
            states (dict):
                'Ds1' (float): Surface layer thickness (m)
                'gs1' (float): Surface moisture conductance (m/s)
                'ks1' (float): Surface layer thermal conductivity (W/m/K)
                'Ts1' (float): Surface layer temperature (K)
                'ksnow' (np.ndarray): Thermal conductivity of snow layers (W/m/K)
                'ksoil' (np.ndarray): Thermal conductivity of soil layers (W/m/K)
        """
        
        # read forcings
        Nsnow = forcing['Nsnow']
        Dsnw = forcing['Dsnw']
        Sice = forcing['Sice']
        Sliq = forcing['Sliq']
        Tsnow = forcing['Tsnow']
        Tsoil = forcing['Tsoil']
        ksoil = forcing['ksoil']
        kbt = forcing['kbt']
        gs1 = forcing['gs1']
        Dzsoil = forcing['Dzsoil']
        Dzbt = forcing['Dzbt']

        ksnow = np.zeros(int(self.Nsmax))
        ksnow[:] = self.kfix

        if self.CONDCT == 1:
            for k in range(Nsnow):
                rhos = self.rhof
                if self.DENSTY != 0:
                    if (Dsnw[k] > EPS):
                        rhos = (Sice[k] + Sliq[k]) / Dsnw[k]
                ksnow[k] = 2.224 * (rhos / WATER_DENSITY)**1.885
        
        # properties of snow and organic layer combined
        #Dsnw[0] += Dzbt # Dz for snow and organic layers
        #ksnow[0] = Dsnw[0] / (Dsnw[0] - Dzbt / ksnow[0] + 
        #                      Dzbt / kbt) # Effective k for stacked snow and organic layers (series conduction)
        # properties of organic layer and soil combined
        Dzsoil += Dzbt # Dz for soil and organic layers
        ksoil = Dzsoil / ((Dzsoil - Dzbt) / ksoil + 
                          Dzbt / kbt) # Effective k for stacked soil and organic layers (series conduction)

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
