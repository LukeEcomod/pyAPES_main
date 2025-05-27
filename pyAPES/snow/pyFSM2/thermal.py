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
                'params' (dict):
                    'kfix' (float): Fixed thermal conductivity of snow (W/m/K)
                    'fcly' (float): Clay fraction in soil
                    'fsnd' (float): Sand fraction in soil
                    'rhof' (float): Fresh snow density (kg/m^3)
                    'gsat' (float): Saturated soil thermal conductivity (W/m/K)
                'layers' (dict):
                    'Nsoil' (int): Number of soil layers
                    'Nsmax' (int): Maximum number of snow layers
                    'Dzsoil' (np.ndarray): Soil layer thicknesses (m)
                'soilprops' (dict):
                    'bch' (float): Soil retention curve parameter
                    'hcap_soil' (float): Volumetric heat capacity of soil (J/m^3/K)
                    'hcon_soil' (float): Soil thermal conductivity (W/m/K)
                    'sathh' (float): Saturated soil matric potential (m)
                    'Vcrit' (float): Critical soil moisture content
                    'Vsat' (float): Saturated soil moisture content
                'physics_options' (dict):
                    'HYDROL' (int): Hydrology scheme selection
                    'CONDCT' (int): Soil thermal conductivity scheme selection
                    'DENSTY' (int): Snow density scheme selection

        Returns:
            self (object)
        """

        self.kfix = properties['params']['kfix']
        self.rhof = properties['params']['rhof']

        self.Nsoil = properties['layers']['Nsoil']
        self.Nsmax = properties['layers']['Nsmax']
        self.Dzsoil = properties['layers']['Dzsoil']

        self.fcly = properties['soilprops']['fcly']
        self.fsnd = properties['soilprops']['fsnd']
        self.gsat = properties['soilprops']['gsat']

        self.HYDRL = properties['physics_options']['HYDRL']
        self.CONDCT = properties['physics_options']['CONDCT']
        self.DENSTY = properties['physics_options']['DENSTY']

        # Soil properties
        self.bch = 3.1 + 15.7*self.fcly - 0.3*self.fsnd
        self.hcap_soil = (2.128*self.fcly + 2.385*self.fsnd) * \
            1e6 / (self.fcly + self.fsnd)
        self.sathh = 10**(0.17 - 0.63*self.fcly - 1.58*self.fsnd)
        self.Vsat = 0.505 - 0.037*self.fcly - 0.142*self.fsnd
        self.Vcrit = self.Vsat*(self.sathh/3.364)**(1/self.bch)
        self.hcon_soil = (K_AIR**self.Vsat) * ((K_CLAY**self.fcly)
                                               * (K_SAND**(1 - self.fcly))**(1 - self.Vsat))

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
                'Vsmc' (np.ndarray): Volumetric soil moisture content (-)
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
        Vsmc = forcing['Vsmc']

        ksoil = np.zeros(int(self.Nsoil))
        ksnow = np.zeros(int(self.Nsmax))
        csoil = np.zeros(int(self.Nsoil))

        ksnow[:] = self.kfix

        if self.CONDCT == 1:
            for k in range(Nsnow):
                rhos = self.rhof
                if self.DENSTY != 0:
                    if (Dsnw[k] > EPS):
                        rhos = (Sice[k] + Sliq[k]) / Dsnw[k]
                ksnow[k] = 2.224 * (rhos / WATER_DENSITY)**1.885

        # Heat capacity and thermal conductivity of soil
        dPsidT = -ICE_DENSITY*LATENT_HEAT_FUSION/(WATER_DENSITY*GRAVITY*T_MELT)
        for k in range(self.Nsoil):
            csoil[k] = self.hcap_soil*self.Dzsoil[k]
            ksoil[k] = self.hcon_soil
            if (Vsmc[k] > EPS):
                dthudT = 0.
                sthu = Vsmc[k]
                sthf = 0.
                Tc = Tsoil[k] - T_MELT
                Tmax = T_MELT + (self.sathh/dPsidT) * \
                    (self.Vsat/Vsmc[k])**self.bch
                if (Tsoil[k] < Tmax):
                    dthudT = (-dPsidT*self.Vsat/(self.bch*self.sathh)) * \
                        (dPsidT*Tc/self.sathh)**((-1/self.bch) - 1.)

                    sthu = self.Vsat*(dPsidT*Tc/self.sathh)**(-1./self.bch)
                    sthu = np.minimum(sthu, Vsmc[k])
                    sthf = (Vsmc[k] - sthu)*WATER_DENSITY/ICE_DENSITY
                Mf = ICE_DENSITY*self.Dzsoil[k]*sthf
                Mu = WATER_DENSITY*self.Dzsoil[k]*sthu
                csoil[k] = self.hcap_soil*self.Dzsoil[k] + SPECIFIC_HEAT_ICE*Mf + SPECIFIC_HEAT_WATER*Mu + \
                    WATER_DENSITY * \
                    self.Dzsoil[k]*((SPECIFIC_HEAT_WATER - SPECIFIC_HEAT_ICE)
                                    * Tc + LATENT_HEAT_FUSION)*dthudT
                Smf = ICE_DENSITY*sthf/(WATER_DENSITY*self.Vsat)
                Smu = sthu/self.Vsat
                thice = 0.
                if (Smf > 0):
                    thice = self.Vsat*Smf/(Smu + Smf)
                thwat = 0.
                if (Smu > 0):
                    thwat = self.Vsat*Smu/(Smu + Smf)
                hcon_sat = self.hcon_soil * \
                    (K_WATER**thwat)*(K_ICE**thice) / (K_AIR**self.Vsat)
                ksoil[k] = (hcon_sat - self.hcon_soil) * \
                    (Smf + Smu) + self.hcon_soil
                if (k == 0):
                    gs1 = self.gsat * \
                        np.maximum((Smu*self.Vsat/self.Vcrit)**2, 1.)

        # Surface layer
        Ds1 = np.maximum(self.Dzsoil[0], Dsnw[0])
        Ts1 = Tsoil[0] + (Tsnow[0] - Tsoil[0])*Dsnw[0]/self.Dzsoil[0]
        ks1 = self.Dzsoil[0]/(2*Dsnw[0]/ksnow[0] +
                              (self.Dzsoil[0] - 2*Dsnw[0])/ksoil[0])
        hs = np.sum(Dsnw)
        if (hs > 0.5*self.Dzsoil[0]):
            ks1 = ksnow[0]
        if (hs > self.Dzsoil[0]):
            Ts1 = Tsnow[0]

        fluxes = {}

        states = {'Ds1': Ds1,
                  'gs1': gs1,
                  'ks1': ks1,
                  'Ts1': Ts1,
                  'csoil': csoil,
                  'ksnow': ksnow,
                  'ksoil': ksoil,
                  }

        return fluxes, states
