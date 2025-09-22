# -*- coding: utf-8 -*-
"""
.. module: snowpack
    :synopsis: APES-model component
.. moduleauthor:: Samuli Launiainen & Kersti Leppä

*Degree-day snowpack model*
"""

import numpy as np
from typing import Dict, List, Tuple

EPS = np.finfo(float).eps  # machine epsilon

class DegreeDaySnow(object):
    def __init__(self, properties) -> object:
        """
        Zero-dimensional snowpack model based on degree-day approach.
        
        Args:
            properties (dict)
                kmelt melting coefficient [m degC-1 s-1]
                kfreeze (float): freezing coefficient coefficient [m degC-1 s-1]
                retention (float): max fraction of liquid water in snow [-]
                Tmelt (float): melting temperature (~0.0 degC) [degC]
                optical_properties (dict):
                    albedo (dict):
                        PAR (float): snow Par-albedo [-]
                        NIR (float): snow NIR-albedo [-]
                    emissivity (float): [-]
       
                initial_conditions (dict):
                    temperature (float): [degC]
                    snow_water_equivalent (float): [kg m-2 == mm]
        Returns:
            self (object)
        """

        #self.properties = properties

        # melting and freezing coefficients [kg m-2 s-1]
        self.kmelt = properties['kmelt']
        self.kfreeze = properties['kfreeze']

        # max fraction of liquid water in snow [-]
        self.retention = properties['retention']
        self.Tmelt = properties['Tmelt']

        self.optical_properties = properties['optical_properties']

        # state variables:
        self.temperature = properties['initial_conditions']['temperature']
        self.swe = properties['initial_conditions']['snow_water_equivalent']  # [kg m-2]
        self.ice = properties['initial_conditions']['snow_water_equivalent'] # ice content
        self.liq = 0.0  # liquid water storage in snowpack [kg m-2]

        # temporary storage of iteration results
        self.iteration_state = None

    def update(self):
        """ 
        Updates snowpack state.
        """
        self.temperature = self.iteration_state['temperature']
        self.ice = self.iteration_state['ice']
        self.liq = self.iteration_state['liq']
        self.swe = self.iteration_state['swe']

    def run(self, dt: float, forcing: Dict) -> Tuple:
        """
        Calculates one timestep and updates snowpack state

        Args:
            dt (float): timestep [s]
            forcing' (dict):
                air_temperature: [degC]
                precipitation_rain: [kg m-2 s-1]
                precipitation_snow: [kg m-2 s-1]

        Returns
            (tuple):
            fluxes (dict):
               potential_infiltration: [kg m-2 s-1]
               water_closure: [kg m-2 s-1]
            states (dict):
               snow_water_equivalent: [kg m-2]
               temperature: [degC]
        """

        """ --- melting and freezing in snowpack --- """
        if forcing['air_temperature'] >= self.Tmelt:
            # [m]
            melt = np.minimum(self.ice,
                              self.kmelt * dt * (forcing['air_temperature'] - self.Tmelt))
            freeze = 0.0

        else:
            melt = 0.0
            freeze = np.minimum(self.liq,
                                self.kfreeze * dt * (self.Tmelt - forcing['air_temperature']))

        """ --- update state of snowpack and compute potential infiltration --- """
        ice = np.maximum(0.0,
                         self.ice + forcing['precipitation_snow'] * dt + freeze - melt)

        liq = np.maximum(0.0,
                         self.liq + forcing['precipitation_rain'] * dt - freeze + melt)

        pot_inf = np.maximum(0.0, liq - ice * self.retention)

        # liquid water and ice in snow, and snow water equivalent [m]
        liq = np.maximum(0.0, liq - pot_inf)
        ice = ice
        swe = liq + ice

        # mass-balance error [kg m-2]
        water_closure = ((swe - self.swe)
                         - (forcing['precipitation_rain'] * dt + forcing['precipitation_snow'] * dt - pot_inf))

        # store iteration state
        self.iteration_state =  {'temperature': forcing['air_temperature'],
                                 'swe': swe,
                                 'ice': ice,
                                 'liq': liq}
        

        fluxes = {'potential_infiltration': pot_inf / dt,
                  'water_closure': water_closure / dt
                 }

        states = {'snow_water_equivalent': swe,
                  'surface_temperature': forcing['air_temperature']
                 }
        
        return fluxes, states

# EOF