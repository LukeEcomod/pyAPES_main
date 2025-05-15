"""
.. module: snow
    :synopsis: APES-model component
.. moduleauthor:: Jari-Pekka Nousu

*Energy-balance (FSM2) based snowpack model driver*

"""

import numpy as np
from typing import Dict, List, Tuple

from pyAPES.snow.pyFSM2.snow import SnowModel
from pyAPES.snow.pyFSM2.srfebal import EnergyBalance
from pyAPES.snow.pyFSM2.swrad import SWrad
from pyAPES.snow.pyFSM2.solarpos import SolarPos

EPS = np.finfo(float).eps  # machine epsilon

class FSM2(object):
    def __init__(self, snowpara) -> object:
        """        
        Args:
            snowpara (Dict):
        Returns:
            self (object)
        """

        # initializing process modules
        self.ebal = EnergyBalance(snowpara)
        self.snow = SnowModel(snowpara)
        self.swrad = SWrad(snowpara)
        self.solarpos = SolarPos()

        self.swe = np.sum(snowpara['initial_conditions']['Sice']) + np.sum(snowpara['initial_conditions']['Sliq'])

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

    def run(self, dt: float, forcing: dict, parameters: dict) -> Tuple:
        """

        Args:
            dt (float): timestep (s)
            forcing (dict):
                SWsrf (float): surface shortwave radiation (W/m2) 
                Sf (float): snowfall rate (kg/m2/s)
                Rf (float): rainfall rate (kg/m2/s)
                LW (float): surface longwave radiation (W/m2)
                Ps (float): atmospheric pressure (?)
                Ta (float): air temperature (K)
                Ua (float): wind speed (m/s)
                Tsoil (float): soil temperature (K)
                Vsmc (float): soil volumetric water content (K)
                ksoil (float): soil thermal conductivity ()
            parameters (dict):
                Dzsoil (float): # First soil layer thickness
                ksoil (float): # Thermal conductivity of first soil layer (W/m/K)
                reference_height (float): first canopy calculation node [m]

        Returns:
            Tuple:
                fluxes (dict):
                    potential_infiltration (float):
                states (dict):
                    swe (float):
                    snow_water_equivalent (float):
                    snow_depth (float):
        """

        fluxes = {}

        SWsrf = forcing['SWsrf']
        Sf = forcing['Sf']
        Rf = forcing['Rf']
        LW = forcing['LW']
        Ps = forcing['Ps']
        RH = forcing['RH']
        Ta = forcing['Ta']
        Ua = forcing['Ua']
        Tsoil = forcing['Tsoil']
        Vsmc = forcing['Vsmc']

        ksoil = parameters['ksoil']
        Dzsoil = parameters['Dzsoil']
        zU = parameters['reference_height']
        zT = parameters['reference_height']

        snow_states = {'Sice': self.snow.Sice,
                       'Sliq': self.snow.Sliq,
                       'Nsnow': self.snow.Nsnow,
                       'Dsnw': self.snow.Dsnw,
                       'Tsnow': self.snow.Tsnow}

        ebal_states = {'Tsrf': self.ebal.Tsrf}

        swrad_forcing = {'Sdif': SWsrf*1.0,
                         'Sdir': 0,  # SWsrf*0.7,
                         'Sf': Sf,
                         'Tsrf': ebal_states['Tsrf'],
                         'Dsnw': snow_states['Dsnw']}

        swrad_fluxes, swrad_states = self.swrad.run(dt, swrad_forcing)

        thermal_forcing = {'Nsnow': snow_states['Nsnow'],
                           'Dsnw': snow_states['Dsnw'],
                           'Sice': snow_states['Sice'],
                           'Sliq': snow_states['Sliq'],
                           'Tsnow': snow_states['Tsnow'],
                           'Tsoil': Tsoil,
                           'Vsmc': Vsmc}

        thermal_fluxes, thermal_states = self.thermal.run(thermal_forcing)

        ebal_forcing = {'Ds1': thermal_states['Ds1'],
                        'fsnow': swrad_states['fsnow'],
                        'gs1': thermal_states['gs1'],
                        'ks1': thermal_states['ks1'],
                        'LW': LW,
                        'Ps': Ps,
                        'RH': RH,
                        'SWsrf': swrad_fluxes['SWsrf'],
                        'Ta': Ta,
                        'Ts1': thermal_states['Ts1'],
                        'Ua': Ua,
                        'Sice': snow_states['Sice'],
                        }
        
        ebal_parameters = {'zU': zU,
                           'zT': zT}

        ebal_fluxes, ebal_states = self.ebal.run(dt, ebal_forcing, ebal_parameters)

        snow_forcing = {'drip': 0,
                        'Esrf': ebal_fluxes['Esrf'],
                        'Gsrf': ebal_fluxes['Gsrf'],
                        'ksoil': ksoil,
                        'ksnow': thermal_states['ksnow'],
                        'Melt': ebal_fluxes['Melt'],
                        'Rf': Rf,
                        'Sf': Sf,
                        'Ta': Ta,
                        'trans': 0,
                        'Tsrf': ebal_states['Tsrf'],
                        'unload': 0,
                        'Tsoil': Tsoil,
                        'ksnow': thermal_states['ksnow'],
                        }
        
        snow_parameters = {'ksoil': ksoil,
                           'Dzsoil': Dzsoil}
        
        snow_fluxes, snow_states = self.snow.run(dt, snow_forcing, snow_parameters)

        # store iteration state
        self.iteration_state = {'temperature': ebal_states['Tsrf'],
                                'swe': snow_states['swe'],
                                'ice': snow_states['Sice'],
                                'liq': snow_states['Sliq']}

        fluxes = {'potential_infiltration': snow_fluxes['Roff'],
                  }

        states = {'snow_water_equivalent': snow_states['swe'],
                  'temperature': ebal_states['Tsrf'],
                  'snow_depth': snow_states['hs']
                  }

        return fluxes, states

# EOF