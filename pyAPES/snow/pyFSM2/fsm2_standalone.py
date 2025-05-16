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
from pyAPES.snow.pyFSM2.soil import SoilModel
from pyAPES.snow.pyFSM2.thermal import Thermal
from pyAPES.snow.pyFSM2.swrad import SWrad

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
        self.soil = SoilModel(snowpara)
        self.snow = SnowModel(snowpara, soil_dz=self.soil.Dzsoil)
        self.thermal = Thermal(snowpara)
        self.swrad = SWrad(snowpara)

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

    def run(self, dt: float, forcing: dict) -> Tuple:
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

        # initial states
        snow_states = {'Sice': self.snow.Sice,
                       'Sliq': self.snow.Sliq,
                       'Nsnow': self.snow.Nsnow,
                       'Dsnw': self.snow.Dsnw,
                       'Tsnow': self.snow.Tsnow}

        soil_states = {'Tsoil': self.soil.Tsoil,
                       'Vsmc': self.soil.Vsmc,
                       }
        ebal_states = {'Tsrf': self.ebal.Tsrf}

        # RUN (shortwave radiation partitioning)
        swrad_forcing = {'Sdif': SWsrf*1.0,
                         'Sdir': 0,  # SWsrf*0.7,
                         'Sf': Sf,
                         'Tsrf': ebal_states['Tsrf'],
                         'Dsnw': snow_states['Dsnw']}
        swrad_fluxes, swrad_states = self.swrad.run(dt, swrad_forcing)
        ##

        # RUN (thermal properties of snow and soil)
        thermal_forcing = {'Nsnow': snow_states['Nsnow'],
                           'Dsnw': snow_states['Dsnw'],
                           'Sice': snow_states['Sice'],
                           'Sliq': snow_states['Sliq'],
                           'Tsnow': snow_states['Tsnow'],
                           'Tsoil': soil_states['Tsoil'],
                           'Vsmc': soil_states['Vsmc']}
        # Coupling with pyAPES:
        # most of thermal_states and _fluxes are forcing or parameters in forestfloor.py
        # coupling with soil or organiclayer?
        # ksnow calculation should be moved to ebal (srfebal.py)
        thermal_fluxes, thermal_states = self.thermal.run(thermal_forcing)
        ##

        # RUN (surface energy balance for snow or soil)       
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
                        'Sliq': snow_states['Sliq'],
                        'Dsnw': snow_states['Dsnw'],
                        'Nsnow': snow_states['Nsnow']
                        }

        ebal_fluxes, ebal_states = self.ebal.run(dt, ebal_forcing)
        ##

        # RUN (snow thermodynamics and hydrology)       
        snow_forcing = {'drip': 0,
                        'Esrf': ebal_fluxes['Esrf'],
                        'Gsrf': ebal_fluxes['Gsrf'],
                        'ksoil': thermal_states['ksoil'],
                        'ksnow': thermal_states['ksnow'],
                        'Melt': ebal_fluxes['Melt'],
                        'Rf': Rf,
                        'Sf': Sf,
                        'Ta': Ta,
                        'trans': 0,
                        'Tsrf': ebal_states['Tsrf'],
                        'unload': 0,
                        'Tsoil': soil_states['Tsoil'],
                        }
        
        snow_fluxes, snow_states = self.snow.run(dt, snow_forcing)
        ##

        # RUN (soil thermodynamics)
        soil_forcing = {'Gsoil': snow_fluxes['Gsoil'],
                        'csoil': thermal_states['csoil'],
                        'ksoil': thermal_states['ksoil']}
        # Coupling with pyAPES:
        soil_fluxes, soil_states = self.soil.run(dt, soil_forcing)
        ##

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