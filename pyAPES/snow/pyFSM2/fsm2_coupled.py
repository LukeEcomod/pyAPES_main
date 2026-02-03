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
from pyAPES.snow.pyFSM2.thermal_coupled import Thermal
from pyAPES.utils.constants import EPS, DEG_TO_KELVIN


class FSM2(object):
    def __init__(self, snowpara) -> object:
        """        
        Args:
            snowpara (Dict):
        Returns:
            self (object)
        """

        # initializing process modules
        self.swrad = SWrad(snowpara)
        self.ebal = EnergyBalance(snowpara)
        self.snow = SnowModel(snowpara)
        self.thermal = Thermal(snowpara)

        self.swe = np.sum(snowpara['initial_conditions']['Sice']) + np.sum(snowpara['initial_conditions']['Sliq'])
        self.ice = np.sum(snowpara['initial_conditions']['Sice'])
        self.liq = np.sum(snowpara['initial_conditions']['Sliq'])
        self.temperature = snowpara['initial_conditions']['Tsnow']

        # temporary storage of iteration results
        self.iteration_state = None
        self.optical_properties = None

    def update(self):
        """
        Updates snowpack state.
        """
        if self.iteration_state is not None and (
            self.iteration_state['swe'] > 0 or self.swe > 0): # current iteration or previous timestep swe > 0
            # updating submodules' states
            self.swrad.update()
            self.ebal.update()
            self.snow.update()
            # updating snowmodel states
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
                Tsoil (float): soil temperature (K)
                ksoil (float): # Thermal conductivity of first soil layer (W/m/K)
                Dzsoil (float): # 
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
        reference_height = forcing['reference_height']
        
        gs1 = forcing['gs1'] # Surface moisture conductance (m/s), used in ebal
        Tsoil = forcing['Tsoil'] # Uppermost soil temperature, used in snow
        Tsoil_surf = forcing['Tsoil_surf'] # Surface temperature
        ksoil = forcing['ksoil'] # Soil layer thermal conductivity (W/m/K), used in ebal
        Dzsoil = forcing['Dzsoil'] # Soil layer thickness, used in snow
        alb0 = forcing['alb0'] # Snow-free surface albedo
        z0sf = forcing['z0sf'] # Snow-free roughness length

        # initial states
        snow_states = {'Sice': self.snow.Sice,
                       'Sliq': self.snow.Sliq,
                       'Nsnow': self.snow.Nsnow,
                       'Dsnw': self.snow.Dsnw,
                       'Tsnow': self.snow.Tsnow}

        ebal_states = {'Tsrf': self.ebal.Tsrf}

        if sum(self.snow.Dsnw) == 0.: # no existing snowpack -> surface temperature from organiclayer.py
            ebal_states = {'Tsrf': Tsoil_surf}

        if Sf > 0 or sum(self.snow.Dsnw) > 0: # solving new or existing snowpack -> surface temperature from fsm

            swrad_forcing = {'Sdif': SWsrf*1.0,
                            'Sdir': 0,  # SWsrf*0.7,
                            'Sf': Sf,
                            'Tsrf': ebal_states['Tsrf'],
                            'Dsnw': snow_states['Dsnw'],
                            'alb0': alb0}

            swrad_fluxes, swrad_states = self.swrad.run(dt, swrad_forcing)

            thermal_forcing = {'Nsnow': snow_states['Nsnow'],
                            'Dsnw': snow_states['Dsnw'],
                            'Sice': snow_states['Sice'],
                            'Sliq': snow_states['Sliq'],
                            'Tsnow': snow_states['Tsnow'],
                            'Tsoil': Tsoil,
                            'ksoil': ksoil,
                            'gs1': gs1,
                            'Dzsoil': Dzsoil}
            
            thermal_fluxes, thermal_states = self.thermal.run(thermal_forcing)

            ebal_forcing = {'Ds1': thermal_states['Ds1'], # surface layer properties
                            'gs1': gs1, #
                            'Ts1': thermal_states['Ts1'], #
                            'ks1': thermal_states['ks1'], #
                            'LW': LW, # canopy forcing
                            'Ps': Ps, #
                            'RH': RH, #
                            'Ta': Ta, #
                            'Ua': Ua, #
                            'fsnow': swrad_states['fsnow'], # snow model forcing
                            'SWsrf': swrad_fluxes['SWsrf'], #
                            'Sice': snow_states['Sice'], #
                            'Sliq': snow_states['Sliq'], #
                            'Dsnw': snow_states['Dsnw'], #
                            'Nsnow': snow_states['Nsnow'], #
                            'reference_height': reference_height, #
                            'Tsrf': ebal_states['Tsrf'],
                            'z0sf': z0sf
                            }

            ebal_fluxes, ebal_states = self.ebal.run(dt, ebal_forcing)

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
                            'Dzsoil': Dzsoil
                            }
                    
            snow_fluxes, snow_states = self.snow.run(dt, snow_forcing)

            # store iteration state
            self.iteration_state = {'temperature': ebal_states['Tsrf'],
                                    'swe': snow_states['swe'],
                                    'ice': snow_states['Sice'],
                                    'liq': snow_states['Sliq'],
                                    }
            
            self.optical_properties = {
                    'emissivity': 1.0,
                    'albedo': {'PAR': swrad_states['srf_albedo'], 
                            'NIR': swrad_states['srf_albedo']}
                    }

            fluxes = {'potential_infiltration': snow_fluxes['Roff'],
                    'snow_heat_flux': snow_fluxes['Gsoil'],  # heat flux to organiclayer
                    'snow_longwave_out': ebal_fluxes['LWout'],
                    'snow_sensible_heat': ebal_fluxes['H'],
                    'snow_latent_heat': ebal_fluxes['LE'],
                    'snow_net_radiation': ebal_fluxes['Rsrf'],
                    'snow_ustar': ebal_fluxes['ustar'],
                    'snow_ga': ebal_fluxes['ga'],
                    'snow_energy_closure': ebal_fluxes['ebal']
                    }

            states = {'snow_water_equivalent': snow_states['swe'],
                    'temperature': ebal_states['Tsrf'] - DEG_TO_KELVIN,
                    'snow_depth': snow_states['hs'],
                    'snow_albedo': swrad_states['snow_albedo'],
                    'srf_albedo': swrad_states['srf_albedo'],
                    'snow_fraction': swrad_states['fsnow'],
                    'snow_stability_factor': ebal_states['rL'],
                    'snow_temperature': snow_states['Tsnow'] - DEG_TO_KELVIN,
                    'snow_layer_depth': snow_states['Dsnw'],
                    'snow_liquid_storage': snow_states['Sliq'],
                    'snow_ice_storage': snow_states['Sice'],
                    'snow_density': snow_states['rhos'],
                    'snow_ks1': thermal_states['ks1'],
                    'snow_layers': snow_states['Nsnow']
                    }
        
        else: # no new or existing snowpack
            # store iteration state
            self.iteration_state = {'temperature': ebal_states['Tsrf'] - DEG_TO_KELVIN,
                                    'swe': 0.,
                                    'ice': 0.,
                                    'liq': 0.,
                                    }
            
            self.optical_properties = {
                    'emissivity': 1.0,
                    'albedo': {'PAR': 0.1, 
                            'NIR': 0.1}
                    }

            fluxes = {'potential_infiltration': Rf,
                    'snow_heat_flux': 0., 
                    'snow_longwave_out': 0.,
                    'snow_sensible_heat': 0.,
                    'snow_latent_heat': 0.,
                    'snow_ustar': 0.,
                    'snow_ga': 0.
                    }

            states = {'snow_water_equivalent': 0.,
                    'temperature': ebal_states['Tsrf'] - DEG_TO_KELVIN,
                    'snow_depth': 0.,
                    'snow_temperature': np.array([np.nan, np.nan, np.nan]),
                    'snow_layer_depth': np.nan,
                    'snow_liquid_storage': 0.,
                    'snow_ice_storage': 0.,
                    'snow_density': np.nan,
                    'snow_layers': snow_states['Nsnow']
                    }

        return fluxes, states

# EOF