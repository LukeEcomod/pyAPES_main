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
from pyAPES.utils.constants import DEG_TO_KELVIN

class FSM2(object):
    def __init__(self, snowpara) -> object:
        """        
        Args:
            snowpara (dict):
                'initial_conditions' (dict)
                    'Sice' (np.ndarray): # Snow ice content (kg/m^2)
                    'Sliq' (np.ndarray): # Snow liquid content (kg/m^2)
                    'Tsrf' (float): # Snow/ground surface temperature (K)
                'layers' (dict)
                    'Nsmax' (int): # Maximum number of snow layers
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
        self.surface_temperature = snowpara['initial_conditions']['Tsrf']
        self.Nsmax = snowpara['layers']['Nsmax']

        # temporary storage of iteration results
        self.iteration_state = None
        self.optical_properties = {
        'emissivity': 1.0,
        'albedo': {'PAR': 0.1, 
                'NIR': 0.1}
        }

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
            self.surface_temperature = self.iteration_state['snow_surface_temperature']
            self.ice = self.iteration_state['ice']
            self.liq = self.iteration_state['liq']
            self.swe = self.iteration_state['swe']

    def run(self, dt: float, forcing: dict) -> Tuple:
        """
        Calculates one timestep

        Args:
            dt (float): timestep (s)
            forcing (dict):
                'SWsrf' (float): surface shortwave radiation (W/m2) 
                'Sf' (float): snowfall rate (kg/m2/s)
                'Rf' (float): rainfall rate (kg/m2/s)
                'LW' (float): surface longwave radiation (W/m2)
                'Ps' (float): atmospheric pressure (?)
                'RH' (float): relative humidity [-]
                'Ta' (float): air temperature (K)
                'Ua' (float): wind speed (m/s)
                'reference_height' (float): first canopy calculation node or forcing height [m]
                'gs1' (float): Surface moisture conductance (m/s)
                'Tsoil' (float): soil temperature (K)
                'Tsoil_surf' (float): # Surface temperature [K]
                'ksoil' (float): Thermal conductivity of first soil layer (W/m/K)
                'kbt' (float):  Thermal conductivity of organic layer (W/m/K)
                'Dzsoil' (float): Soil layer thickness (m)
                'Dzbt' (float):  Thickness of organic layer (m)
                'alb0' (float): Surface albedo [-]
                'z0sf' (float): Surface roughness length [m]

        Returns:
            Tuple:
                fluxes (dict):
                    'potential_infiltration' (float):
                    'snow_heat_flux' (float): Heat flux into snow/ground surface (W/m^2)
                    'snow_longwave_out' (float): Outgoing LW radiation (W/m^2)
                    'snow_shortwave_out' (float): Outgoing SW radiation (W/m^2)
                    'snow_sensible_heat' (float): Sensible heat flux to the atmosphere (W/m^2)
                    'snow_latent_heat' (float): Latent heat flux to the atmosphere (W/m^2)
                    'snow_net_radiation' (float): Net radiation (W/m^2)
                    'snow_energy_closure' (float): Energy balance closure (W/m^2)
                    'snow_water_closure' (float): Water balance closure (m)
                states (dict):
                    'snow_water_equivalent' (float): # Total snow mass on ground (kg/m^2)
                    'snow_depth' (float): # Snow depth (m)
                    'snow_albedo' (float): Snow albedo
                    'snow_fraction' (float): Snow cover fraction
                    'snow_temperature' (np.ndarray): # Snow layer temperatures (K)
                    'snow_layer_depth' (np.ndarray): # Snow layer thicknesses (m)
                    'snow_liquid_storage' (np.ndarray): # Snow liquid content (kg/m^2)
                    'snow_ice_storage' (np.ndarray): # Snow ice content (kg/m^2)
                    'snow_density' (np.ndarray): # Snow layer densities (kg/m^3)
                    'snow_layers' (float): Number of snow layers
        """

        SWsrf = forcing['SWsrf']
        Sf = forcing['Sf']
        Rf = forcing['Rf']
        LW = forcing['LW']
        Ps = forcing['Ps']
        RH = forcing['RH']
        Ta = forcing['Ta']
        Ua = forcing['Ua']
        reference_height = forcing['reference_height'] 
        gs1 = forcing['gs1']
        Tsoil = forcing['Tsoil']
        Tsoil_surf = forcing['Tsoil_surf']
        ksoil = forcing['ksoil']
        kbt = forcing['kbt']
        Dzsoil = forcing['Dzsoil']
        Dzbt = forcing['Dzbt']
        alb0 = forcing['alb0']
        z0sf = forcing['z0sf']

        # initial states
        snow_states = {'Sice': self.snow.Sice,
                       'Sliq': self.snow.Sliq,
                       'Nsnow': self.snow.Nsnow,
                       'Dsnw': self.snow.Dsnw,
                       'Tsnow': self.snow.Tsnow}

        ebal_states = {'Tsrf': self.ebal.Tsrf}

        # initialize fluxes and states
        fluxes = {'potential_infiltration': Rf,
                'snow_heat_flux': 0.,
                'snow_longwave_out': 0.,
                'snow_shortwave_out': 0.,
                'snow_sensible_heat': 0.,
                'snow_latent_heat': 0.,
                'snow_net_radiation': 0.,
                'snow_energy_closure': 0.,
                'snow_water_closure': 0.
                }
        
        states = {'snow_water_equivalent': 0.,
                'snow_surface_temperature': ebal_states['Tsrf'] - DEG_TO_KELVIN,
                'snow_depth': 0.,
                'snow_albedo': 0.,
                'snow_fraction': 0.,
                'snow_temperature': np.full(self.Nsmax, np.nan),
                'snow_layer_depth': np.full(self.Nsmax, np.nan),
                'snow_liquid_storage': np.full(self.Nsmax, np.nan),
                'snow_ice_storage': np.full(self.Nsmax, np.nan),
                'snow_density': np.full(self.Nsmax, np.nan),
                'snow_layers': snow_states['Nsnow']
                }
        
        self.iteration_state = {'snow_surface_temperature': ebal_states['Tsrf'] - DEG_TO_KELVIN,
                                'swe': 0.,
                                'ice': 0.,
                                'liq': 0.,
                                }

        if sum(self.snow.Dsnw) == 0.: # no existing snowpack -> surface temperature from organiclayer.py
            ebal_states = {'Tsrf': Tsoil_surf}

        if Sf > 0 or sum(self.snow.Dsnw) > 0: # solving new or existing snowpack -> surface temperature from fsm

            swrad_forcing = {'Sdif': SWsrf*1.0,
                            'Sdir': 0.,
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
                            'kbt': kbt,
                            'gs1': gs1,
                            'Dzsoil': Dzsoil,
                            'Dzbt': Dzbt}
            
            thermal_fluxes, thermal_states = self.thermal.run(thermal_forcing)

            ebal_forcing = {'Ds1': thermal_states['Ds1'], # surface layer properties
                            'gs1': gs1, #
                            'Ts1': thermal_states['Ts1'], #
                            'ks1': thermal_states['ks1'], #
                            'LW': LW, #
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
                            'trans': 0.,
                            'Tsrf': ebal_states['Tsrf'],
                            'unload': 0.,
                            'Tsoil': Tsoil,
                            'Dzsoil': Dzsoil
                            }
                    
            snow_fluxes, snow_states = self.snow.run(dt, snow_forcing)

            # store iteration state
            self.iteration_state = {'snow_surface_temperature': ebal_states['Tsrf'],
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
                    'snow_shortwave_out': swrad_fluxes['SWout'],
                    'snow_sensible_heat': ebal_fluxes['H'],
                    'snow_latent_heat': ebal_fluxes['LE'],
                    'snow_net_radiation': ebal_fluxes['Rsrf'],
                    'snow_energy_closure': ebal_fluxes['ebal'],
                    'snow_water_closure': snow_fluxes['wbal']
                    }

            states = {'snow_water_equivalent': snow_states['swe'],
                    'snow_surface_temperature': ebal_states['Tsrf'] - DEG_TO_KELVIN,
                    'snow_depth': snow_states['hs'],
                    'snow_albedo': swrad_states['snow_albedo'],
                    'snow_fraction': swrad_states['fsnow'],
                    'snow_temperature': snow_states['Tsnow'] - DEG_TO_KELVIN,
                    'snow_layer_depth': snow_states['Dsnw'],
                    'snow_liquid_storage': snow_states['Sliq'],
                    'snow_ice_storage': snow_states['Sice'],
                    'snow_density': snow_states['rhos'],
                    'snow_layers': snow_states['Nsnow']
                    }

        return fluxes, states

# EOF