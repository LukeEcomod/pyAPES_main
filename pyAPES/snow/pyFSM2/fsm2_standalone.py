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
from pyAPES.snow.pyFSM2.soil import SoilModel
from pyAPES.snow.pyFSM2.thermal_standalone import Thermal
from pyAPES.utils.constants import DEG_TO_KELVIN


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
        self.soil = SoilModel(snowpara)

        self.swe = np.sum(snowpara['initial_conditions']['Sice']) + np.sum(snowpara['initial_conditions']['Sliq'])
        self.ice = np.sum(snowpara['initial_conditions']['Sice'])
        self.liq = np.sum(snowpara['initial_conditions']['Sliq'])
        self.temperature = snowpara['initial_conditions']['Tsnow']

        # parameters
        self.alb0 = snowpara['soilprops']['alb0']
        self.z0sf = snowpara['soilprops']['z0sf']
        self.reference_height = snowpara['params']['zU']

        # temporary storage of iteration results
        self.iteration_state = None
        self.optical_properties = None

    def update(self):
        """
        Updates snowpack state.
        """
        # updating submodules' states
        self.swrad.update()
        self.ebal.update()
        self.snow.update()
        self.soil.update()
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
                reference_height (float): [m]

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
        
        Dzsoil = self.soil.Dzsoil # Soil layer thickness, used in snow
        alb0 = self.alb0 # Snow-free surface albedo
        z0sf = self.z0sf # Snow-free roughness length
        reference_height = self.reference_height # zU

        # initial states
        snow_states = {'Sice': self.snow.Sice,
                       'Sliq': self.snow.Sliq,
                       'Nsnow': self.snow.Nsnow,
                       'Dsnw': self.snow.Dsnw,
                       'Tsnow': self.snow.Tsnow}

        ebal_states = {'Tsrf': self.ebal.Tsrf}

        soil_states = {'Tsoil': self.soil.Tsoil,
                       'Vsmc': self.soil.Vsmc}


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
                        'Tsoil': soil_states['Tsoil'],
                        'Dzsoil': Dzsoil,
                        'Vsmc': soil_states['Vsmc']}
        
        thermal_fluxes, thermal_states = self.thermal.run(thermal_forcing)

        ebal_forcing = {'Ds1': thermal_states['Ds1'], # surface layer properties
                        'gs1': thermal_states['gs1'], #
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
                        'ksoil': thermal_states['ksoil'][0],
                        'ksnow': thermal_states['ksnow'],
                        'Melt': ebal_fluxes['Melt'],
                        'Rf': Rf,
                        'Sf': Sf,
                        'Ta': Ta,
                        'trans': 0,
                        'Tsrf': ebal_states['Tsrf'],
                        'unload': 0,
                        'Tsoil': soil_states['Tsoil'][0],
                        'Dzsoil': Dzsoil[0]
                        }
                
        snow_fluxes, snow_states = self.snow.run(dt, snow_forcing)

        # RUN (soil thermodynamics)
        soil_forcing = {'Gsoil': snow_fluxes['Gsoil'],
                        'csoil': thermal_states['csoil'],
                        'ksoil': thermal_states['ksoil']}
        soil_fluxes, soil_states = self.soil.run(dt, soil_forcing)


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
                'snow_heat_flux': snow_fluxes['Gsoil'],
                'snow_longwave_out': ebal_fluxes['LWout'],
                'snow_sensible_heat': ebal_fluxes['H'],
                'snow_latent_heat': ebal_fluxes['LE'],
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
                'snow_density': snow_states['rhos']
                }

        return fluxes, states

# EOF