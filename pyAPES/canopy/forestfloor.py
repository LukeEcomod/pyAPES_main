# -*- coding: utf-8 -*-
"""
.. module: forestfloor
    :synopsis: pyAPES-model canopy component
.. moduleauthor:: Antti-Jussi Kieloaho, Samuli Launiainen, Kersti Leppä

*Forest bottom layer combined water, energy and carbon balance*

In pyAPES, canopy.forestfloor is used to compute energy balance, water and carbon exchange
of the forestfloor or peatland ground layer. It handles fractional tiling of different 
bottomlayer (moss, litter, bare soil) types (pyAPES.bottomlayer -package), and
overlying snowpack (pyAPES.snow -package).

Called from pyAPES.canopy.canopy.CanopyModel

Next developments planned:
    - snowpack: energy balance and snowdepth changes with a process model
    - bare soil energy balance implementation (see pyAPES.bottomlayer.baresoil)
    - organic layer freezing/thawing (to pyAPES.bottomlayer.organiclayer)
    - replace bulk soil respiration model with separate soil C module (pyAPES.soil) + autotrophic respiration (pyAPES.planttype)

"""
import numpy as np
from typing import List, Dict, Tuple
import logging
from pyAPES.utils.constants import EPS, MOLAR_MASS_H2O, LATENT_HEAT

from pyAPES.bottomlayer.organiclayer import OrganicLayer
from pyAPES.snow.snowpack import DegreeDaySnow
from pyAPES.bottomlayer.carbon import SoilRespiration

logger = logging.getLogger(__name__)

class ForestFloor(object):
    r"""
    ForestFloor computes energy balance, water and carbon exchange
    of forestfloor or peatland ground layer.
   """
    def __init__(self, para: Dict, respiration_profile: np.ndarray=None) -> object:
        """
        Initializes forestfloor object
        
        Args:
             para (dict):
             bottom_layer_types (list of dicts)
                name (str)
                layer_type (str), 'bryophyte' or 'litter'
                coverage (float): [-]
                height (float): [m]
                roughness_height(float): [m], for moss-air conductance
                bulk_density (float): [kg m-3]
                max_water_content (float): [g g-1]
                max_symplast_water_content (float): [g g-1]
                min_water_content (float): [g g-1]
                water_retention (dict), micropore system
                    # theta_s [m3 m-3], saturated water content; computed from max_symplast_water_content & bulk_density
                    # theta_r [m3 m-3], residual water content; computed from min_water_content & bulk_density
                    alpha [cm-1] air entry suction
                    n [-], pore size distribution
                    saturated conductivity': [m s-1]
                    pore connectivity': (l) [-]
                porosity (float): [m3 m-3], macroporosity
                photosynthesis (dict): community-level photosyntesis model, only if layer_type == 'bryophyte'
                    Vcmax (float): [umol m-2 (ground) s-1], maximum carboxylation velocity at 25 C
                    Jmax (float): [umol m-2 (ground) s-1], maximum electron transport rate at 25 C
                    Rd (float): [umol m-2 (ground) s-1], dark respiration rate at 25 C
                    alpha (float): [mol mol-1], quantum efficiency
                    theta (float): [-], curvature parameter
                    beta (float): [-], co-limitation parameter
                    
                    gopt (float): [mol m-2 (ground) s-1], conductance for CO2 at optimum water content
                    wopt (float): [g g-1], parameter of conductance - water content relationship
                    a0 (float): [-], parameter of conductance - water content curve
                    a1 (float): [-] parameter of conductance - water content curve
                    CAP_desic (list) [-], parameters of desiccation curve
                    tresp (dict): parameters of temperature response curve
                        Vcmax (list): [activation energy [kJ mol-1], 
                                     deactivation energy [kJ mol-1],
                                     entropy factor [kJ mol-1]]
                                        
                        Jmax (list): [activation energy [kJ mol-1], 
                                     deactivation energy [kJ mol-1],
                                     entropy factor [kJ mol-1]]
                        Rd (list): [activation energy [kJ mol-1]]
                
                respiration (dict): only if layer_type == 'litter'
                    q10 (float): [-], temperature sensitivity
                    r10 (float): [umol m-1 (ground) s-1], base respiration at 10 C
                
                optical_properties (dict):
                    albedo (dict):
                        PAR [-];
                        NIR [-];
                        emissivity [-]
         
                initial_conditions (dict)
                    temperature (float): [degC]
                    water_content (float): [g g-1]
                
            snowpack (dict):
                kmelt (float): [m K-1 s-1], melting coefficient
                kfreeze (float): [m K-1 s-1], freezing  coefficient 
                retention (float): [-], max fraction of liquid water in snow
                Tmelt (float): [degC], melting temperature
                optical_properties (dict):
                    albedo (dict):
                        PAR [-]
                        NIR [-]
                    emissivity [-]
                initial_conditions (dict):
                    temperature (float): [degC]
                    snow_water_equivalent (float): [kg m-2 == mm]
                    
            soil_respiration (dict)
                r10 (float): [umol m-2 (ground) s-1], base respiration rate
                q10 (float): [-] temperature sensitivity [-]
                moisture_coeff (list), moisture response parameters
        
        Returns:
            
            self (object)
        
        """

        # -- forest floor tiled surface of organic layers. snowpack can overly ground
        self.snowpack = DegreeDaySnow(para['snowpack'])

        self.soilrespiration = SoilRespiration(para['soil_respiration'], weights=respiration_profile)

        # BottomLayer types can include both bryophytes and litter. Append to list and
        # compute area-weighted forest floor temperature and optical properties
        bltypes = []
        bltnames = list(para['bottom_layer_types'].keys())
        bltnames.sort()

        for bt in bltnames:
            if para['bottom_layer_types'][bt]['coverage'] > 0:
                # case coverage > 0:
                bltypes.append(OrganicLayer(para['bottom_layer_types'][bt]))
            else:
                logger.info('Forestfloor: %s, coverage 0.0 omitted!', bt)

        f_organic = sum([bt.coverage for bt in bltypes])
        if abs(f_organic - 1.0) > EPS:
            raise ValueError('The sum of organic type coverage must = 1, ' +
                             'now %.2f' % f_organic)

        self.bottomlayer_types = bltypes
        logger.info('Forestfloor has %s bottomlayer types', len(self.bottomlayer_types))

        if self.snowpack.swe > 0:
            self.temperature = self.snowpack.temperature
            self.surface_temperature = self.snowpack.temperature
            self.albedo = self.snowpack.optical_properties['albedo']
            self.emissivity = self.snowpack.optical_properties['emissivity']
        else:
            self.temperature = sum([bt.coverage * bt.temperature
                                    for bt in self.bottomlayer_types])
            self.surface_temperature = sum([bt.coverage * bt.surface_temperature
                                    for bt in self.bottomlayer_types])
            self.albedo = {'PAR': sum([bt.coverage * bt.albedo['PAR']
                                       for bt in self.bottomlayer_types]),
                           'NIR': sum([bt.coverage * bt.albedo['NIR']
                                       for bt in self.bottomlayer_types])}
            self.emissivity = sum([bt.coverage * bt.emissivity
                                   for bt in self.bottomlayer_types])
        self.water_storage = sum([bt.coverage * bt.water_storage
                                  for bt in self.bottomlayer_types])


    def update(self) -> None:
        """ 
        Updates forestfloor-object state variables
        """
        self.snowpack.update()

        for bt in self.bottomlayer_types:
            bt.update_state()

        if self.snowpack.swe > 0:
            self.temperature = self.snowpack.temperature
            self.surface_temperature = self.snowpack.temperature
            self.albedo = self.snowpack.optical_properties['albedo']
            self.emissivity = self.snowpack.optical_properties['emissivity']
        else: # NOTE! forestfloor temperature in snow-free conditions is weighted average of moss surface temperature!
            self.temperature = sum([bt.coverage * bt.temperature
                                    for bt in self.bottomlayer_types])
            self.surface_temperature = sum([bt.coverage * bt.surface_temperature
                                            for bt in self.bottomlayer_types])
            self.albedo['PAR'] = sum([bt.coverage * bt.albedo['PAR']
                                      for bt in self.bottomlayer_types])
            self.albedo['NIR'] = sum([bt.coverage * bt.albedo['NIR']
                                      for bt in self.bottomlayer_types])
            self.emissivity = sum([bt.coverage * bt.emissivity
                                   for bt in self.bottomlayer_types])
            self.water_storage = sum([bt.coverage * bt.water_storage
                                      for bt in self.bottomlayer_types])

    def run(self, dt: float, forcing: Dict, parameters: Dict, controls: Dict) -> Tuple:
        """
        Computes water, energy and CO2 balance at the forestfloor over timestep dt.
        Handles tiled OrganicLayer types at forest floor, and averages fluxes and state.

        Args:
            forcing (dict):
                precipitation_rain [kg m-2 s-1 ]
                precipitation_snow [kg m-2 s-1 ]
                par [W m-2]
                nir [W m-2] if energy_balance is True
                lw_dn [W m-2] if energy_balance is True
                h2o [mol mol-1]
                co2 [ppm]
                air_temperature [degC]
                air_pressure [Pa]
                soil_temperature': [degC]
                soil_water_potential [Pa]
                soil_volumetric_water [m3 m-3]
                soil_volumetric_air [m3 m-3]
                soil_pond_storage [kg m-2]
            
            parameters (dict):
                soil_thermal_conductivity [W m-1 K-1] (if controls['energy balance'] = True)
                soil_hydraulic_conductivity [m s-1]
                depth [m] of first soil calculation node
                reference_height [m] of first canopy calculation node
           
            controls (dict):
                energy_balance (bool)
                logger_info (str)
        
        Returns:
            (tuple):    
                fluxes (dict): forestfloor aggregated fluxes
                    net_radiation [W m-2]
                    sensible_heat [W m-2]
                    latent_heat [W m-2]
                    ground_heat [W m-2]
                    energy_closure [W m-2]
                    evaporation [kg m-2 s-1],  tiles + soil below
                    soil_evaporation [kg m-2 s-1], from soil
                    throughfall [kg m-2 s-1], to soil profile
                    capillary_rise [kg m-2 s-1], from soil profile
                    pond_recharge [kg m-2 s-1], flux to/from pond storage
                    water_closure [kg m-2 s-1], mass-balance error
                    co2_flux [umolm m-2 (ground) s-1], forest-floor NEE
                    photosynthesis [umolm m-2 (ground) s-1], bottomlayer types GPP
                    respiration [umolm m-2 (ground) s-1], bottomlayer types + soil
                    soil_respiration [umolm m-2 (ground) s-1], soil
                
                state (dict): forestfloor aggregated state
                    temperature [degC]
                    surface_temperature [degC]
                    water_storage [kg m-2]
                    snow_water_equivalent [kg m-2]

            blt_outputs (list): bottomlayer_type -specific fluxes and state: list of Dicts with keys
                net_radiation [W m-2]
                latent_heat [W m-2]
                sensible_heat [W m-2]
                ground_heat [W m-2] (negative towards soil)
                heat_advection [W m-2]
                water_closure [kg m-2 s-1]
                energy_closure [W m-2]
                evaporation [kg m-2 s-1]
                interception [kg m-2 s-1]
                pond_recharge [kg m-2 s-1]
                capillary_rise [kg m-2 s-1]
                throughfall [kg m-2 s-1]
                temperature [degC]
                volumetric_water [m3 m-3]
                water_potential [m]
                water_content [g g-1]
                water_storage [kg m-2]
                hydraulic_conductivity [m s-1]
                thermal_conductivity [W m-1 K-1]

        """
        
        # initialize fluxes and states
        fluxes = {
            'net_radiation': 0.0, # [W m-2]
            'sensible_heat': 0.0, # [W m-2]
            'latent_heat': 0.0, # [W m-2]
            'ground_heat': 0.0, # [W m-2]
            'energy_closure': 0.0, # [W m-2]

            'evaporation': 0.0,  # [kg m-2 s-1]
            'soil_evaporation': 0.0, # [kg m-2 s-1]
            'throughfall': 0.0,  # [kg m-2 s-1]
            'capillary_rise': 0.0,  # [kg m-2 s-1]
            'pond_recharge': 0.0, # [kg m-2 s-1]
            'water_closure': 0.0, # [kg m-2 s-1]

            'net_co2': 0.0, # [umol m-2(ground) s-1]
            'photosynthesis': 0.0,  # [umol m-2(ground) s-1]
            'respiration': 0.0,  # [umol m-2(ground) s-1]
            'soil_respiration': 0.0,  # [umol m-2(ground) s-1]
        }

        state = {
            'temperature': 0.0,  # [degC]
            'surface_temperature': 0.0,  # [degC]
            'water_storage': 0.0, # [kg m-2]
            'snow_water_equivalent': 0.0, # [kg m-2]
            # not needed as we take optical properties from previous dt
            #'albedo': None,
            #'emissivity': None
         }

        # --- Soil respiration
        fluxes['soil_respiration'] = self.soilrespiration.respiration(
                                        forcing['soil_temperature'],
                                        forcing['soil_volumetric_water'],
                                        forcing['soil_volumetric_air'])

        fluxes['respiration'] += fluxes['soil_respiration']
        fluxes['net_co2'] += fluxes['soil_respiration']

        # --- Snow: now using simple degree-day model
        snow_forcing = {
            'precipitation_rain': forcing['precipitation_rain'],
            'precipitation_snow': forcing['precipitation_snow'],
            'air_temperature': forcing['air_temperature'],
        }

        fluxes_snow, states_snow = self.snowpack.run(dt=dt, forcing=snow_forcing)

        # --- solve bottomlayer types and aggregate forest floor fluxes & state
        org_forcing = forcing.copy()
        del org_forcing['precipitation_rain'], org_forcing['precipitation_snow']

        org_forcing.update(
                {'precipitation': fluxes_snow['potential_infiltration'],
                'soil_temperature': forcing['soil_temperature'][0],
                'snow_water_equivalent': states_snow['snow_water_equivalent']}
                )

        # bottomlayer-type specific fluxes and state for output: list of dicts
        bt_results = []

        for bt in self.bottomlayer_types:
            bt_flx, bt_state = bt.run(dt, org_forcing, parameters, controls)

            # effective forest floor fluxes and state
            for key in fluxes.keys():
                if key in bt_flx.keys():
                    fluxes[key] += bt.coverage * bt_flx[key]

            state['temperature'] += bt.coverage * bt_state['temperature']
            state['surface_temperature'] += bt.coverage * bt_state['surface_temperature']
            state['water_storage'] += bt.coverage * bt_state['water_storage']

            # merge dicts and append to gt_results
            bt_flx.update(bt_state)
            bt_results.append(bt_flx)
            del bt_flx, bt_state

        fluxes['evaporation'] += fluxes['soil_evaporation']
        fluxes['latent_heat'] += LATENT_HEAT / MOLAR_MASS_H2O * fluxes['soil_evaporation']

#            # this creates problem if forcing['air_temperature'] <0 and self.snowpack.swe >0
#        if self.snowpack.swe > 0:
#            fluxes['throughfall'] += fluxes_snow['potential_infiltration']
#
#            # some groundheat flux to keep soil temperatures reasonable
#            # this creates problem if forcing['air_temperature'] <0 and self.snowpack.swe >0
#            fluxes['ground_heat'] += (
#                parameters['soil_thermal_conductivity']
#                / abs(parameters['soil_depth'])
#                * (min(forcing['air_temperature'],0.0) - forcing['soil_temperature'][0])
#            )
#
#            state['snow_water_equivalent'] = states_snow['snow_water_equivalent']
#            state['temperature'] = states_snow['temperature']
#            state['surface_temperature'] = states_snow['temperature']

        # bottomlayer_type specific results (fluxes & state): convert list of dicts to dict of lists
        blt_outputs = {}
        for k,v in bt_results[0].items():
            blt_outputs[k] = [x[k] for x in bt_results]

        return fluxes, state, blt_outputs

# EOF