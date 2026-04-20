# -*- coding: utf-8 -*-
"""
.. module: photo_ebal
    :synopsis: Coupled leaf energy balance and gas-exchange.
.. moduleauthor:: Samuli Launiainen & Kersti Leppä

Note: This module is not currently used in the pyAPES model. Coupled leaf energy
balance is solved in planttype.PlantType.leaf_gas_exchange().
"""

import numpy as np
import logging
from typing import Dict

from pyAPES.microclimate.micromet import e_sat, latent_heat
from pyAPES.leaf.boundarylayer import leaf_boundary_layer_conductance
from pyAPES.utils.constants import SPECIFIC_HEAT_AIR, MOLAR_MASS_H2O, H2O_CO2_RATIO, EPS
from pyAPES.leaf.photo import photo_c3_medlyn_farquhar, photo_c3_bwb, photo_c3_analytical

logger = logging.getLogger(__name__)


def leaf_Ags_ebal(photop: Dict, leafp: Dict, forcing: Dict, controls: Dict,
                  df: float = 1.0, dict_output: bool = True, logger_info: str = ''):
    r"""
    Computes leaf net CO2 flux (An), respiration (Rd), transpiration (E) and estimates
    leaf temperature (Tl) and sensible heat fluxes (H) based on leaf energy balance equation coupled with
    leaf-level photosynthesis and stomatal control schemes.

    Energy balance is solved using Taylor's expansion (i.e isothermal net radiation -approximation) which
    eliminates need for iterations with the long-wave radiation-sceme.

    Depending on choise of 'photo_model', photosynthesis is calculated based on biochemical model of Farquhar et
    al. (1980), coupled with alternative stomatal control schemes (Medlyn, Ball-Woodrow-Berry, Hari, Katul-Vico et al.)
    In all these models, stomatal conductance (gs) is proportional to instantaneous An, either by optimal stomatal control principles or
    using semi-empirical models.

    Args:
        photop (dict): leaf gas-exchange parameters
            Vcmax: [umol m-2 s-1] maximum carboxylation velocity at 25C
            Jmax: [umol m-2 s-1] maximum rate of electron transport at 25C
            Rd: [umol m-2 s-1] dark respiration rate at 25C
            alpha: apparent quantum yield parameter
            theta: [-] co-limitation parameter of Farquhar-model
            La: stomatal parameter (Lambda, m [-], ...) depending on model
            g1: [kPa^0.5] stomatal slope parameter of USO -model
            g0: [mol m-2 s-1] residual conductance for CO2
            beta: [-] co-limitation parameter of Farquhar-model
            drp: drought response parameters of Medlyn stomatal model and apparent Vcmax
                 list: [Rew_crit_g1, slope_g1, Rew_crit_appVcmax, slope_appVcmax]
            tresp' (dict): parameters of photosynthetic temperature response curve
                - Vcmax (list): [activation energy [kJ mol-1],
                                 deactivation energy [kJ mol-1]
                                 entropy factor [kJ mol-1]
                                ]
                - Jmax (list): [activation energy [kJ mol-1],
                                 deactivation energy [kJ mol-1]
                                 entropy factor [kJ mol-1]
                                ]
                - Rd (list): [activation energy [kJ mol-1]

        leafp (dict): leaf properties
            lt: leaf lengthscale [m]

        forcing (dict):
            h2o: water vapor mixing ratio [mol mol-1]
            co2: carbon dioxide mixing ratio [ppm]]
            air_temperature: ambient air temperature [degC]
            par_incident: incident PAR at leaves [umol m-2 s-1]
            sw_absorbed: absorbed SW (PAR + NIR) at leaves [W m-2]
            lw_net: net isothermal long-wave radiation [W m-2]
            wind_speed: mean wind speed [m s-1]
            air_pressure: ambient pressure [Pa]
            leaf_temperature: initial guess for leaf temperature (optional) [degC]
            average_leaf_temperature: leaf temperature used for computing LWnet (optional) [degC]
            radiative_conductance: radiative conductance used in computing LWnet (optional) [degC]

        controls (dict):
            photo_model (str): photosysthesis model
                CO_OPTI (Vico et al., 2014)
                MEDLYN_FARQUHAR Medlyn et al., 2011 with co-limitation Farquhar)
                BWB (Ball et al., 1987 with co-limitation Farquhar)
            energy_balance (bool): True -> computes leaf temperature by solving energy balance
        dict_output (bool): True -> returns output as dict, False as separate arrays (optional)
        logger_info (str): optional

    OUTPUT:
        (dict):
            net_co2: net CO2 flux (umol m-2 leaf s-1)
            dark_respiration: CO2 respiration (umol m-2 leaf s-1)
            transpiration: H2O flux (transpiration) (mol m-2 leaf s-1)
            sensible_heat: sensible heat flux (W m-2 leaf)
            fr: non-isothermal radiative flux (W m-2)
            Tl: leaf temperature (degC)
            stomatal_conductance: stomatal conductance for H2O (mol m-2 leaf s-1)
            boundary_conductance: boundary layer conductance for H2O (mol m-2 leaf s-1)
            leaf_internal_co2: leaf internal CO2 mixing ratio (mol/mol)
            leaf_surface_co2: leaf surface CO2 mixing ratio (mol/mol)

    Note: Vectorized code can be used in multi-layer sense where inputs are vectors of equal length
    """

    # -- parameters -----
    lt = leafp['lt']

    T = np.array(forcing['air_temperature'], ndmin=1)
    H2O = np.array(forcing['h2o'], ndmin=1)
    Qp = forcing['par_incident']
    P = forcing['air_pressure']
    U = forcing['wind_speed']
    CO2 = forcing['co2']

    Ebal = controls['energy_balance']
    model = controls['photo_model']

    if Ebal:
        SWabs = np.array(forcing['sw_absorbed'], ndmin=1)
        LWnet = np.array(forcing['lw_net'], ndmin=1)
        Rabs = SWabs + LWnet
        # canopy nodes
        ic = np.where(abs(LWnet) > 0.0)

    if 'leaf_temperature' in forcing:
        Tl_ini = np.array(forcing['leaf_temperature'], ndmin=1).copy()

    else:
        Tl_ini = T.copy()

    Tl = Tl_ini.copy()
    Told = Tl.copy()

    if 'radiative_conductance' in forcing:
        gr = df * np.array(forcing['radiative_conductance'], ndmin=1)
    else:
        gr = np.zeros(len(T))

    if 'average_leaf_temperature' in forcing:
        Tl_ave = np.array(forcing['average_leaf_temperature'], ndmin=1)

    else:
        Tl_ave = Tl.copy()

    # vapor pressure
    esat, s = e_sat(Tl)
    s = s / P  # slope of esat, mol/mol / degC
    Dleaf = esat / P - H2O

    Lv = latent_heat(T) * MOLAR_MASS_H2O

    itermax = 20
    err = 999.0
    iter_no = 0

    while err > 0.01 and iter_no < itermax:
        iter_no += 1
        # boundary layer conductance
        gb_h, gb_c, gb_v = leaf_boundary_layer_conductance(
            U, lt, T, 0.5 * (Tl + Told) - T, P)

        Told = Tl.copy()

        if model.upper() == 'MEDLYN_FARQUHAR':
            An, Rd, fe, gs_opt, Ci, Cs = photo_c3_medlyn_farquhar(
                photop, Qp, Tl, Dleaf, CO2, gb_c, gb_v, P=P)

        if model.upper() == 'BWB':
            rh = (1 - Dleaf*P / esat)  # rh at leaf (-)
            An, Rd, fe, gs_opt, Ci, Cs = photo_c3_bwb(
                photop, Qp, Tl, rh, CO2, gb_c, gb_v, P=P)

        # --- analytical co-limitation model Vico et al. 2013
        if model.upper() == 'CO_OPTI':
            An, Rd, fe, gs_opt, Ci, Cs = photo_c3_analytical(
                photop, Qp, Tl, Dleaf, CO2, gb_c, gb_v)

        gsv = H2O_CO2_RATIO*gs_opt
        # condensation only on dry leaf part
        geff_v = np.where(Dleaf > 0.0, (gb_v*gsv) / (gb_v + gsv), df * gb_v)
        gb_h = df * gb_h  # sensible heat exchange only through dry leaf part

        # solve leaf temperature from energy balance
        if Ebal:
            Tl[ic] = (Rabs[ic] + SPECIFIC_HEAT_AIR*gr[ic]*Tl_ave[ic] + SPECIFIC_HEAT_AIR*gb_h[ic]*T[ic] - Lv[ic]*geff_v[ic]*Dleaf[ic]
                      + Lv[ic]*s[ic]*geff_v[ic]*Told[ic]) / (SPECIFIC_HEAT_AIR*(gr[ic] + gb_h[ic]) + Lv[ic]*s[ic]*geff_v[ic])
            err = np.nanmax(abs(Tl - Told))

            if (err < 0.01 or iter_no == itermax) and abs(np.mean(T) - np.mean(Tl)) > 20.0:
                logger.debug(logger_info + ' Unrealistic leaf temperature %.2f set to air temperature %.2f, %.2f, %.2f, %.2f, %.2f',
                             np.mean(Tl), np.mean(T),
                             np.mean(LWnet), np.mean(Tl_ave), np.mean(Tl_ini), np.mean(H2O))
                Tl = T.copy()
                Ebal = False  # recompute without solving leaf temperature
                err = 999.

            elif iter_no == itermax and err > 0.05:
                logger.debug(logger_info + ' Maximum number of iterations reached: Tl = %.2f (err = %.2f)',
                             np.mean(Tl), err)

            # vapor pressure
            esat, s = e_sat(Tl)
            s = s / P  # slope of esat, mol/mol / degC
            Dleaf = esat / P - H2O
        else:
            err = 0.0

    # outputs
    H = SPECIFIC_HEAT_AIR*gb_h*(Tl - T)  # Wm-2
    # flux due to radiative conductance (Wm-2)
    Fr = SPECIFIC_HEAT_AIR*gr*(Tl - Tl_ave)
    # condensation accounted for in wetleaf water balance
    E = geff_v * np.maximum(0.0, Dleaf)
    LE = E * Lv  # condensation accounted for in wetleaf energy balance

    if dict_output:  # return dict

        x = {'net_co2': An,
             'dark_respiration': Rd,
             'transpiration': E,
             'sensible_heat': H,
             'latent_heat': LE,
             'fr': Fr,
             'leaf_temperature': Tl,
             # gsv gets high when VPD->0
             'stomatal_conductance': np.minimum(gsv, 1.0),
             'boundary_conductance': gb_v,
             'leaf_internal_co2': Ci,
             'leaf_surface_co2': Cs}
        return x
    else:  # return 11 arrays
        return An, Rd, E, H, Fr, Tl, Ci, Cs, gsv, gs_opt, gb_v
