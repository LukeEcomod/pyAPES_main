# -*- coding: utf-8 -*-
"""
.. module: photo
    :synopsis: pyAPES leaf component. Describes coupled leaf/needle-scale photosynthesis and stomatal control using variants of
Farquhar model and 'A-gs' schemes. Includes also leaf energy balance.
.. moduleauthor:: Samuli Launiainen & Kersti Leppä

Key references:
    Launiainen et al. 2015 Ecol. Mod.
    Farquhar et al. 1980 Planta
    Medlyn et al., 2002a,b
    Medlyn et al. 2011
    Katul et al. 2009, 2010
    Vico et al. 2013
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import List, Dict, Tuple

from pyAPES.microclimate.micromet import e_sat, latent_heat
from pyAPES.leaf.boundarylayer import leaf_boundary_layer_conductance
from pyAPES.utils.constants import DEG_TO_KELVIN, PAR_TO_UMOL, EPS, SPECIFIC_HEAT_AIR, \
    GAS_CONSTANT, O2_IN_AIR, MOLAR_MASS_H2O

H2O_CO2_RATIO = 1.6  # H2O to CO2 diffusivity ratio [-]
TN = 25.0 + DEG_TO_KELVIN  # reference temperature, 283.15 [K]

logger = logging.getLogger(__name__)

# %%
# --- coupled leaf energy balance and gas-exchange


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
            #kn: [-] nitrogen attenuation factor in canopy, Not used ??
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
                # MEDLYN (Medlyn et al., 2011 with co-limitation Farquhar)
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

    Note: NOT currently used in pyAPES-MLM! There coupled leaf energy balance is solved in planttype.PlantType.leaf_gas_exchange()

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
        # geff_v = (gb_v*gsv) / (gb_v + gsv)
        # molm-2s-1, condensation only on dry leaf part
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
            # s[esat / P < H2O] = EPS
            # Dleaf = np.maximum(EPS, esat / P - H2O)  # mol/mol
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


# %%
#  ---- photosynthesis - stomatal control (A-gs) models ----- """

def photo_c3_analytical(photop: Dict, Qp: np.ndarray, T: np.ndarray, VPD: np.ndarray,
                        ca: np.ndarray, gb_c: np.ndarray, gb_v: np.ndarray):
    """
    Leaf photosynthesis and gas-exchange by co-limitation optimality model of
    Vico et al. 2013 AFM
    Args:
        photop - parameter dict with keys: Vcmax, Jmax, Rd, alpha, theta, La, tresp
           can be scalars or arrays.
           tresp - dictionary with keys: Vcmax, Jmax, Rd: temperature sensitivity
           parameters. OMIT key 'tresp' if no temperature adjustments for photoparameters!
        Qp - incident PAR at leaves [umolm-2s-1]
        T - leaf temperature [degC]
        VPD - leaf-air vapor pressure difference [mol mol-1]
        ca - ambient CO2 [ppm]
        gb_c - boundary-layer conductance for CO2 [mol m-2 s-1]
        gb_v - boundary-layer conductance for H2O [mol m-2 s-1]

    Returns:
        An - net CO2 flux [umol m-2 s-1]
        Rd - dark respiration [umol m-2 s-1]
        fe - leaf transpiration rate [mol m-2 s-1]
        gs - stomatal conductance for CO2 [mol m-2 s-1]
        ci - leaf internal CO2 [ppm]
        cs - leaf surface CO2 [ppm]
    """

    Tk = T + DEG_TO_KELVIN

    MaxIter = 20

    # --- params ----
    Vcmax = photop['Vcmax']
    Jmax = photop['Jmax']
    Rd = photop['Rd']
    alpha = photop['alpha']
    theta = photop['theta']
    La = photop['La']
    g0 = photop['g0']

    # From Bernacchi et al. 2001

    # --- CO2 compensation point -------
    Tau_c = 42.75 * np.exp(37830*(Tk - TN) / (TN * GAS_CONSTANT * Tk))

    # ---- Kc & Ko (umol/mol), Rubisco activity for CO2 & O2 ------
    Kc = 404.9 * np.exp(79430.0*(Tk - TN) / (TN * GAS_CONSTANT * Tk))
    Ko = 2.784e5 * np.exp(36380.0*(Tk - TN) / (TN * GAS_CONSTANT * Tk))

    if 'tresp' in photop:  # adjust parameters for temperature
        tresp = photop['tresp']
        Vcmax_T = tresp['Vcmax']
        Jmax_T = tresp['Jmax']
        Rd_T = tresp['Rd']
        Vcmax, Jmax, Rd, Tau_c = photo_temperature_response(
            Vcmax, Jmax, Rd, Vcmax_T, Jmax_T, Rd_T, Tk)

    # --- model parameters k1_c, k2_c [umol/m2/s]
    Km = Kc*(1.0 + O2_IN_AIR / Ko)
    J = (Jmax + alpha*Qp - ((Jmax + alpha*Qp)**2.0 -
         (4*theta*Jmax*alpha*Qp))**(0.5)) / (2.0*theta)

    k1_c = J/4.0
    k2_c = (J/4.0) * Km / Vcmax

    # --- iterative solution for cs
    err = 9999.0
    cnt = 1
    cs = ca  # leaf surface CO2
    while err > 0.01 and cnt < MaxIter:
        NUM1 = -k1_c * (k2_c - (cs - 2*Tau_c))
        DEN1 = (k2_c + cs)**2
        NUM2 = (
            np.sqrt(H2O_CO2_RATIO * VPD * La * k1_c**2
                    * (cs - Tau_c) * (k2_c + Tau_c)
                    * ((k2_c + (cs - 2.0 * H2O_CO2_RATIO * VPD * La))**2)
                    * (k2_c + (cs - H2O_CO2_RATIO * VPD * La)))
        )

        DEN2 = H2O_CO2_RATIO*VPD*La * \
            ((k2_c + cs)**2) * (k2_c + (cs - H2O_CO2_RATIO*VPD*La))

        gs_opt = (NUM1 / DEN1) + (NUM2 / DEN2) + EPS

        ci = (
            (1. / (2 * gs_opt))
            * (-k1_c - k2_c*gs_opt
               + cs*gs_opt + Rd
               + np.sqrt((k1_c + k2_c*gs_opt - cs*gs_opt - Rd)**2
                         - 4*gs_opt*(-k1_c*Tau_c - k2_c*cs*gs_opt - k2_c*Rd)))
        )

        An = gs_opt*(cs - ci)
        An1 = np.maximum(An, 0.0)
        cs0 = cs
        cs = ca - An1 / gb_c

        err = np.nanmax(abs(cs - cs0))
        cnt = cnt + 1
        # print('err', err)

    ix = np.where(An < 0)
    gs_opt[ix] = g0

    if type(ca) is float:
        ci[ix] = ca
        cs[ix] = ca
    else:
        ci[ix] = ca[ix]
        cs[ix] = ca[ix]

    gs_v = H2O_CO2_RATIO*gs_opt

    geff = (gb_v*gs_v) / (gb_v + gs_v)  # molm-2s-1
    fe = geff*VPD  # leaf transpiration rate

    if len(An) == 1:
        return float(An), float(Rd), float(fe), float(gs_opt), float(ci), float(cs)
    else:
        return An, Rd, fe, gs_opt, ci, cs


def photo_c3_medlyn(photop: Dict, Qp: np.ndarray, T: np.ndarray, VPD: np.ndarray,
                    ca: np.ndarray, gb_c: np.ndarray, gb_v=np.ndarray, P: float = 101300.0) -> Tuple:
    """
    Leaf gas-exchange by Farquhar-Medlyn Unified Stomatal Optimality (USO) -model (Medlyn et al., 2011 GCB).
    Av, Aj co-limitation implemented here as in Vico et al. 2013 AFM

    Args:
        photop - parameter dict with keys: Vcmax, Jmax, Rd, alpha, theta, La, tresp
           can be scalars or arrays.
           tresp - dictionary with keys: Vcmax, Jmax, Rd: temperature sensitivity
           parameters. OMIT key 'tresp' if no temperature adjustments for photoparameters!
        Qp - incident PAR at leaves [umolm-2s-1]
        T - leaf temperature [degC]
        VPD - leaf-air vapor pressure difference [mol mol-1]
        ca - ambient CO2 [ppm]
        gb_c - boundary-layer conductance for CO2 [mol m-2 s-1]
        gb_v - boundary-layer conductance for H2O [mol m-2 s-1]

    Returns:
        An - net CO2 flux [umol m-2 s-1]
        Rd - dark respiration [umol m-2 s-1]
        fe - leaf transpiration rate [mol m-2 s-1]
        gs - stomatal conductance for CO2 [mol m-2 s-1]
        ci - leaf internal CO2 [ppm]
        cs - leaf surface CO2 [ppm]
    """
    Tk = T + DEG_TO_KELVIN
    VPD = 1e-3 * VPD * P  # kPa

    MaxIter = 50

    # --- params ----
    Vcmax = photop['Vcmax']
    Jmax = photop['Jmax']
    Rd = photop['Rd']
    alpha = photop['alpha']
    theta = photop['theta']
    g1 = photop['g1']  # slope parameter
    g0 = photop['g0']

    # --- CO2 compensation point -------
    Tau_c = 42.75 * np.exp(37830*(Tk - TN) / (TN * GAS_CONSTANT * Tk))

    # ---- Kc & Ko (umol/mol), Rubisco activity for CO2 & O2 ------
    Kc = 404.9 * np.exp(79430.0*(Tk - TN) / (TN * GAS_CONSTANT * Tk))
    Ko = 2.784e5 * np.exp(36380.0*(Tk - TN) / (TN * GAS_CONSTANT * Tk))

    if 'tresp' in photop:  # adjust parameters for temperature
        tresp = photop['tresp']
        Vcmax_T = tresp['Vcmax']
        Jmax_T = tresp['Jmax']
        Rd_T = tresp['Rd']
        Vcmax, Jmax, Rd, Tau_c = photo_temperature_response(
            Vcmax, Jmax, Rd, Vcmax_T, Jmax_T, Rd_T, Tk)

    # --- model parameters k1_c, k2_c [umol/m2/s]
    Km = Kc*(1.0 + O2_IN_AIR / Ko)
    J = (Jmax + alpha*Qp - ((Jmax + alpha*Qp)**2.0 -
         (4*theta*Jmax*alpha*Qp))**(0.5)) / (2*theta)
    k1_c = J / 4.0
    k2_c = J / 4.0 * Km / Vcmax

    # --- iterative solution for cs and ci
    err = 9999.0
    cnt = 1
    cs = ca  # leaf surface CO2
    ci = 0.8*ca  # internal CO2
    while err > 0.01 and cnt < MaxIter:
        # CO2 demand (Vico eq. 1) & gs_opt (Medlyn eq. xx)
        An = k1_c * (ci - Tau_c) / (k2_c + ci) - Rd  # umolm-2s-1
        An1 = np.maximum(An, 0.0)
        gs_opt = (1.0 + g1 / (VPD**0.5)) * An1 / (cs - Tau_c)  # mol m-2s-1
        gs_opt = np.maximum(g0, gs_opt)  # g0 is the lower limit

        # CO2 supply
        cs = np.maximum(ca - An1 / gb_c, 0.5*ca)  # through boundary layer
        ci0 = ci
        ci = np.maximum(cs - An1 / gs_opt, 0.5*ca)  # through stomata

        err = max(abs(ci0 - ci))
        cnt += 1
    # when Rd > photo, assume stomata closed and ci == ca
    ix = np.where(An < 0)
    if type(ca) is float:
        ci[ix] = ca
        cs[ix] = ca
    else:
        ci[ix] = ca[ix]
        cs[ix] = ca[ix]
    gs_opt[ix] = g0
    gs_v = H2O_CO2_RATIO*gs_opt

    geff = (gb_v*gs_v) / (gb_v + gs_v)  # molm-2s-1
    fe = geff * VPD / (1e-3 * P)  # leaf transpiration rate

    return An, Rd, fe, gs_opt, ci, cs


def photo_c3_medlyn_farquhar(photop: Dict, Qp: np.ndarray, T: np.ndarray, VPD: np.ndarray,
                             ca: np.ndarray, gb_c: np.ndarray, gb_v: np.ndarray, P: float = 101300.0) -> Tuple:
    """
    Leaf gas-exchange by Farquhar-Medlyn Unified Stomatal Optimality (USO) -model (Medlyn et al., 2011 GCB), 
    where co-limitation as in standard Farquhar-model

    Args:
        photop - parameter dict with keys: Vcmax, Jmax, Rd, alpha, theta, beta, g1, g0, tresp
           can be scalars or arrays.
           tresp - dictionary with keys: Vcmax, Jmax, Rd: temperature sensitivity
           parameters. OMIT key 'tresp' if no temperature adjustments for photoparameters!
        Qp - incident PAR at leaves [umolm-2s-1]
        T - leaf temperature [degC]
        VPD - leaf-air vapor pressure difference [mol mol-1]
        ca - ambient CO2 [ppm]
        gb_c - boundary-layer conductance for CO2 [mol m-2 s-1]
        gb_v - boundary-layer conductance for H2O [mol m-2 s-1]

    Returns:
        An - net CO2 flux [umol m-2 s-1]
        Rd - dark respiration [umol m-2 s-1]
        fe - leaf transpiration rate [mol m-2 s-1]
        gs - stomatal conductance for CO2 [mol m-2 s-1]
        ci - leaf internal CO2 [ppm]
        cs - leaf surface CO2 [ppm]
    """
    Tk = T + DEG_TO_KELVIN
    VPD = np.maximum(EPS, 1e-3 * VPD * P)  # kPa

    MaxIter = 50

    # --- params ----
    Vcmax = photop['Vcmax']  # [umol m-2 s-1]
    Jmax = photop['Jmax']
    Rd = photop['Rd']
    alpha = photop['alpha']  # [-]
    theta = photop['theta']
    g1 = photop['g1']  # [kPa^0.5], slope parameter
    g0 = photop['g0']  # [mol m-2 (leaf) s-1], for CO2
    beta = photop['beta']

    # --- CO2 compensation point -------
    Tau_c = 42.75 * np.exp(37830*(Tk - TN) / (TN * GAS_CONSTANT * Tk))

    # ---- Kc & Ko (umol/mol), Rubisco activity for CO2 & O2 ------
    Kc = 404.9 * np.exp(79430.0*(Tk - TN) / (TN * GAS_CONSTANT * Tk))
    Ko = 2.784e5 * np.exp(36380.0*(Tk - TN) / (TN * GAS_CONSTANT * Tk))

    if 'tresp' in photop:  # adjust parameters for temperature
        tresp = photop['tresp']
        Vcmax_T = tresp['Vcmax']
        Jmax_T = tresp['Jmax']
        Rd_T = tresp['Rd']
        Vcmax, Jmax, Rd, Tau_c = photo_temperature_response(
            Vcmax, Jmax, Rd, Vcmax_T, Jmax_T, Rd_T, Tk)

    # --- model parameters k1_c, k2_c [umol/m2/s]
    Km = Kc*(1.0 + O2_IN_AIR / Ko)
    J = (Jmax + alpha*Qp - ((Jmax + alpha*Qp)**2.0 -
         (4*theta*Jmax*alpha*Qp))**(0.5)) / (2*theta)
    # k1_c = J / 4.0
    # k2_c = J / 4.0 * Km / Vcmax

    # --- iterative solution for cs and ci
    err = 9999.0
    cnt = 1
    cs = ca  # leaf surface CO2
    ci = 0.8*ca  # internal CO2
    while err > 0.01 and cnt < MaxIter:
        # -- rubisco -limited rate
        Av = Vcmax * (ci - Tau_c) / (ci + Km)
        # -- RuBP -regeneration limited rate
        Aj = J/4.0 * (ci - Tau_c) / (ci + 2.0*Tau_c)

        # An = np.minimum(Av, Aj) - Rd  # single limiting rate
        x = Av + Aj
        y = Av * Aj
        An = (x - (x**2.0 - 4.0*beta*y)**0.5) / \
            (2.0*beta) - Rd  # co-limitation

        An1 = np.maximum(An, 0.0)
        # print An1
        # stomatal conductance
        gs_opt = g0 + (1.0 + g1 / (VPD**0.5)) * An1 / cs
        gs_opt = np.maximum(g0, gs_opt)  # gcut is the lower limit
        # print gs_opt
        # CO2 supply
        cs = np.maximum(ca - An1 / gb_c, 0.5*ca)  # through boundary layer
        ci0 = ci
        ci = np.maximum(cs - An1 / gs_opt, 0.1*ca)  # through stomata

        err = max(abs(ci0 - ci))
        cnt += 1

    # when Rd > photo, assume stomata closed and ci == ca
    ix = np.where(An < 0)
    if type(ca) is float:
        ci[ix] = ca
        cs[ix] = ca
    else:
        ci[ix] = ca[ix]
        cs[ix] = ca[ix]
    gs_opt[ix] = g0
    gs_v = H2O_CO2_RATIO*gs_opt

    geff = (gb_v*gs_v) / (gb_v + gs_v)  # molm-2s-1
    fe = geff * VPD / (1e-3 * P)  # leaf transpiration rate

    return An, Rd, fe, gs_opt, ci, cs


def photo_c3_medlyn_farquhar_gm(photop: Dict, Qp: np.ndarray, Qa: np.ndarray, T: np.ndarray, VPD: np.ndarray,
                                ca: np.ndarray, gb_c: np.ndarray, gb_v: np.ndarray, P: float = 101300.0) -> Tuple:
    """
        Leaf gas-exchange by Farquhar-Medlyn Unified Stomatal Optimality (USO) -model (Medlyn et al., 2011 GCB), 
        where co-limitation as in standard Farquhar-model. 

        Args:
            photop - parameter dict with keys: Vcmax, Jmax, Rd, alpha, theta, beta, g1, g0, tresp, gm, !!! parstress missing !!!
            can be scalars or arrays.
            tresp - dictionary with keys: Vcmax, Jmax, Rd: temperature sensitivity
            parameters. OMIT key 'tresp' if no temperature adjustments for photoparameters!
            Qp - incident PAR at leaves [umolm-2s-1]
            T - leaf temperature [degC]
            VPD - leaf-air vapor pressure difference [mol mol-1]
            ca - ambient CO2 [ppm]
            gb_c - boundary-layer conductance for CO2 [mol m-2 s-1]
            gb_v - boundary-layer conductance for H2O [mol m-2 s-1]

        Returns:
            An - net CO2 flux [umol m-2 s-1]
            Rd - dark respiration [umol m-2 s-1]
            fe - leaf transpiration rate [mol m-2 s-1]
            gs - stomatal conductance for CO2 [mol m-2 s-1]
            ci - leaf internal CO2 [ppm]
            cs - leaf surface CO2 [ppm]
            gm - mesophyll conductance for CO2 [mol m-2 s-1] !!!!! CHANGE OUTPUT ? !!!!
            cc - chloroplast CO2 [ppm]
        """

    Tk = T + DEG_TO_KELVIN
    VPD = np.maximum(EPS, 1e-3 * VPD * P)  # kPa

    MaxIter = 50

    # --- params ----
    Vcmax = photop['Vcmax']  # [umol m-2 s-1]
    Jmax = photop['Jmax']
    Rd = photop['Rd']
    alpha = photop['alpha']  # [-]
    theta = photop['theta']
    g1 = photop['g1']  # [kPa^0.5], slope parameter
    g0 = photop['g0']  # [mol m-2 (leaf) s-1], for CO2
    gm = photop['gm']['gm25']  # [mol m-2 (leaf) s-1] for CO2
    gm_fmin = photop['gm']['fmin']
    beta = photop['beta']
    # --- CO2 compensation point -------
    Tau_c = 42.75 * np.exp(37830*(Tk - TN) / (TN * GAS_CONSTANT * Tk))
    # ---- Kc & Ko (umol/mol), Rubisco activity for CO2 & O2 ------
    Kc = 404.9 * np.exp(79430.0*(Tk - TN) / (TN * GAS_CONSTANT * Tk))
    Ko = 2.784e5 * np.exp(36380.0*(Tk - TN) / (TN * GAS_CONSTANT * Tk))

    if 'tresp' in photop:  # adjust parameters for temperature
        tresp = photop['tresp']
        Vcmax_T = tresp['Vcmax']
        Jmax_T = tresp['Jmax']
        Rd_T = tresp['Rd']
        if photop['gm']['tempstress']:
            gm_T = tresp['gm']
            Vcmax, Jmax, Rd, Tau_c, gm = photo_temperature_response_gm(
                Vcmax, Jmax, Rd, gm, Vcmax_T, Jmax_T, Rd_T, gm_T, Tk)
        else:
            Vcmax, Jmax, Rd, Tau_c = photo_temperature_response(
                Vcmax, Jmax, Rd, Vcmax_T, Jmax_T, Rd_T, Tk)

    if photop['gm']['parstress']:
        gm = gm*(1-(1-gm_fmin)*np.exp(-0.003*Qa))  # Knauer et al., 2019, GCB

    gm = np.maximum(gm_fmin*gm, gm)
    # --- model parameters k1_c, k2_c [umol/m2/s]
    Km = Kc*(1.0 + O2_IN_AIR / Ko)
    J = (Jmax + alpha*Qp - ((Jmax + alpha*Qp)**2.0 -
         (4*theta*Jmax*alpha*Qp))**(0.5)) / (2*theta)
    # k1_c = J / 4.0
    # k2_c = J / 4.0 * Km / Vcmax

    # --- iterative solution for An, gs, cs, ci and cc
    err = 9999.0
    cnt = 1
    cs = ca  # leaf surface CO2
    ci = 0.8*ca  # internal CO2
    cc = 0.7*ca  # chlorophyll CO2
    An = np.zeros_like(ca)  # initial guess for photosynthesis rate
    gs_opt = np.zeros_like(ca)  # initial guess for gs
    gm = np.zeros_like(ca) + gm 
    while err > 0.01 and cnt < MaxIter:
        # This loop solves An, gs, cs, ci and cc
        # save old values before new round
        prev_values = np.concatenate((An, gs_opt, cs, ci, cc))
        # -- rubisco -limited rate
        Av = Vcmax * (cc - Tau_c) / (cc + Km)
        # -- RuBP -regeneration limited rate
        Aj = J/4.0 * (cc - Tau_c) / (cc + 2.0*Tau_c)
        
        # An = np.minimum(Av, Aj) - Rd  # single limiting rate
        x = Av + Aj
        y = Av * Aj
        An = (x - (x**2.0 - 4.0*beta*y)**0.5) / \
            (2.0*beta) - Rd  # co-limitation

        An1 = np.maximum(An, 0.0)
        # stomatal conductance
        gs_opt = g0 + (1.0 + g1 / (VPD**0.5)) * An1 / cs
        gs_opt = np.maximum(g0, gs_opt)  # gcut is the lower limit

        # CO2 supply
        cs = np.maximum(ca - An1 / gb_c, 0.5*ca)  # through boundary layer

        ci = np.maximum(cs - An1 / gs_opt, 0.1*ca)  # through stomata
        # cc0=cc
        cc = np.maximum(ci - An1 / gm, 0.05*ca)  # through mesophyll

        # Testaa konvergoituuko cs, ci kun cc konvergoituu
        new_values = np.concatenate((An, gs_opt, cs, ci, cc))

        err = np.linalg.norm(prev_values-new_values)/np.linalg.norm(new_values)
        # err = max(abs(cc0 - cc))

        cnt += 1

    # Log if no convergence

    if cnt == MaxIter:
        logger.debug(' Maximum number of iterations reached: cc = %.2f (err = %.2f)',
                np.mean(cc), err)
    
    # when Rd > photo, assume stomata closed and cc == ci == ca
    ix = np.where(An < 0)
    if type(ca) is float:
        ci[ix] = ca
        cs[ix] = ca
        cc[ix] = ca
    else:
        ci[ix] = ca[ix]
        cs[ix] = ca[ix]
        cc[ix] = ca[ix]
    gs_opt[ix] = g0
    gs_v = H2O_CO2_RATIO*gs_opt

    geff = (gb_v*gs_v) / (gb_v + gs_v)  # molm-2s-1
    fe = geff * VPD / (1e-3 * P)  # leaf transpiration rate

    return An, Rd, fe, gs_opt, ci, cs, gm, cc # change output order?


def photo_c3_bwb(photop: Dict, Qp: np.ndarray, T: np.ndarray, RH: np.ndarray,
                 ca: np.ndarray, gb_c: np.ndarray, gb_v: np.ndarray, P: float = 101300.0) -> Tuple:
    """
    Leaf gas-exchange by Farquhar-Ball-Woodrow-Berry model, co-limitation as in standard Farquhar-
    model

    Args:
        photop - parameter dict with keys: Vcmax, Jmax, Rd, alpha, theta, beta, g1, g0, tresp
           can be scalars or arrays.
           tresp - dictionary with keys: Vcmax, Jmax, Rd: temperature sensitivity
           parameters. OMIT key 'tresp' if no temperature adjustments for photoparameters!
        Qp - incident PAR at leaves [umolm-2s-1]
        T - leaf temperature [degC]
        VPD - leaf-air vapor pressure difference [mol mol-1]
        ca - ambient CO2 [ppm]
        gb_c - boundary-layer conductance for CO2 [mol m-2 s-1]
        gb_v - boundary-layer conductance for H2O [mol m-2 s-1]

    Returns:
        An - net CO2 flux [umol m-2 s-1]
        Rd - dark respiration [umol m-2 s-1]
        fe - leaf transpiration rate [mol m-2 s-1]
        gs - stomatal conductance for CO2 [mol m-2 s-1]
        ci - leaf internal CO2 [ppm]
        cs - leaf surface CO2 [ppm]
    """
    Tk = T + DEG_TO_KELVIN

    MaxIter = 50

    # --- params ----
    Vcmax = photop['Vcmax']
    Jmax = photop['Jmax']
    Rd = photop['Rd']
    alpha = photop['alpha']
    theta = photop['theta']
    g1 = photop['g1']  # slope parameter
    g0 = photop['g0']
    beta = photop['beta']

    # --- CO2 compensation point -------
    Tau_c = 42.75 * np.exp(37830*(Tk - TN) / (TN * GAS_CONSTANT * Tk))

    # ---- Kc & Ko (umol/mol), Rubisco activity for CO2 & O2 ------
    Kc = 404.9 * np.exp(79430.0*(Tk - TN) / (TN * GAS_CONSTANT * Tk))
    Ko = 2.784e5 * np.exp(36380.0*(Tk - TN) / (TN * GAS_CONSTANT * Tk))

    if 'tresp' in photop:  # adjust parameters for temperature
        tresp = photop['tresp']
        Vcmax_T = tresp['Vcmax']
        Jmax_T = tresp['Jmax']
        Rd_T = tresp['Rd']
        Vcmax, Jmax, Rd, Tau_c = photo_temperature_response(
            Vcmax, Jmax, Rd, Vcmax_T, Jmax_T, Rd_T, Tk)

    # --- model parameters k1_c, k2_c [umol/m2/s]
    Km = Kc*(1.0 + O2_IN_AIR / Ko)
    J = (Jmax + alpha*Qp - ((Jmax + alpha*Qp)**2.0 -
         (4*theta*Jmax*alpha*Qp))**(0.5)) / (2*theta)

    # --- iterative solution for cs and ci
    err = 9999.0
    cnt = 1
    cs = ca  # leaf surface CO2
    ci = 0.8*ca  # internal CO2
    while err > 0.01 and cnt < MaxIter:
        # -- rubisco -limited rate
        Av = Vcmax * (ci - Tau_c) / (ci + Km)
        # -- RuBP -regeneration limited rate
        Aj = J/4.0 * (ci - Tau_c) / (ci + 2.0*Tau_c)

        # An = np.minimum(Av, Aj) - Rd  # single limiting rate
        # co-limitation
        x = Av + Aj
        y = Av * Aj
        An = (x - (x**2.0 - 4.0*beta*y)**0.5) / (2.0*beta) - Rd

        An1 = np.maximum(An, 0.0)
        # bwb -scheme
        gs_opt = g0 + g1 * An1 / ((cs - Tau_c))*RH
        gs_opt = np.maximum(g0, gs_opt)  # g0 is the lower limit

        # CO2 supply
        cs = np.maximum(ca - An1 / gb_c, 0.5*ca)  # through boundary layer
        ci0 = ci
        ci = np.maximum(cs - An1 / gs_opt, 0.1*ca)  # through stomata

        err = max(abs(ci0 - ci))
        cnt += 1

    # when Rd > photo, assume stomata closed and ci == ca
    ix = np.where(An < 0)
    gs_opt[ix] = g0
    ci[ix] = ca[ix]
    cs[ix] = ca[ix]
    gs_v = H2O_CO2_RATIO*gs_opt

    geff = (gb_v*gs_v) / (gb_v + gs_v)  # molm-2s-1
    esat, _ = e_sat(T)
    VPD = (1.0 - RH) * esat / P  # mol mol-1
    fe = geff*VPD  # leaf transpiration rate

    return An, Rd, fe, gs_opt, ci, cs


def photo_farquhar(photop: Dict, Qp: np.ndarray, ci: np.ndarray, T: np.ndarray,
                   co_limi: bool = False) -> Tuple:
    """
    Farquhar model for leaf-level photosynthesis, dark respiration and net CO2 exchange.

    Args:
        photop - dict with keys:
            Vcmax
            Jmax
            Rd
            qeff
            alpha
            theta
            beta
        Qp - incident Par (umolm-2s-1)
        ci - leaf internal CO2 mixing ratio (ppm)
        T - leaf temperature (degC)
        co_limi (bool)- True uses co-limitation function of Vico et al., 2014.
    Returns:
        An - leaf net CO2 exchange [umol m-2 leaf s-1]
        Rd - leaf dark respiration rate [umol m-2 leaf s-1]
        Av - rubisco limited rate (if co_limi==True)
        Aj - electron transport limited rate (if co_limi==True)
        Tau_c - CO2 compensation point [ppm]
        Kc - Rubisco activity for CO2 [umol mol-1]
        Ko - Rubisco activity for O [umol mol-1] 
        Km - Kc*(1.0 + O2_IN_AIR / Ko) [umol mol-1]
        J - electron transport rate [umol m-2 leaf s-1] CHECK UNITS!!

    NOTE: original and co_limi -versions converge when beta ~ 0.8
    """
    Tk = T + DEG_TO_KELVIN  # K

    # --- params ----
    Vcmax = photop['Vcmax']
    Jmax = photop['Jmax']
    Rd = photop['Rd']
    alpha = photop['alpha']
    theta = photop['theta']
    beta = photop['beta']  # co-limitation parameter

    # --- CO2 compensation point -------
    Tau_c = 42.75 * np.exp(37830*(Tk - TN) / (TN * GAS_CONSTANT * Tk))

    # ---- Kc & Ko (umol/mol), Rubisco activity for CO2 & O2 ------
    Kc = 404.9 * np.exp(79430.0*(Tk - TN) / (TN * GAS_CONSTANT * Tk))
    Ko = 2.784e5 * np.exp(36380.0*(Tk - TN) / (TN * GAS_CONSTANT * Tk))

    if 'tresp' in photop:  # adjust parameters for temperature
        tresp = photop['tresp']
        Vcmax_T = tresp['Vcmax']
        Jmax_T = tresp['Jmax']
        Rd_T = tresp['Rd']
        Vcmax, Jmax, Rd, Tau_c = photo_temperature_response(
            Vcmax, Jmax, Rd, Vcmax_T, Jmax_T, Rd_T, Tk)

    Km = Kc*(1.0 + O2_IN_AIR / Ko)
    J = (Jmax + alpha*Qp - ((Jmax + alpha*Qp)**2.0 -
         (4.0*theta*Jmax*alpha*Qp))**0.5) / (2.0*theta)

    if not co_limi:
        # -- rubisco -limited rate
        Av = Vcmax * (ci - Tau_c) / (ci + Km)
        # -- RuBP -regeneration limited rate
        Aj = J/4 * (ci - Tau_c) / (ci + 2.0*Tau_c)

        # An = np.minimum(Av, Aj) - Rd  # single limiting rate
        x = Av + Aj
        y = Av * Aj
        An = (x - (x**2 - 4*beta*y)**0.5) / (2*beta) - Rd  # co-limitation
        return An, Rd, Av, Aj, Tau_c, Kc, Ko, Km, J
    else:   # use Vico et al. eq. 1
        k1_c = J / 4.0
        k2_c = (J / 4.0) * Km / Vcmax

        An = k1_c * (ci - Tau_c) / (k2_c + ci) - Rd
        return An, Rd, Tau_c, Kc, Ko, Km, J


def photo_temperature_response(Vcmax0: np.ndarray, Jmax0: np.ndarray, Rd0: np.ndarray,
                               Vcmax_T: list, Jmax_T: list, Rd_T: list, T: np.ndarray):
    """
    Adjusts Farquhar-parameters for temperature

    Args:
        - Vcmax25, maximum carboxylation velocity at 25 degC
        - Jmax25, maximum electron transport rate at 25 degC
        - Rd25, dark respiration rate at 25 degC
        - Vcmax_T (list) [activation energy [kJ mol-1], 
                              deactivation energy [kJ mol-1], 
                              entropy factor [kJ mol-1]
                             ]
        - Jmax_T (list): [activation energy [kJ mol-1], 
                              deactivation energy [kJ mol-1], 
                              entropy factor [kJ mol-1]
                             ]
       - Rd_T (list): [activation energy [kJ mol-1]
       - T [K] temperature


    Returns:
        - Vcmax at temperature T
        - Jmax at temperature T
        - Rd at temperature T
        - Gamma_star [ppm], CO2 compensation point at T

    Reference:
        Medlyn et al., 2002.Plant Cell Environ. 25, 1167-1179; based on Bernacchi
        et al. 2001. Plant Cell Environ., 24, 253-260.
    """

    # --- CO2 compensation point -------
    Gamma_star = 42.75 * np.exp(37830*(T - TN) / (TN * GAS_CONSTANT * T))

    # ------  Vcmax (umol m-2(leaf)s-1) ------------
    Ha = 1e3 * Vcmax_T[0]  # J mol-1, activation energy Vcmax
    Hd = 1e3 * Vcmax_T[1]  # J mol-1, deactivation energy Vcmax
    Sd = Vcmax_T[2]  # entropy factor J mol-1 K-1

    NOM = np.exp(Ha * (T - TN) / (GAS_CONSTANT*TN*T)) * \
        (1.0 + np.exp((TN*Sd - Hd) / (TN*GAS_CONSTANT)))
    DENOM = (1.0 + np.exp((T*Sd - Hd) / (T*GAS_CONSTANT)))
    Vcmax = Vcmax0 * NOM / DENOM

    del Ha, Hd, Sd, DENOM, NOM

    # ----  Jmax (umol m-2(leaf)s-1) ------------
    Ha = 1e3 * Jmax_T[0]  # J mol-1, activation energy Vcmax
    Hd = 1e3 * Jmax_T[1]  # J mol-1, deactivation energy Vcmax
    Sd = Jmax_T[2]  # entropy factor J mol-1 K-1

    NOM = np.exp(Ha * (T - TN) / (GAS_CONSTANT*TN*T)) * \
        (1.0 + np.exp((TN*Sd - Hd) / (TN*GAS_CONSTANT)))
    DENOM = (1.0 + np.exp((T*Sd - Hd) / (T*GAS_CONSTANT)))
    Jmax = Jmax0*NOM / DENOM

    del Ha, Hd, Sd, DENOM, NOM

    # --- Rd (umol m-2(leaf)s-1) -------
    Ha = 1e3 * Rd_T[0]  # J mol-1, activation energy dark respiration
    Rd = Rd0 * np.exp(Ha*(T - TN) / (TN * GAS_CONSTANT * T))

    return Vcmax, Jmax, Rd, Gamma_star


def photo_temperature_response_gm(Vcmax0: np.ndarray, Jmax0: np.ndarray, Rd0: np.ndarray, gm0: np.ndarray,
                                  Vcmax_T: list, Jmax_T: list, Rd_T: list, gm_T: list, T: np.ndarray,):
    """
    Adjusts Farquhar-parameters for temperature

    Args:
        - Vcmax25, maximum carboxylation velocity at 25 degC
        - Jmax25, maximum electron transport rate at 25 degC
        - Rd25, dark respiration rate at 25 degC
        - Vcmax_T (list) [activation energy [kJ mol-1], 
                              deactivation energy [kJ mol-1], 
                              entropy factor [kJ mol-1]
                             ]
        - Jmax_T (list): [activation energy [kJ mol-1], 
                              deactivation energy [kJ mol-1], 
                              entropy factor [kJ mol-1]
                             ]
       - Rd_T (list): [activation energy [kJ mol-1]
       - T [K] temperature
       - gm0 (np.ndarray, defaults to None): mesophyll conductance at 25 degC
       - gm_T (list, defaults to None): [activation energy [kJ mol-1], 
                                        deactivation energy [kJ mol-1], 
                                        entropy factor [kJ mol-1]
                                        ]


    Returns:
        - Vcmax at temperature T
        - Jmax at temperature T
        - Rd at temperature T
        - Gamma_star [ppm], CO2 compensation point at T

    Reference:
        Medlyn et al., 2002.Plant Cell Environ. 25, 1167-1179; based on Bernacchi
        et al. 2001. Plant Cell Environ., 24, 253-260.
    """

    # --- CO2 compensation point -------
    Gamma_star = 42.75 * np.exp(37830*(T - TN) / (TN * GAS_CONSTANT * T))

    # ------  Vcmax (umol m-2(leaf)s-1) ------------
    Ha = 1e3 * Vcmax_T[0]  # J mol-1, activation energy Vcmax
    Hd = 1e3 * Vcmax_T[1]  # J mol-1, deactivation energy Vcmax
    Sd = Vcmax_T[2]  # entropy factor J mol-1 K-1

    NOM = np.exp(Ha * (T - TN) / (GAS_CONSTANT*TN*T)) * \
        (1.0 + np.exp((TN*Sd - Hd) / (TN*GAS_CONSTANT)))
    DENOM = (1.0 + np.exp((T*Sd - Hd) / (T*GAS_CONSTANT)))
    Vcmax = Vcmax0 * NOM / DENOM

    del Ha, Hd, Sd, DENOM, NOM

    # ----  Jmax (umol m-2(leaf)s-1) ------------
    Ha = 1e3 * Jmax_T[0]  # J mol-1, activation energy Vcmax
    Hd = 1e3 * Jmax_T[1]  # J mol-1, deactivation energy Vcmax
    Sd = Jmax_T[2]  # entropy factor J mol-1 K-1

    NOM = np.exp(Ha * (T - TN) / (GAS_CONSTANT*TN*T)) * \
        (1.0 + np.exp((TN*Sd - Hd) / (TN*GAS_CONSTANT)))
    DENOM = (1.0 + np.exp((T*Sd - Hd) / (T*GAS_CONSTANT)))
    Jmax = Jmax0*NOM / DENOM

    del Ha, Hd, Sd, DENOM, NOM

    # --- Rd (umol m-2(leaf)s-1) -------
    Ha = 1e3 * Rd_T[0]  # J mol-1, activation energy dark respiration
    Rd = Rd0 * np.exp(Ha*(T - TN) / (TN * GAS_CONSTANT * T))

    del Ha

    # --- gm (mol m-2(leaf)s-1) ------
    Ha = 1e3 * gm_T[0]  # J mol-1, activation energy gm
    Hd = 1e3 * gm_T[1]  # J mol-1, deactivation energy gm
    Sd = gm_T[2]  # entropy factor J mol-1 K-1

    NOM = np.exp(Ha * (T - TN) / (GAS_CONSTANT*TN*T)) * \
        (1.0 + np.exp((TN*Sd - Hd) / (TN*GAS_CONSTANT)))
    DENOM = (1.0 + np.exp((T*Sd - Hd) / (T*GAS_CONSTANT)))
    gm = gm0*NOM / DENOM

    return Vcmax, Jmax, Rd, Gamma_star, gm


def apparent_photocapacity(b: List, psi_leaf: np.ndarray) -> float:
    """
    Relative photosynthetic capacity as a function of leaf water potential
    Function shape from Kellomäki & Wang, adjustments for Vcmax and Jmax
    Args:
       beta (list|array) - parameters
       psi (float|array) - leaf water potential (MPa)
    Returns:
       f - relative value [0.2 - 1.0]
    """
    psi_leaf = np.array(np.size(psi_leaf), ndmin=1)
    f = (1.0 + np.exp(b[0] * b[1])) / (1.0 + np.exp(b[0] * (b[1] - psi_leaf)))
    f[f < 0.2] = 0.2

    return f


def topt_deltaS_conversion(Ha: float, Hd: float, dS: float = None, Topt: float = None) -> float:
    """
    Converts between entropy factor Sd [kJ mol-1] and temperature optimum
    Topt [k]. Medlyn et al. 2002 PCE 25, 1167-1179 eq.19.

    Args:
        - 'Ha' (float): activation energy [kJ mol-1]
        - 'Hd' (float): deactivation energy [kJ mol-1]
        - 'dS' (float): entropy factor [kJ mol-1]
        - 'Topt' (float): temperature optimum [K]
    Returns:
        - 'Topt' or 'dS' (float)

    """
    R = 8.314427  # gas constant, J mol-1 K-1

    if dS:  # Sv --> Topt
        xout = Hd / (dS - R * np.log(Ha / (Hd - Ha)))
    elif Topt:  # Topt -->Sv
        c = R * np.log(Ha / (Hd - Ha))
        xout = (Hd + Topt * c) / Topt

    return xout


def photo_Toptima(T10: float) -> Tuple:
    """
    computes acclimation of temperature optima of Vcmax and Jmax to 10-day mean air temperature
    Args:
        T10 - 10-day mean temperature [degC]
    Returns:
        Tv - temperature optima of Jmax [degC]
        Tj - temperature optima of Jmax [degC]
        rjv - ratio of Jmax25 / Vcmax25 [-]

    Reference:
        Lombardozzi et al., 2015 GRL, eq. 3 & 4
    """
    # --- parameters
    Hav = 72000.0  # J mol-1
    Haj = 50000.0  # J mol-1
    Hd = 200000.0  # J mol.1

    T10 = np.minimum(40.0, np.maximum(10.0, T10))  # range 10...40 degC
    # vcmax T-optima
    dSv = 668.39 - 1.07*T10  # J mol-1
    Tv = Hd / (dSv - GAS_CONSTANT * np.log(Hav / (Hd - Hav))) - \
        DEG_TO_KELVIN  # degC
    # jmax T-optima
    dSj = 659.70 - 0.75*T10  # J mol-1
    Tj = Hd / (dSj - GAS_CONSTANT * np.log(Haj / (Hd - Haj))) - \
        DEG_TO_KELVIN  # degC

    rjv = 2.59 - 0.035*T10  # Jmax25 / Vcmax25

    return Tv, Tj, rjv


# %%
# --- scripts for testing functions ---- """

def test_leafscale(method='MEDLYN_FARQUHAR', species='pine', Ebal=False):
    gamma = 1.0
    gfact = 1.0
    if species.upper() == 'PINE':
        photop = {
            #                'Vcmax': 55.0,
            #                'Jmax': 105.0,
            #                'Rd': 1.3,
            #                'tresp': {
            #                    'Vcmax': [78.0, 200.0, 650.0],
            #                    'Jmax': [56.0, 200.0, 647.0],
            #                    'Rd': [33.0]
            #                    },
            'Vcmax': 94.0,  # Tarvainen et al. 2018 Physiol. Plant.
            'Jmax': 143.0,
            'Rd': 1.3,
            'tresp': {
                'Vcmax': [78.3, 200.0, 650.1],
                'Jmax': [56.0, 200.0, 647.9],
                'Rd': [33.0]
            },
            'alpha': gamma * 0.2,
            'theta': 0.7,
            'La': 1600.0,
            'g1': gfact * 2.3,
            'g0': 1.0e-3,
            'kn': 0.6,
            'beta': 0.95,
            'drp': 0.7,

        }
        leafp = {
            'lt': 0.02,
            'par_alb': 0.12,
            'nir_alb': 0.55,
            'emi': 0.98
        }
    if species.upper() == 'SPRUCE':
        photop = {
            #                'Vcmax': 60.0,
            #                'Jmax': 114.0,
            #                'Rd': 1.5,
            #                'tresp': {
            #                    'Vcmax': [53.2, 202.0, 640.3],  # Tarvainen et al. 2013 Oecologia
            #                    'Jmax': [38.4, 202.0, 655.8],
            #                    'Rd': [33.0]
            #                    },
            'Vcmax': 69.7,  # Tarvainen et al. 2013 Oecologia
            'Jmax': 130.2,
            'Rd': 1.3,
            'tresp': {
                'Vcmax': [53.2, 200.0, 640.0],
                'Jmax': [38.4, 200.0, 655.5],
                'Rd': [33.0]
            },
            'alpha': gamma * 0.2,
            'theta': 0.7,
            'La': 1600.0,
            'g1': gfact * 2.3,
            'g0': 1.0e-3,
            'kn': 0.6,
            'beta': 0.95,
            'drp': 0.7,

        }
        leafp = {
            'lt': 0.02,
            'par_alb': 0.12,
            'nir_alb': 0.55,
            'emi': 0.98
        }
    if species.upper() == 'DECID':
        photop = {
            #                'Vcmax': 50.0,
            #                'Jmax': 95.0,
            #                'Rd': 1.3,
            #                'tresp': {
            #                    'Vcmax': [77.0, 200.0, 636.7],  # Medlyn et al 2002.
            #                    'Jmax': [42.8, 200.0, 637.0],
            #                    'Rd': [33.0]
            #                    },
            'Vcmax': 69.1,  # Medlyn et al 2002.
            'Jmax': 116.3,
            'Rd': 1.3,
            'tresp': {
                'Vcmax': [77.0, 200.0, 636.4],
                'Jmax': [42.8, 200.0, 636.6],
                'Rd': [33.0]
            },
            'alpha': gamma * 0.2,
            'theta': 0.7,
            'La': 600.0,
            'g1': gfact * 4.5,
            'g0': 1.0e-3,
            'kn': 0.6,
            'beta': 0.95,
            'drp': 0.7,

        }
        leafp = {
            'lt': 0.05,
            'par_alb': 0.12,
            'nir_alb': 0.55,
            'emi': 0.98
        }
    if species.upper() == 'SHRUBS':
        photop = {
            'Vcmax': 50.0,
            'Jmax': 95.0,
            'Rd': 1.3,
            'alpha': gamma * 0.2,
            'theta': 0.7,
            'La': 600.0,
            'g1': gfact * 4.5,
            'g0': 1.0e-3,
            'kn': 0.3,
            'beta': 0.95,
            'drp': 0.7,
            'tresp': {
                'Vcmax': [77.0, 200.0, 636.7],
                'Jmax': [42.8, 200.0, 637.0],
                'Rd': [33.0]
            }
        }
        leafp = {
            'lt': 0.02,
            'par_alb': 0.12,
            'nir_alb': 0.55,
            'emi': 0.98
        }
    # env. conditions
    N = 50
    P = 101300.0
    Qp = 1000. * np.ones(N)  # np.linspace(1.,1800.,50)#
    CO2 = 400. * np.ones(N)
    U = 1.0  # np.array([10.0, 1.0, 0.1, 0.01])
    T = np.linspace(1., 39., 50)  # 10. * np.ones(N) #
    esat, s = e_sat(T)
    H2O = (85.0 / 100.0) * esat / P
    SWabs = 0.5 * (1-leafp['par_alb']) * Qp / PAR_TO_UMOL + \
        0.5 * (1-leafp['nir_alb']) * Qp / PAR_TO_UMOL
    LWnet = -30.0 * np.ones(N)

    forcing = {
        'h2o': H2O,
        'co2': CO2,
        'air_temperature': T,
        'par_incident': Qp,
        'sw_absorbed': SWabs,
        'lw_net': LWnet,
        'wind_speed': U,
        'air_pressure': P
    }

    controls = {
        'photo_model': method,
        'energy_balance': Ebal
    }

    x = leaf_Ags_ebal(photop, leafp, forcing, controls)
#    print(x)
    Y = T
    plt.figure(5)
    plt.subplot(421)
    plt.plot(Y, x['net_co2'], 'o')
    plt.title('net_co2')
    plt.subplot(422)
    plt.plot(Y, x['transpiration'], 'o')
    plt.title('transpiration')
    plt.subplot(423)
    plt.plot(Y, x['net_co2'] + x['dark_respiration'], 'o')
    plt.title('co2 uptake')
    plt.subplot(424)
    plt.plot(Y, x['dark_respiration'], 'o')
    plt.title('dark_respiration')
    plt.subplot(425)
    plt.plot(Y, x['stomatal_conductance'], 'o')
    plt.title('stomatal_conductance')
    plt.subplot(426)
    plt.plot(Y, x['boundary_conductance'], 'o')
    plt.title('boundary_conductance')
    plt.subplot(427)
    plt.plot(Y, x['leaf_internal_co2'], 'o')
    plt.title('leaf_internal_co2')
    plt.subplot(428)
    plt.plot(Y, x['leaf_surface_co2'], 'o')
    plt.title('leaf_surface_co2')
    plt.tight_layout()


def test_photo_temperature_response(species='pine'):
    T = np.linspace(1., 40., 79)
    Tk = T + DEG_TO_KELVIN
    if species.upper() == 'PINE':
        photop = {
            'Vcmax': 55.0,
            'Jmax': 105.0,
            'Rd': 1.3,
            'tresp': {
                'Vcmax': [78.3, 200.0, 650.1],
                'Jmax': [56.0, 200.0, 647.9],
                'Rd': [33.0]
            }
        }
    if species.upper() == 'SPRUCE':
        photop = {
            'Vcmax': 60.0,
            'Jmax': 114.0,
            'Rd': 1.5,
            'tresp': {
                # Tarvainen et al. 2013 Oecologia
                'Vcmax': [53.2, 202.0, 640.3],
                'Jmax': [38.4, 202.0, 655.8],
                'Rd': [33.0]
            }
        }
    if species.upper() == 'DECID':
        photop = {
            'Vcmax': 50.0,
            'Jmax': 95.0,
            'Rd': 1.3,
            'tresp': {
                'Vcmax': [77.0, 200.0, 636.7],  # Medlyn et al 2002.
                'Jmax': [42.8, 200.0, 637.0],
                'Rd': [33.0]
            }
        }
    if species.upper() == 'SHRUBS':
        photop = {
            'Vcmax': 50.0,
            'Jmax': 95.0,
            'Rd': 1.3,
            'tresp': {
                'Vcmax': [77.0, 200.0, 636.7],
                'Jmax': [42.8, 200.0, 637.0],
                'Rd': [33.0]
            }
        }

    if species.upper() == 'PINE2':
        photop = {
            'Vcmax': 55.0,
            'Jmax': 105.0,
            'Rd': 1.3,
            'tresp': {
                'Vcmax': [72., 200., 649.],  # (Kattge and Knorr, 2007)
                'Jmax': [50., 200., 646.],  # (Kattge and Knorr, 2007)
                'Rd': [33.0]
            }
        }
    if species.upper() == 'SPRUCE2':
        photop = {
            'Vcmax': 60.0,
            'Jmax': 114.0,
            'Rd': 1.5,
            'tresp': {
                'Vcmax': [72., 200., 649.],  # (Kattge and Knorr, 2007)
                'Jmax': [50., 200., 646.],  # (Kattge and Knorr, 2007)
                'Rd': [33.0]
            }
        }
    if species.upper() == 'DECID2':
        photop = {
            'Vcmax': 50.0,
            'Jmax': 95.0,
            'Rd': 1.3,
            'tresp': {
                'Vcmax': [72., 200., 649.],  # (Kattge and Knorr, 2007)
                'Jmax': [50., 200., 646.],  # (Kattge and Knorr, 2007)
                'Rd': [33.0]
            }
        }

    Vcmax = photop['Vcmax']
    Jmax = photop['Jmax']
    Rd = photop['Rd']
    tresp = photop['tresp']
    Vcmax_T = tresp['Vcmax']
    Jmax_T = tresp['Jmax']
    Rd_T = tresp['Rd']
    Vcmax, Jmax, Rd, Tau_c = photo_temperature_response(
        Vcmax, Jmax, Rd, Vcmax_T, Jmax_T, Rd_T, Tk)

    plt.figure(4)
    plt.subplot(311)
    plt.plot(T, Vcmax, 'o')
    plt.title('Vcmax')
    plt.subplot(312)
    plt.plot(T, Jmax, 'o')
    plt.title('Jmax')
    plt.subplot(313)
    plt.plot(T, Rd, 'o')
    plt.title('Rd')


def Topt_to_Sd(Ha, Hd, Topt):
    Sd = Hd * 1e3 / (Topt + DEG_TO_KELVIN) + \
        GAS_CONSTANT * np.log(Ha / (Hd - Ha))
    return Sd


def Sd_to_Topt(Ha, Hd, Sd):
    Topt = Hd*1e3 / (Sd + GAS_CONSTANT * np.log(Ha / (Hd - Ha)))
    return Topt - DEG_TO_KELVIN

# EOF
