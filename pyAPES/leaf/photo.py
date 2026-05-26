# -*- coding: utf-8 -*-
"""
.. module: photo
    :synopsis: pyAPES leaf component. Describes leaf/needle-scale photosynthesis and stomatal
    control using variants of the Farquhar model and 'A-gs' schemes.
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
import logging
from typing import List, Dict, Tuple

from pyAPES.microclimate.micromet import e_sat
from pyAPES.utils.constants import DEG_TO_KELVIN, EPS, GAS_CONSTANT, O2_IN_AIR, \
    H2O_CO2_RATIO, TN_GAS_CONSTANT, TN


logger = logging.getLogger(__name__)



# %%
# --- Photosynthesis model class and forcing utilities


class PhotosynthesisModel():
    """
    Wrapper class for running leaf-level photosynthesis models.

    Note: Currently only the MEDLYN_FARQUHAR model (photo_c3_medlyn_farquhar) is
    used in the pyAPES model.
    """

    def __init__(self, photo_model: str):
        self.photo_model = photo_model

        if self.photo_model.upper() == 'MEDLYN_FARQUHAR':
            self.output_names = ['An', 'Rd', 'fe', 'gs_opt', 'Ci', 'Cs']

    def run(self, forcing: Dict, photop: Dict) -> Dict:
        if self.photo_model.upper() == 'MEDLYN_FARQUHAR':
            results = photo_c3_medlyn_farquhar_fast(photop,
                                               forcing['Qp'], forcing['T'], forcing['VPD'],
                                               forcing['co2'], forcing['gb_c'], forcing['gb_v'],
                                               P=forcing['air_pressure'])
            results_dict = {k: v for k, v in zip(self.output_names, results)}
        return results_dict


def initialize_photo_forcing(num_elements: int) -> Dict:
    forcing = {}
    forcing['Qp'] = np.zeros((num_elements,)) * np.nan
    forcing['T'] = np.zeros((num_elements,)) * np.nan
    forcing['VPD'] = np.zeros((num_elements,)) * np.nan
    forcing['co2'] = np.zeros((num_elements,)) * np.nan
    forcing['gb_c'] = np.zeros((num_elements,)) * np.nan
    forcing['gb_v'] = np.zeros((num_elements,)) * np.nan

    return forcing


def set_photo_forcing(forcing: Dict, Qp, T, VPD, co2, gb_c, gb_v, P) -> Dict:
    keys = ['Qp', 'T', 'VPD', 'co2', 'gb_c', 'gb_v', 'air_pressure']
    values = (Qp, T, VPD, co2, gb_c, gb_v, P)

    for key, val in zip(keys, values):
        forcing[key] = val

    return forcing


# %%
# --- Photosynthesis - stomatal control (A-gs) models


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

    TN_GAS_CONSTANT_Tk = TN * GAS_CONSTANT * Tk
    Tk_minus_TN = Tk - TN
    # --- CO2 compensation point -------
    Tau_c = 42.75 * np.exp(37830*(Tk_minus_TN) / (TN_GAS_CONSTANT_Tk))

    # ---- Kc & Ko (umol/mol), Rubisco activity for CO2 & O2 ------
    Kc = 404.9 * np.exp(79430.0*(Tk_minus_TN) / (TN_GAS_CONSTANT_Tk))
    Ko = 2.784e5 * np.exp(36380.0*(Tk_minus_TN) / (TN_GAS_CONSTANT_Tk))

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

        x = Av + Aj
        y = Av * Aj
        An = (x - (x**2.0 - 4.0*beta*y)**0.5) / \
            (2.0*beta) - Rd  # co-limitation

        An1 = np.maximum(An, 0.0)
        # stomatal conductance
        gs_opt = g0 + (1.0 + g1 / (VPD**0.5)) * An1 / cs
        gs_opt[gs_opt < g0] = g0
        # CO2 supply
        cs = np.maximum(ca - An1 / gb_c, 0.5*ca)  # through boundary layer
        ci0 = ci
        ci = np.maximum(cs - An1 / gs_opt, 0.1*ca)  # through stomata

        err = max((ci0 - ci)*(ci0-ci))
        cnt += 1

    # when Rd > photo, assume stomata closed and ci == ca
    ix = np.where(An < 0)
    ci[ix] = ca[ix]
    cs[ix] = ca[ix]
    gs_opt[ix] = g0
    gs_v = H2O_CO2_RATIO*gs_opt

    geff = (gb_v*gs_v) / (gb_v + gs_v)  # molm-2s-1
    fe = geff * VPD / (1e-3 * P)  # leaf transpiration rate

    return An, Rd, fe, gs_opt, ci, cs


def photo_c3_medlyn_farquhar_fast(photop: Dict, Qp: np.ndarray, T: np.ndarray, VPD: np.ndarray,
                                  ca: np.ndarray, gb_c: np.ndarray, gb_v: np.ndarray, P: float = 101300.0) -> Tuple:
    """
    Optimized version of photo_c3_medlyn_farquhar for benchmarking.

    Changes from the original:
    - Precompute loop-invariant expressions outside the iteration loop:
        g1_factor, J4, two_Tau_c, four_beta, inv_two_beta, ca_half, ca_tenth
    - Replace x**2.0 with x*x and (...)**0.5 with np.sqrt() to avoid power dispatch
    - Precompute P_milli = 1e-3 * P (used twice)
    - Precompute JaQ = Jmax + alpha*Qp (appears twice in J formula)
    - Replace Python built-in max() with np.max() (avoids Python-level iteration over array)
    - Replace gs_opt[gs_opt < g0] = g0 with np.maximum(g0, gs_opt, out=gs_opt) (in-place, no bool mask alloc)
    """
    Tk = T + DEG_TO_KELVIN
    P_milli = 1e-3 * P  # kPa
    VPD = np.maximum(EPS, VPD * P_milli)  # kPa

    # --- params ----
    Vcmax = photop['Vcmax']  # [umol m-2 s-1]
    Jmax = photop['Jmax']
    Rd = photop['Rd']
    alpha = photop['alpha']  # [-]
    theta = photop['theta']
    g1 = photop['g1']  # [kPa^0.5], slope parameter
    g0 = photop['g0']  # [mol m-2 (leaf) s-1], for CO2
    beta = photop['beta']

    TN_GAS_CONSTANT_Tk = TN * GAS_CONSTANT * Tk
    Tk_minus_TN = Tk - TN
    # --- CO2 compensation point -------
    Tau_c = 42.75 * np.exp(37830*(Tk_minus_TN) / (TN_GAS_CONSTANT_Tk))

    # ---- Kc & Ko (umol/mol), Rubisco activity for CO2 & O2 ------
    Kc = 404.9 * np.exp(79430.0*(Tk_minus_TN) / (TN_GAS_CONSTANT_Tk))
    Ko = 2.784e5 * np.exp(36380.0*(Tk_minus_TN) / (TN_GAS_CONSTANT_Tk))

    if 'tresp' in photop:  # adjust parameters for temperature
        tresp = photop['tresp']
        Vcmax, Jmax, Rd, Tau_c = photo_temperature_response_fast(
            Vcmax, Jmax, Rd, tresp['Vcmax'], tresp['Jmax'], tresp['Rd'], Tk)

    # --- model parameters
    Km = Kc * (1.0 + O2_IN_AIR / Ko)
    JaQ = Jmax + alpha * Qp  # precompute: appears twice in J formula
    J = (JaQ - np.sqrt(JaQ * JaQ - 4 * theta * Jmax * alpha * Qp)) / (2 * theta)

    # precompute loop-invariant expressions
    J4 = J * 0.25
    two_Tau_c = 2.0 * Tau_c
    four_beta = 4.0 * beta
    inv_two_beta = 0.5 / beta
    g1_factor = 1.0 + g1 / np.sqrt(VPD)  # VPD is constant through the loop
    ca_half = 0.5 * ca
    ca_tenth = 0.1 * ca

    # --- iterative solution for cs and ci
    err = 9999.0
    cnt = 1
    cs = ca  # leaf surface CO2
    ci = 0.8 * ca  # internal CO2
    while err > 0.01 and cnt < 50:
        # -- rubisco-limited rate
        Av = Vcmax * (ci - Tau_c) / (ci + Km)
        # -- RuBP-regeneration limited rate
        Aj = J4 * (ci - Tau_c) / (ci + two_Tau_c)

        x = Av + Aj
        y = Av * Aj
        An = (x - np.sqrt(x * x - four_beta * y)) * inv_two_beta - Rd  # co-limitation

        An1 = np.maximum(An, 0.0)
        # stomatal conductance
        gs_opt = g0 + g1_factor * An1 / cs
        np.maximum(g0, gs_opt, out=gs_opt)
        # CO2 supply
        cs = np.maximum(ca - An1 / gb_c, ca_half)   # through boundary layer
        ci0 = ci
        ci = np.maximum(cs - An1 / gs_opt, ca_tenth)  # through stomata

        err = np.max((ci0 - ci) * (ci0 - ci))
        cnt += 1

    # when Rd > photo, assume stomata closed and ci == ca
    ix = np.where(An < 0)
    ci[ix] = ca[ix]
    cs[ix] = ca[ix]
    gs_opt[ix] = g0
    gs_v = H2O_CO2_RATIO * gs_opt

    geff = (gb_v * gs_v) / (gb_v + gs_v)  # molm-2s-1
    fe = geff * VPD / P_milli  # leaf transpiration rate

    return An, Rd, fe, gs_opt, ci, cs


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

    TN_GAS_CONSTANT_T = TN_GAS_CONSTANT * T
    T_GAS_CONSTANT = T*GAS_CONSTANT
    T_minus_TN = T - TN

    # --- CO2 compensation point -------
    Gamma_star = 42.75 * np.exp(37830*(T_minus_TN) / (TN_GAS_CONSTANT_T))

    # ------  Vcmax (umol m-2(leaf)s-1) ------------
    Ha = 1e3*Vcmax_T[0]  # J mol-1, activation energy Vcmax
    Hd = 1e3*Vcmax_T[1]  # J mol-1, deactivation energy Vcmax
    Sd = Vcmax_T[2]  # entropy factor J mol-1 K-1

    NOM = np.exp(Ha * (T_minus_TN) / (TN_GAS_CONSTANT_T)) * \
        (1.0 + np.exp((TN*Sd - Hd) / (TN_GAS_CONSTANT)))
    DENOM = (1.0 + np.exp((T*Sd - Hd) / (T_GAS_CONSTANT)))

    Vcmax = Vcmax0 * NOM / DENOM

    # ------  Jmax (umol m-2(leaf)s-1) ----------
    Ha = 1e3*Jmax_T[0]  # J mol-1, activation energy Jmax
    Hd = 1e3*Jmax_T[1]  # J mol-1, deactivation energy Jmax
    Sd = Jmax_T[2]  # entropy factor J mol-1 K-1

    NOM = np.exp(Ha * (T_minus_TN) / (TN_GAS_CONSTANT_T)) * \
        (1.0 + np.exp((TN*Sd - Hd) / (TN_GAS_CONSTANT)))
    DENOM = (1.0 + np.exp((T*Sd - Hd) / (T_GAS_CONSTANT)))

    Jmax = Jmax0 * NOM / DENOM

    # ------ Rd (umol m-2(leaf)s-1) --------
    Ha = 1e3*Rd_T[0]  # J mol-1, activation energy dark respiration
    Rd = Rd0 * np.exp(Ha*(T_minus_TN) / (TN_GAS_CONSTANT_T))

    return Vcmax, Jmax, Rd, Gamma_star


def photo_temperature_response_fast(Vcmax0: np.ndarray, Jmax0: np.ndarray, Rd0: np.ndarray,
                                    Vcmax_T: list, Jmax_T: list, Rd_T: list, T: np.ndarray):
    """
    Optimized version of photo_temperature_response for benchmarking.

    Changes from the original:
    - Precompute T_ratio = T_minus_TN / TN_GAS_CONSTANT_T once and reuse in all 4 exp()
      numerators (Gamma_star, Vcmax NOM, Jmax NOM, Rd) — saves 3 array divisions
    - Precompute inv_T_GAS_CONSTANT = 1 / (T * GAS_CONSTANT) once and reuse in both
      Vcmax and Jmax DENOM expressions — replaces 1 array division with 1 array multiply
    - Rename Ha/Hd/Sd with suffixes to avoid variable rebinding between blocks
    """
    T_minus_TN = T - TN
    TN_GAS_CONSTANT_T = TN_GAS_CONSTANT * T

    # precompute: used in all 4 exp() numerators — saves 3 array divisions
    T_ratio = T_minus_TN / TN_GAS_CONSTANT_T
    # precompute: used in both Vcmax and Jmax DENOM — saves 1 array division
    inv_T_GAS_CONSTANT = 1.0 / (T * GAS_CONSTANT)

    # --- CO2 compensation point -------
    Gamma_star = 42.75 * np.exp(37830 * T_ratio)

    # ------  Vcmax (umol m-2(leaf)s-1) ------------
    Ha_v = 1e3 * Vcmax_T[0]  # J mol-1, activation energy Vcmax
    Hd_v = 1e3 * Vcmax_T[1]  # J mol-1, deactivation energy Vcmax
    Sd_v = Vcmax_T[2]         # entropy factor J mol-1 K-1

    NOM = np.exp(Ha_v * T_ratio) * (1.0 + np.exp((TN*Sd_v - Hd_v) / TN_GAS_CONSTANT))
    DENOM = 1.0 + np.exp((T*Sd_v - Hd_v) * inv_T_GAS_CONSTANT)
    Vcmax = Vcmax0 * NOM / DENOM

    # ------  Jmax (umol m-2(leaf)s-1) ----------
    Ha_j = 1e3 * Jmax_T[0]  # J mol-1, activation energy Jmax
    Hd_j = 1e3 * Jmax_T[1]  # J mol-1, deactivation energy Jmax
    Sd_j = Jmax_T[2]         # entropy factor J mol-1 K-1

    NOM = np.exp(Ha_j * T_ratio) * (1.0 + np.exp((TN*Sd_j - Hd_j) / TN_GAS_CONSTANT))
    DENOM = 1.0 + np.exp((T*Sd_j - Hd_j) * inv_T_GAS_CONSTANT)
    Jmax = Jmax0 * NOM / DENOM

    # ------ Rd (umol m-2(leaf)s-1) --------
    Ha_rd = 1e3 * Rd_T[0]  # J mol-1, activation energy dark respiration
    Rd = Rd0 * np.exp(Ha_rd * T_ratio)

    return Vcmax, Jmax, Rd, Gamma_star


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

        x = Av + Aj
        y = Av * Aj
        An = (x - (x**2 - 4*beta*y)**0.5) / (2*beta) - Rd  # co-limitation
        return An, Rd, Av, Aj, Tau_c, Kc, Ko, Km, J
    else:   # use Vico et al. eq. 1
        k1_c = J / 4.0
        k2_c = (J / 4.0) * Km / Vcmax

        An = k1_c * (ci - Tau_c) / (k2_c + ci) - Rd
        return An, Rd, Tau_c, Kc, Ko, Km, J


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


def Topt_to_Sd(Ha, Hd, Topt):
    Sd = Hd * 1e3 / (Topt + DEG_TO_KELVIN) + \
        GAS_CONSTANT * np.log(Ha / (Hd - Ha))
    return Sd


def Sd_to_Topt(Ha, Hd, Sd):
    Topt = Hd*1e3 / (Sd + GAS_CONSTANT * np.log(Ha / (Hd - Ha)))
    return Topt - DEG_TO_KELVIN

def Vcmax_from_Nleaf(N, pft):
    """
    Linear scaling between Vcmax and leaf nitrogen.
    Reference:  Kattge et al. 2009. Quantifying photosynthetic capacity and its relationship 
    to leaf nitrogen content for global‐scale terrestrial biosphere models. 
    Global Change Biology, 15(4), pp.976-991.; Their Table 2.

    Args:
        N (float or array) - leaf N content (g m-2)
        pft (str) - pft name
    Returns:
        Vcmax25 (float or array) - maximum carboxylation velocity (umol m-2(leaf) s-1)
    """

    pft_params = {
   
        "Tropical (oxisols)": {"iV": 1.99, "SD_iV": 5.14, "sV": 10.71, "SD_sV": 1.90, "corr": -0.93},
        "Tropical (nonoxisols)": {"iV": 6.35, "SD_iV": 5.52, "sV": 25.88, "SD_sV": 5.16, "corr": -0.93},
        "Temperate broadleaf": {"iV": 5.40, "SD_iV": 1.74, "sV": 30.38, "SD_sV": 1.34, "corr": -0.90},
        "Coniferous": {"iV": 34.05, "SD_iV": 5.90, "sV": 9.71, "SD_sV": 2.26, "corr": -0.91},
        "Shrubs": {"iV": 4.61, "SD_iV": 8.22, "sV": 30.20, "SD_sV": 4.77, "corr": -0.93},
        "C3 herbaceous": {"iV": 23.74, "SD_iV": 6.87, "sV": 28.17, "SD_sV": 4.67, "corr": -0.96},
        "C3 crops": {"iV": 22.22, "SD_iV": 16.74, "sV": 41.27, "SD_sV": 12.53, "corr": -0.96},
    }

    a0 = pft_params[pft]['iV']
    a1 = pft_params[pft]['sV']
    
    vmax25 = a0 + a1 * N
    return vmax25

        # if 'kn' in self.photop0:
        #     kn = self.photop0['kn']
        #     Lc = np.flipud(np.cumsum(np.flipud(self.lad*self.dz)))
        #     Lc = Lc / np.maximum(Lc[0], EPS)
        #     f = np.exp(-kn*Lc)
def canopygradient(kn, Lc):
    Lc = Lc / np.maximum(Lc[-1], 0.0001)
    print(Lc)
    f = np.exp(-kn*Lc)
    return f
# EOF
