# -*- coding: utf-8 -*-
"""
.. module: soil.heat
    :synopsis: pyAPES soil component
.. moduleauthor:: Kersti Leppä, Samuli Launiainen

Heat balance and phase-changes in 1D soil profile.

References:

"""

import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, Tuple, List

from pyAPES.soil.water import wrc
from pyAPES.utils.utilities import tridiag as thomas, spatial_average

from pyAPES.utils.constants import K_WATER, K_ICE, K_AIR, K_ORG, K_SAND, K_SILT, K_CLAY,\
    CV_AIR, CV_WATER, CV_ICE, CV_ORGANIC, CV_MINERAL, LATENT_HEAT_FREEZING, FREEZING_POINT_H2O, ICE_DENSITY

logger = logging.getLogger(__name__)

class Heat(object):

    def __init__(self, 
                 grid: Dict, 
                 profile_propeties: Dict, 
                 model_specs: Dict, 
                 volumetric_water_content: np.ndarray) -> object:
        r"""
        1D soil heat balance model

        Args:
            grid (dict):
                z: node elevations, soil surface = 0.0 [m]
                dz: thickness of computational layers [m]
                dzu: distance to upper node [m]
                dzl: distance to lower node [m]
            profile_propeties (dict):
                porosity (array): soil porosity (=ThetaS) [m3 m-3]
                solid_heat_capacity (array): [J m-3 (solid volume) K-1]
                    ! if None/NaN, estimated from organic/mineral composition
                solid_composition (dict):
                    organic (array): fraction [-] of solid volume
                    sand (array)
                    silt (array)
                    clay (array)
                freezing_curve (array): freezing curve parameter [-]
                bedrock_thermal_conductivity (array): must be NaN above bedrock depth [W m-1 K-1]
            model_specs (dict):
                solve (bool): solve heat balance
                initial_condition (dict):
                    temperature (array|float): initial temperature [degC]
                lower_boundary (dict):
                    type (str): 'flux' or 'temperature'
                    value (float): value of flux [W m-2] or temperature [degC]
        Returns:
            self (object)

        """
        # lower boundary condition
        self.lbc = model_specs['lower_boundary']

        # grid
        self.grid = grid
        
        # dummy
        self.zeros = np.zeros(len(self.grid['z']))

        # profile parameters
        self.porosity = profile_propeties['porosity']
        self.bedrock_thermal_conductivity = profile_propeties['bedrock_thermal_conductivity']
        self.solid_composition = profile_propeties['solid_composition']
        self.freezing_curve = profile_propeties['freezing_curve']
        
        self.solid_heat_capacity = profile_propeties['solid_heat_capacity']
        
        #.. if not as input, estimate from volumetric fractions
        ix = np.where(np.isnan(self.solid_heat_capacity))[0]
        self.solid_heat_capacity[ix] = solid_volumetric_heat_capacity(self.solid_composition['organic'][ix])

        # initialize state
        self.update_state(model_specs['initial_condition'], volumetric_water_content)

        # model specs
        if model_specs['solve']:
            info = 'Soil heat balance solved.'
            # Keep track of dt used in solution
            self.subdt = None
        else:
            info = 'Soil heat balance not solved. Using prescribed (time-dependent) inputs'

        logger.info(info)

    def run(self, 
            dt: float, 
            forcing: Dict, 
            volumetric_water_content: np.ndarray, 
            heat_sink: np.ndarray=None, 
            lower_boundary: Dict=None) -> Tuple:

        r"""
        Computes soil heat balance for timestep dt.

        Args:
            dt (float): time step [s]
            forcing (dict):
                ground_heat_flux (float): heat flux from soil surface [W m-2]. Give if ubc is flux-based
                temperature (float): soil surface temperature [degC]. Give if ubc is soil surface temperature
            volumetric_water_content (array):  [m3 m-3]
            heat_sink (array or None): layerwise sink term [W m-3] ! if None set to zero
            lower_boundary (dict or None): allows boundary condition type and value to be changed during simulations
                type (str): 'flux' or 'temperature'
                value (float): value of flux [W m-2] or temperature [degC]
        Returns:
            fluxes (dict):
                energy_closure (float): [W m-2]
                heat_flux (array): heat flux between layers [W m-2]
        
        """
        if self.subdt is None:
            self.subdt = dt

        if lower_boundary is not None:  # allows lower boundary to change in time
            self.lbc = lower_boundary

        if 'ground_heat_flux' in forcing:
            upper_boundary ={'type': 'flux', 'value': forcing['ground_heat_flux']}
        else:
            upper_boundary ={'type': 'temperature', 'value': forcing['temperature']}

        if heat_sink is None:
            heat_sink = self.zeros.copy()

        fluxes, state, self.subdt = heatflow1D(t_final=dt,
                                               grid=self.grid,
                                               T_ini=self.T,
                                               Wtot=volumetric_water_content,
                                               poros=self.porosity,
                                               solid_composition=self.solid_composition,
                                               cs=self.solid_heat_capacity,
                                               bedrockL=self.bedrock_thermal_conductivity,
                                               fp=self.freezing_curve,
                                               sink=heat_sink,
                                               ubc=upper_boundary,
                                               lbc=self.lbc,
                                               steps=dt / self.subdt,
                                               date=forcing['date'])

        self.update_state(state, volumetric_water_content)

        return fluxes

    def update_state(self, state: Dict={}, Wtot: np.ndarray=0.0) -> None:
        r"""
        Updates Heat-object state

        Args:
            state (dict):
                temperature (array|float): [degC]
                volumetric_ice_content (array|float): [m3 m-3]
                volumetric_water_content (array|float): [m3 m-3]
        Returns:
            None
        """
        if 'temperature' in state:
            if isinstance(state['temperature'], float):
                self.T = self.zeros + state['temperature']
            else:
                self.T = np.array(state['temperature'])
        if 'volumetric_ice_content' in state:
            self.Wice = state['volumetric_ice_content']
        else:
            _, self.Wice, _ = frozen_water(self.T, Wtot, self.freezing_curve, To=FREEZING_POINT_H2O)
        if 'volumetric_liquid_water_content' in state:
            self.Wliq = state['volumetric_liquid_water_content']
        else:
            self.Wliq = Wtot - self.Wice
        self.Wair = self.porosity - Wtot
        self.thermal_conductivity = thermal_conductivity(self.porosity, self.Wliq, self.Wice,
                                                         solid_composition=self.solid_composition,
                                                         bedrockL=self.bedrock_thermal_conductivity)


#%%
#-- Soil heat transfer in 1D profile

def heatflow1D(t_final: float, grid: Dict, T_ini: np.ndarray, Wtot: np.ndarray, 
               poros: np.ndarray, solid_composition: Dict, cs: np.ndarray, bedrockL: np.ndarray, 
               fp: np.ndarray, sink: np.ndarray, 
               ubc: Dict, lbc: Dict, steps: int=10, date: str=None)-> Tuple:
    r"""
    Solves soil heat conduction in 1-D using implicit finite difference solution
    of heat (Fourier) equation. Neglects heat transfer (heat convection) with water flow.

    References:
        Adapted from method presented by Van Dam and Feddes (2000) and Hansson et al., 200x

    Notes:
        Heat convection within column and from/to column neglected. 
        Some problems with convergence can occur around zero temperatures (freeze/thaw). Check energy closure
        to identify those periods if exist.

    Args:
        t_final (float): solution timestep [s]
        grid (dict):
            z: node elevations, soil surface = 0.0 [m]
            dz: thickness of computational layers [m]
            dzu: distance to upper node [m]
            dzl: distance to lower node [m]
        T_ini (array): initial temperature [degC]
        Wtot (array): volumetric water (and ice) content [m3 m-3]
        poros (array): porosity [m3 m-3]
        solid_composition (dict): fractions of solid volume [-]
            organic' (array)
            sand (array)
            silt (array)
            clay (array)
        solid_heat_capacity (array): [J m-3 (solid volume) K-1]
        bedrockL (array): bedrock thermal conductivity, NaN above bedrock depth [W m-1 K-1]
        fp (array): freezing curve parameter [-]
        sink: heat sink term from layers [W m-3]
        ubc (dict): upper boundary condition
            type (str): 'flux' or 'temperature'
            value (float): value of flux [W m-2] or temperature [degC]
        lbc (dict): lower boundary condition
            type (str): 'flux' or 'temperature'
            value (float): value of flux [W m-2] or temperature [degC]
        steps (int or float): initial number of subtimesteps used to proceed to 't_final'
        date (str): for logger
    
    Returns:
        fluxes (dict):
            energy_closure (float): [W m-2]
            heat_flux (array): heat flux between nodes [W m-2]
        state (dict):
            temperature (array): [degC]
            volumetric_ice_content (array): [m3 m-3]
        dto (float): internal timestep used [s]

    """

    # grid
    z = grid['z']
    dz = grid['dz']
    dzu = grid['dzu']
    dzl = grid['dzl']

    N = len(z)

    # initiate
    T = T_ini.copy()
    # initiate heat flux array [W/m3]
    Fheat = np.zeros(N+1)
    # Liquid and ice content, and dWliq/dTs
    Wliq, Wice, gamma = frozen_water(T, Wtot, fp=fp, To=FREEZING_POINT_H2O)

    # specifications for iterative solution
    Conv_crit1 = 1.0e-3  # degC
    Conv_crit2 = 1.0e-5  # ice content m3/m3
    # running time [s]
    t = 0.0
    # initial and computational time step [s]
    dto = t_final / steps
    dt = dto  # adjusted during solution

    """ solve water flow for 0...t_final """
    while t < t_final:
        # these will stay during iteration over time step "dt"
        T_old = T
        Wice_old = Wice

        # old bulk soil heat capacity [Jm-3K-1]
        CP_old = volumetric_heat_capacity(poros, cs, Wtot, Wice)

        # thermal conductivity - this remains constant during iteration
        Kt = thermal_conductivity(poros, Wliq, Wice, solid_composition, bedrockL)
        Kt = spatial_average(Kt, method='arithmetic')

        # changes during iteration
        T_iter = T.copy()
        Wice_iter = Wice.copy()

        err1 = 999.0
        err2 = 999.0
        iterNo = 0

        """ iterative solution of heat equation """
        while err1 > Conv_crit1 or err2 > Conv_crit2:

            iterNo += 1

            # bulk soil heat capacity [Jm-3K-1]
            CP = volumetric_heat_capacity(poros, cs, Wtot, Wice)
            # heat capacity due to freezing/thawing [Jm-3K-1]
            A = ICE_DENSITY*LATENT_HEAT_FREEZING*gamma

            """ set up tridiagonal matrix """
            a = np.zeros(N)
            b = np.zeros(N)
            g = np.zeros(N)
            f = np.zeros(N)

            # intermediate nodes
            b[1:-1] = CP[1:-1] + A[1:-1] + dt / dz[1:-1] * (Kt[1:N-1] / dzu[1:-1] + Kt[2:N] / dzl[1:-1])
            a[1:-1] = - dt / (dz[1:-1]*dzu[1:-1]) * Kt[1:N-1]
            g[1:-1] = - dt / (dz[1:-1]*dzl[1:-1]) * Kt[2:N]
            f[1:-1] = CP_old[1:-1] * T_old[1:-1] + A[1:-1] * T_iter[1:-1] \
                    + LATENT_HEAT_FREEZING*ICE_DENSITY*(Wice_iter[1:-1] - Wice_old[1:-1]) - sink[1:-1] * dt

            # top node i=0
            if ubc['type'] == 'flux':  # flux bc
                F_sur = ubc['value']
                b[0] = CP[0] + A[0] + dt / (dz[0] * dzl[0]) * Kt[1]
                a[0] = 0.0
                g[0] = -dt / (dz[0] * dzl[0]) * Kt[1]
                f[0] = CP_old[0]*T_old[0] + A[0]*T_iter[0] + LATENT_HEAT_FREEZING*ICE_DENSITY*(Wice_iter[0] - Wice_old[0])\
                        - dt / dz[0] * F_sur - dt*sink[0]

            if ubc['type'] == 'temperature':  # temperature bc
                T_sur = ubc['value']
                b[0] = CP[0] + A[0] + dt / dz[0]* (Kt[0] / dzu[0] + Kt[1] / dzl[0])
                a[0] = 0.0
                g[0] = -dt / (dz[0]*dzl[0]) * Kt[1]
                f[0] = CP_old[0]*T_old[0] + A[0]*T_iter[0] + LATENT_HEAT_FREEZING*ICE_DENSITY*(Wice_iter[0] - Wice_old[0])\
                        + dt / (dz[0]*dzu[0])*Kt[0]*T_sur - dt*sink[0]

            # bottom node i=N
            if lbc['type'] == 'flux':  # flux bc
                F_bot = lbc['value']
                b[-1] = CP[-1] + A[-1] + dt / (dz[-1]*dzu[-1]) * Kt[N-1]
                a[-1] = -dt / (dz[-1]*dzu[-1]) * Kt[N-1]
                g[-1] = 0.0
                f[-1] = CP_old[-1]*T_old[-1] + A[-1]*T_iter[-1] + LATENT_HEAT_FREEZING*ICE_DENSITY*(Wice_iter[-1] - Wice_old[-1])\
                        - dt / dz[-1]*F_bot - dt*sink[-1]

            if lbc['type'] == 'temperature':  # temperature bc
                T_bot = lbc['value']
                b[-1] = CP[-1] + A[-1] + dt / dz[-1] * (Kt[N-1] / dzu[-1] + Kt[N] / dzl[-1])
                a[-1] = -dt / (dz[-1]*dzu[-1]) * Kt[N-1]
                g[-1] = 0.0
                f[-1] = CP_old[-1]*T_old[-1] + A[-1]*T_iter[-1] + LATENT_HEAT_FREEZING*ICE_DENSITY*(Wice_iter[-1] - Wice_old[-1])\
                        + dt/(dz[-1]*dzl[-1]) * Kt[N]*T_bot - dt*sink[-1]

            """ solve triagonal matrix system """
            # save old iteration values
            T_iterold = T_iter.copy()
            Wice_iterold = Wice_iter.copy()

            # solve new temperature and ice content
            T_iter = thomas(a, b, g, f)
            Wliq_iter, Wice_iter, gamma = frozen_water(T_iter, Wtot, fp=fp, To=FREEZING_POINT_H2O)

            # if problems reaching convergence devide time step and retry
            if iterNo == 20:
                if dt / 3.0 > 101.0:
                    dt = max(dt / 3.0, 300.0)
#                    logger.debug('%s (iteration %s) More than 20 iterations, retry with dt = %.1f s',
#                                 date, iterNo, dt)
                    iterNo = 0
                    T_iter = T_old.copy()
                    Wice_iter = Wice_old.copy()
                    continue
                else:  # convergence not reached with dt=300.0s, break
#                    logger.debug('%s (iteration %s) Solution not converging, err_T: %.5f, err_Wice: %.5f, dt = %.1f s',
#                                 date, iterNo, err1, err2, dt)
                    if err1 > 0.01:
                        T_iter = T_old.copy()
                        Wice_iter = Wice_old.copy()
                    break
            # check solution, if problem continues break
            elif any(np.isnan(T_iter)):
                if dt / 3.0 > 101.0:
                    dt = max(dt / 3.0, 300.0)
                    logger.debug('%s (iteration %s) Solution blowing up, retry with dt = %.1f s',
                                 date, iterNo, dt)
                    iterNo = 0
                    T_iter = T_old.copy()
                    Wice_iter = Wice_old.copy()
                    continue
                else:  # convergence not reached with dt=300.0s, break
                    logger.debug('%s (iteration %s) No solution found (blow up), T and Wice set to old values.',
                                 date,
                                 iterNo)
                    T_iter = T_old.copy()
                    Wice_iter = Wice_old.copy()
                    break

            # errors for determining convergence
            err1 = np.max(abs(T_iter - T_iterold))
            err2 = np.max(abs(Wice_iter - Wice_iterold))

        """ iteration loop ends """

        # new state at t
        T = T_iter.copy()
        # Liquid and ice content, and dWice/dTs
        Wliq, Wice, gamma = frozen_water(T, Wtot, fp=fp, To=FREEZING_POINT_H2O)
        # Heat flux [W m-2]
        Fheat[1:-1] += -Kt[1:-1]*(T[1:] - T[:-1])/dzl[:-1] * dt / t_final
        if ubc['type'] == 'temperature':
            Fheat[0] += -Kt[0]*(T[0] - T_sur)/dzu[0] * dt / t_final
        if ubc['type'] == 'flux':
            Fheat[0] += -F_sur * dt / t_final
        if lbc['type'] == 'temperature':
            Fheat[-1] += -Kt[-1]*(T_bot - T[-1])/dzl[-1] * dt / t_final
        if lbc['type'] == 'flux':
            Fheat[-1] += -F_bot * dt / t_final

        """ solution time and new timestep """
        t += dt

        dt_old = dt  # save temporarily
        # select new time step based on convergence
        if iterNo <= 3:
            dt = dt * 2.0
        elif iterNo >= 6:
            dt = dt / 2.0

        # limit to minimum of 300s
        dt = max(dt, 300.0)

        # save dto for output to be used in next run of heatflow1D()
        if dt_old == t_final or t_final > t:
            dto = min(dt, t_final)

        # limit by time left to solve
        dt = min(dt, t_final-t)

    """ time stepping loop ends """

    def heat_content(x):
        _, wice, _ = frozen_water(x, Wtot, fp=fp, To=FREEZING_POINT_H2O)
        Cv = volumetric_heat_capacity(poros, cs, wtot=Wtot, wice=wice)
        return Cv * x - LATENT_HEAT_FREEZING * ICE_DENSITY * wice

    def heat_balance(x):
        return heat_content(T_ini) + t_final * (Fheat[:-1] - Fheat[1:]) / dz - heat_content(x)

    heat_be = sum(dz * heat_balance(T))

    fluxes = {'energy_closure': heat_be / t_final,
              'heat_flux': Fheat[:-1]}

    state = {'temperature': T,
             'volumetric_ice_content': Wice}

    return fluxes, state, dto

# ---utility functions

def frozen_water(T: np.ndarray, wtot: np.ndarray, fp: float=2.0, To: float=0.0) -> Tuple:
    r""" 
    Approximates ice content from soil temperature and total water content.

    References:
        For peat soils, see experiment of Nagare et al. 2012:
        http://scholars.wlu.ca/cgi/viewcontent.cgi?article=1018&context=geog_faculty
    Args:
        T (array|float): soil temperature [degC]
        wtot (array|float): total volumetric water content [m3 m-3]
        fp (array|float): parameter of freezing curve [-]. 2...4 for clay and 0.5-1.5 for sandy soils; 
            < 0.5 for peat soils (Nagare et al. 2012 HESS)
        To (array|float): freezing temperature of soil water [degC]
    
    Returns:
        wliq (array|float): volumetric water content [m3 m-3]
        wice (array|float): volumetric ice content [m3 m-3]
        gamma (array|float): dwice/dT

    """

    wtot = np.array(wtot)
    T = np.array(T)
    fp = np.array(fp)

    wice = wtot*(1.0 - np.exp(-(To - T) / fp))
    # derivative dwliq/dT
    gamma = (wtot - wice) / fp

    ix = np.where(T > To)[0]
    wice[ix] = 0.0
    gamma[ix] = 0.0

    wliq = wtot - wice

    return wliq, wice, gamma

def solid_volumetric_heat_capacity(f_org: float) -> float:
    r"""
    Computes volumetric heat capacity of solids in soil
    based on organic and bulk mineral volume fractions. 
    Uses single bulk value for soil minerals heat capacity.

    Args:
        f_org (array|float): fraction of organic matter of solid volume [-]
    Returns:
        cv_solids: volumetric heat capacity of solids [J m-3 (solids) K-1]
    """
    cv_solids = f_org * CV_ORGANIC + (1. - f_org) * CV_MINERAL

    return cv_solids

def volumetric_heat_capacity(poros, cv_solids, wtot=0.0, wice=0.0):
    r"""
    Computes volumetric heat capacity of soil from soil volumetric texture, water and ice fractions.

    Args:
        poros (array|float): porosity [m3 m-3]
        cv_solids (array|float): heat capacity of solids [J m-3(solids) K-1]
        wtot (array|float): volumetric water content [m3 m-3]
        wice (array|float): volumetric ice content [m3 m-3]
    Returns:
        cv (array|float): volumetric heat capacity of soil [J m-3 K-1]
    """
    wair = poros - wtot
    wliq = wtot - wice

    cv = cv_solids * (1. - poros) + CV_WATER * wliq + CV_ICE * wice + CV_AIR * wair

    return cv

def thermal_conductivity(poros, wliq, wice, solid_composition, bedrockL):
    r"""
    Thermal conductivity in 1D soil profile

    Uses Tian et al. 2016 simplified de Vries-model for mineral layers. 
    For organic layers (org. fract. > 0.9) uses o'Donnell et al. 2009.
    The deVries-type model gives reasonable approximation also in organic layers when
    wliq < 0.4, at wliw = 0.9 ~25% underestimate compared to o'Donnell.

    References:
        Tian et al. 2016: A simplified de Vries-based model to estimate thermal
        conductivity of unfrozen and frozen soil. E.J.Soil Sci 67, 564-572.
        o'Donnell et al. 2009. Thermal conductivity of organic soil horizons. Soil Sci. 174, 646-651.
    
    Args:
        poros (array): porosity [m3 m-3]
        wliq (array): volumetric liquid water content [m3 m-3]
        wice (array): volumetric ice content [m3 m-3]
        solid_composition (dict): fractions of solid volume [-]
            organic (array)
            sand (array)
            silt (array)
            clay (array)
        bedrockL (array): bedrock thermal conductivity, NaN above bedrock depth [W m-1 K-1]
    Returns:
        L (array): thermal conductivity [W m-1 K-1]

    """
    f_sand = solid_composition['sand']
    f_silt = solid_composition['silt']
    f_clay = solid_composition['clay']
    f_org = solid_composition['organic']

    poros = np.array(poros, ndmin=1)
    wliq = np.array(wliq, ndmin=1)
    wice = np.array(wice, ndmin=1)
    wair = poros - wliq - wice

    f_solid = 1 - poros

    # component thermal conductivities W m-1 K-1
    ks = K_SAND**f_sand * K_SILT**f_silt * K_CLAY**f_clay * K_ORG**f_org
    kw = K_WATER
    ki = K_ICE
    kg = K_AIR

    # shape factors
    g_sand = 0.182
    g_silt = 0.0534
    g_clay = 0.00775
    g_org = 0.33  # campbell & norman, 1998
    g_min = g_sand * f_sand + g_silt * f_silt + g_clay * f_clay + g_org * f_org

    g_air = 0.333 * (1.0 - wair / poros)
    g_ice = 0.333 * (1.0 - wice / poros)

     # --- weight factors [-]------
    r = 2.0 / 3.0
    # solid minerals
    f_min = r*(1.0 + g_min*(ks / kw - 1.0))**(-1) + (1.0 - r)*(1.0 + (1.0-2*g_min)*(ks / kw - 1.0))**(-1.)
    # gas phase
    f_gas = r*(1.0 + g_air*(kg / kw - 1.0))**(-1) + (1.0 - r)*(1.0 + (1.0 - 2*g_air)*(kg / kw - 1.0))**(-1.)
    # liquid phase
    f_ice = r*(1.0 + g_ice*(ki / kw - 1.0))**(-1) + (1.0 - r)*(1.0 + (1.0-2*g_ice)*(ki / kw - 1.0))**(-1.)

    # thermal conductivity W m-1 K-1

    # L = (wliq*kw + f_gas*wair*kg + f_min*poros*ks + f_ice*wice*ki) \
    #     / (wliq + f_gas*wair + f_min*poros + f_ice*wice)

    L = (wliq*kw + f_gas*wair*kg + f_min*f_solid*ks + f_ice*wice*ki) \
        / (wliq + f_gas*wair + f_min*f_solid + f_ice*wice)

    # in fully organic layer, use o'Donnell et al. 2009
    L[f_org >= 0.9] = 0.032 + 5e-1 * wliq[f_org >= 0.9]

    # in bedrock
    L[~np.isnan(bedrockL)] = bedrockL[~np.isnan(bedrockL)]

    return L

# EOF