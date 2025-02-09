# -*- coding: utf-8 -*-
"""
.. module: soil.water
    :synopsis: pyAPES soil component
.. moduleauthor:: Kersti Leppä & Samuli Launiainen

Vertically-resolved soil water balance.

References:
    vanDam & Feddes (2000): Numerical simulation of infiltration, evaporation and shallow
    groundwater levels with the Richards equation, J.Hydrol 233, 72-85.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from typing import Dict, List, Tuple
import logging

from pyAPES.utils.utilities import tridiag as thomas, spatial_average
from pyAPES.utils.constants import EPS, WATER_DENSITY

logger = logging.getLogger(__name__)

class Water_1D(object):

    def __init__(self, grid: Dict, profile_propeties: Dict, model_specs: Dict):
        r""" 
        1D soil water balance model using Richards-equation or Equilibrium scheme.

        Args:
            grid (dict):
                z (array): node elevations, soil surface = 0.0 [m]
                dz (array): thickness of computational layers [m]
                dzu (array): distance to upper node [m]
                dzl (array): distance to lower node [m]
            
            profile_propeties (dict): arrays of len(z)
                pF (dict): water retention parameters (van Genuchten)
                    ThetaS (array): saturated water content [m3 m-3]
                    ThetaR (array): residual water content [m3 m-3]
                    alpha (array): air entry suction [cm-1]
                    n (array): pore size distribution [-]
                saturated_conductivity_vertical (array): [m s-1]
                saturated_conductivity_horizontal (array): [m s-1]
            
            model_specs (dict):
                solve (bool): solve water flow
                type (str): 'Richards' | 'Equilibrium'. Solution approach
                pond_storage_max (float):  maximum pond depth [m]
                
                initial_condition (dict):
                    ground_water_level (float): groundwater depth [m]
                    pond_storage (float): pond depth at surface [m]
                
                lower_boundary (dict): lower boundary condition
                    type (str): 'impermeable', 'flux', 'free_drain' or 'head'
                    value (float or None): give for 'head' [m] and 'flux' [m s-1]
                    depth (float): depth of impermeable boundary (=bedrock) [m]
                
                drainage_equation (dict):
                    type (str): 'hooghoudt' / None (no other options yet)
                    depth (float): drain depth [m]
                    spacing (float): drain spacing [m]
                    width (float):  drain width [m]
        Returns:
            self (object)
        """

        # lower boundary condition
        self.lbc = model_specs['lower_boundary']

        # in case there is impermeable layer (bedrock bottom), water flow is solved only above this
        if self.lbc['type'] == 'impermeable':
            bedrock_depth = self.lbc['depth']
        else:
            bedrock_depth = grid['z'][-1]
        self.ix = np.where(grid['z'] >= bedrock_depth)
        self.grid = {key: grid[key][self.ix] for key in grid.keys()}
        # dummy
        self.zeros = np.zeros(len(self.grid['z']))

        # profile parameters
        self.pF = {key: profile_propeties['pF'][key][self.ix] for key in profile_propeties['pF'].keys()}
        self.Kvsat = profile_propeties['saturated_conductivity_vertical'][self.ix]
        self.Khsat = profile_propeties['saturated_conductivity_horizontal'][self.ix]

        # initialize state
        self.update_state(model_specs['initial_condition'])

        # model specs
        self.h_pond_max = model_specs['pond_storage_max']

        if model_specs['solve']:
            self.solution_type = model_specs['type'].upper()
            if self.solution_type == 'EQUILIBRIUM':
                # interpolated functions for soil column ground water dpeth vs. water storage
                self.wsto_gwl_functions = gwl_Wsto(self.grid['dz'], self.pF)
            else:
                self.solution_type = 'RICHARDS EQUATION'
            info = 'Water balance in soil solved using: %s' % self.solution_type
            # drainage equation
            self.drainage_equation = model_specs['drainage_equation']
            if self.drainage_equation['type'] == None:
                info += ' & no lateral drainage'
            else:
                self.drainage_equation['type'] = model_specs['drainage_equation']['type'].upper()
                if self.drainage_equation['type'] != 'HOOGHOUDT':
                    raise ValueError("Unknown drainage equation type %s"
                                     % self.drainage_equation['type'])
                info += ' & %s' % self.drainage_equation['type']
            # Keep track of dt used in solution
            self.subdt = None
        else:
            info = 'Water balance in soil not solved.'

        logger.info(info)

    def run(self, dt: float, forcing: Dict, water_sink: np.ndarray=None, lower_boundary: Dict=None) -> Dict:
        r""" Runs soil water balance.
        Args:
            dt: time step [s]
            forcing (dict):
                potential_infiltration: [m s-1]
                potential_evaporation: [m s-1]
                atmospheric_pressure_head: [m]
            water_sink (array|None): sink term from layers, e.g. root sink [m s-1]
                ! array length can be only root zone or whole soil profile; if None set to zero
            lower_boundary (dict or None): allows boundary to be changed in time
                type: 'impermeable', 'flux', 'free_drain', 'head'
                value: give for 'head' [m] and 'flux' [m s-1]
        Returns:
            fluxes (dict): [m s-1]
                infiltration
                evaporation
                drainage
                transpiration
                surface_runoff
                water_closure
        """
        if self.subdt is None:
            self.subdt = dt

        if lower_boundary is not None:  # allows lower boundary to change in time
            self.lbc = lower_boundary

        rootsink = self.zeros.copy()
        if water_sink is not None:
            rootsink[0:min(len(water_sink), len(rootsink))] = water_sink
            rootsink = rootsink / self.grid['dz']  # [m3 m-3 s-1]

        # drainage from each layer [m m-1 s-1]
        if self.drainage_equation['type'] == 'HOOGHOUDT':
            drainage_flux = drainage_hooghoud(self.grid['dz'],
                                              self.Khsat,
                                              self.gwl,
                                              self.drainage_equation['depth'],
                                              self.drainage_equation['spacing'],
                                              self.drainage_equation['width'])
        else:
            drainage_flux = self.zeros.copy()  # no drainage

        # initial state
        initial_state = {'water_potential': self.h,
                         'pond_storage': self.h_pond}

        # solve water balance in 1D soil profile
        if self.solution_type == 'EQUILIBRIUM':  # solving based on equilibrium
            fluxes, state = waterStorage1D(t_final=dt,
                                           grid=self.grid,
                                           forcing=forcing,
                                           initial_state=initial_state,
                                           pF=self.pF,
                                           Ksat=self.Kvsat,
                                           wsto_gwl=self.wsto_gwl_functions,
                                           q_sink=rootsink,
                                           q_drain=drainage_flux,
                                           lbc=self.lbc,
                                           h_pond_max=self.h_pond_max)

        else:  # Richards equation for solving water flow
            fluxes, state, self.subdt = waterFlow1D(t_final=dt,
                                                    grid=self.grid,
                                                    forcing=forcing,
                                                    initial_state=initial_state,
                                                    pF=self.pF,
                                                    Ksat=self.Kvsat,
                                                    q_sink=rootsink,
                                                    q_drain=drainage_flux,
                                                    lbc=self.lbc,
                                                    h_pond_max=self.h_pond_max,
                                                    steps=dt / self.subdt)

        self.update_state(state)

        return fluxes

    def update_state(self, state: Dict):
        r""" 
        Updates object state
        
        Args:
            state (dict):
                'ground_water_level' (float):  [m]
                'pond_storage' (float): [m]
                'water_potential' (array): [m]
                'volumetric_water_content' (array): [m]
        Returns:
            None, updates .gwl, (.h_pond), .h, .Wtot, .Kv, .Kh
        """
        if 'ground_water_level' in state:
            self.gwl = state['ground_water_level']
            
            if 'water_potential' in state:
                self.h = state['water_potential']
            else:
                self.h = self.gwl - self.grid['z']  # steady state
            
            if 'volumetric_water_content' in state:
                if isinstance(state['volumetric_water_content'], float):
                    self.Wtot = self.zeros + state['volumetric_water_content']
                else:
                    self.Wtot = np.array(state['volumetric_water_content'])
            else:
                self.Wtot = h_to_cellmoist(self.pF, self.h, self.grid['dz'])
        
        elif 'volumetric_water_content' in state:
            if isinstance(state['volumetric_water_content'], float):
                self.Wtot = self.zeros + state['volumetric_water_content']
            else:
                self.Wtot = np.array(state['volumetric_water_content'])
            self.Wtot = np.minimum(self.Wtot, self.pF['ThetaS'])
            self.h = wrc(self.pF, self.Wtot, var='Th')
            self.gwl = get_gwl(self.h, self.grid['z'])
        else:
            raise ValueError("Problem in water.update_state()")

        if 'pond_storage' in state:
            self.h_pond = state['pond_storage']
        self.Kv = hydraulic_conductivity(self.pF, x=self.h, Ksat=self.Kvsat)
        self.Kh = hydraulic_conductivity(self.pF, x=self.h, Ksat=self.Khsat)

#%%% SAMULI ADDING BUCKET MODEL HERE
        
class WaterBucket(object):
    """
    Single-layer soil water bucket model with bulk properties
    References: Loosely following Guswa et al, 2002 WRR
    """

    def __init__(self, p: Dict, Wliq: float, lower_boundary: Dict, SurfSto: float=0.0):
        """
        Args:
            p (dict):
                'depth' (float) - bucket depth [m]
                'pF' (dict) - vanGenuchten water retention parameters {dict}
                'Ksat' (float)- saturated hydr. cond. [ms-1]
                'MaxPond' (float)- maximum ponding depth above bucket [m]
            Wliq (float) - initial vol. water content [m3m-3]    
            lower_boundary (dict): lower boundary condition
                    type (str): 'impermeable', 'flux', 'free_drain' or 'head'
                    value (float or None): give for 'head' [m] and 'flux' [m s-1]
                    depth (float): depth of impermeable boundary (=bedrock) [m]
        """
        
        self.D = p['depth']
        self.Ksat = p['Ksat']
        self.pF = p['pF']
        
        self.Fc = psi_theta(self.pF, x=-1.0)
        self.Wp = psi_theta(self.pF, x=-150.0)
        
        self.poros = self.pF['ThetaS'] # porosity
        
        # water storages
        self.MaxSto = self.poros * self.D
        self.MaxPond = p['MaxPond']
        
        # initial state
        self.SurfSto = SurfSto
        self.WatSto = self.D * p['Wliq']
        self.Wliq = p['Wliq']
        self.Wair = self.poros - self.Wliq        
        self.Sat = self.Wliq / self.poros
        self.h = theta_psi(self.pF, self.Wliq)
        self.Kh = self.Ksat * hydraulic_conductivity(self.pF, self.h)
        
        # relatively extractable water
        self.Rew = np.minimum( (self.Wliq - self.Wp) / (self.Fc - self.Wp + EPS), 1.0)
                        
    def update_state(self, dWat: float, dSur: float=0.0):
        """
        updates state by computed dSto [m] and dSurf [m] 
        """
        self.WatSto += dWat
        self.SurfSto +=dSur
        
        self.Wliq = self.poros * self.WatSto / self.MaxSto
        self.Wair = self.poros - self.Wliq
        self.Sat = self.Wliq/self.poros
        self.h = theta_psi(self.pF, self.Wliq)
        self.Kh = self.Ksat * hydraulic_conductivity(self.pF, self.h)            
        self.Rew = np.minimum((self.Wliq - self.Wp) / (self.Fc - self.Wp + EPS), 1.0)      
        
    def run(self, dt: float, rr: float=0.0, et: float=0.0, latflow: float=0.0):
        r""" Bucket model water balance
        Args: 
            dt [s]
            rr - potential infiltration [mm s-1 = kg m-2 s-1]
            et - evapotranspiration sink [mm s-1]
            latflow - lateral sink/source term[mm s-1]
        Returns: 
            infil [mm] - infiltration [m]
            Roff [mm] - surface runoff
            drain [mm] - percolation /leakage
            roff [mm] - surface runoff
            et [mm] - evapotranspiration
            mbe [mm] - mass balance error
        """
        # fluxes        
        Qin = (rr + latflow) * dt / WATER_DENSITY # m, potential inputs   
        et = et * dt / WATER_DENSITY
        
        # free drainage from profile
        drain = min(self.Kh * dt, max(0, (self.Wliq - self.Fc)) * self.D) # m
                
        # infiltr is restricted by availability, Ksat or available pore space
        infil = min(Qin, self.Ksat*dt, (self.MaxSto - self.WatSto + drain + et)) 

        # change in surface and soil water store
        SurfSto0 = self.SurfSto       
        dSto = (infil - drain - et)
        
        #in case of Qin excess, update SurfSto and route Roff
        q = Qin - infil
        if q > 0:
            dSur = min(self.MaxPond - self.SurfSto, q)
            roff = q - dSur #runoff, m
        else:
            dSur = 0.0
            roff = 0.0
        
        #update state variables
        self.update_state(dSto, dSur)
        
        #mass balance error
        mbe = dSto + (self.SurfSto - SurfSto0) - (rr + latflow - et - drain - roff)
                    
        return WATER_DENSITY * infil, WATER_DENSITY * roff, WATER_DENSITY * drain, WATER_DENSITY * mbe 


# %%
# --- functions to solve water balance in a 1D column

def waterFlow1D(t_final: float, grid: np.ndarray, forcing: Dict, initial_state: Dict, 
                pF: Dict, Ksat: np.ndarray, q_sink: np.ndarray, q_drain: np.ndarray=0.0, 
                lbc: Dict={'type': 'impermeable', 'value': None},
                h_pond_max: float=0.0, cosalfa: float=1.0, steps: float=10) -> Tuple:
    r"""
    Solves soil water flow in 1-D using implicit, backward finite difference solution of Richard's equation.

    Args:
        t_final (float): solution timestep [s]
        grid (dict):
            z (array): node elevations, soil surface = 0.0 [m]
            dz (array): thickness of computational layers [m]
            dzu (array): distance to upper node [m]
            dzl (array): distance to lower node [m]
        forcing (dict):
            potential_infiltration (float): [m s-1]
            potential_evaporation (float): [m s-1]
            atmospheric_pressure_head (float): [m]
        initial_state (dict):
            water_potential (array): initial water potential [m]
            pond_storage (float): initial pond storage [m]
        pF (dict): water retention parameters (van Genuchten)
            ThetaS (array): saturated water content [m3 m-3]
            ThetaR (array):residual water content [m3 m-3]
            alpha (array):air entry suction [cm-1]
            n (array):pore size distribution [-]
        Ksat (array): saturated hydraulic conductivity [m s-1]
        q_sink (array): water sink term from layers, e.g. root sink [m3 m-3 s-1 = s-1]
        q_drain (array): water sink due to drainage per layer [m3 m-3 s-1 = s-1]
        lbc (dict):
                'type': 'impermeable', 'flux', 'free_drain', 'head', 'head_oneway'
                'value': give for flux [m s-1] or head [m] -based lbc 
        h_pond_max (float): maximum depth allowed ponding at surface [m]
        cosalfa (float): 1 for vertical water flow, 0 for horizontal
        steps (int): initial number of subtimesteps used to proceed to 't_final'
    
    Returns:
        fluxes (dict): [m s-1]
            infiltration (float)
            evaporation (float)
            drainage (float)
            transpiration (float)
            surface_runoff (float)
            water_closure (float)
        state (dict):
            water_potential: [m]
            volumetric_water_content: [m3 m-3]
            ground_water_level: [m]
            pond_storage: [m]
        dto (float): internal solver timestep used [s]

    References:
        vanDam & Feddes (2000): Numerical simulation of infiltration, evaporation and shallow
        groundwater levels with the Richards equation, J.Hydrol 233, 72-85.

    Note: Implement macropore bypass flow?
    """

    # forcing
    Prec = forcing['potential_infiltration']
    Evap = forcing['potential_evaporation']
    
    if 'atmospheric_pressure_head' in forcing:
        h_atm = forcing['atmospheric_pressure_head']
    else:
        h_atm = -1000.0

    # net sink/source term
    S = q_sink + q_drain  # root uptake + lateral flow (e.g. by ditches)

    # cumulative boundary fluxes for 0...t_final
    C_inf, C_eva, C_dra, C_trans, C_roff = 0.0, 0.0, 0.0, 0.0, 0.0

    # grid
    z = grid['z']
    dz = grid['dz']
    dzu = grid['dzu']
    dzl = grid['dzl']

    N = len(z)

    # soil hydraulic conductivity and porosity
    if type(Ksat) is float:
        Ksat = np.zeros(N) + Ksat
    poros = pF['ThetaS']

    # initial state
    h_ini = initial_state['water_potential']
    pond_ini = initial_state['pond_storage'] - forcing['pond_recharge'] * t_final
    W_ini = h_to_cellmoist(pF, h_ini, dz)

    # variables updated during solution
    W = W_ini.copy()
    h = h_ini.copy()
    h_pond = pond_ini

    # specifications for iterative solution
    # running time [s]
    t = 0.0
    
    # initial and computational time step [s]
    dto = t_final / steps
    dt = dto  # adjusted during solution
    if Prec > 0.0:  # decrease time step during precipitation
        dt = min(225.0, dt)
    
    # convergence criteria
    Conv_crit = 1.0e-12  # for soil moisture
    Conv_crit2 = 1.0e-10  # for pressure head, decreased to 1.0e-8 when profile saturated

    # hydraulic condictivity based on previous time step
    KLh = hydraulic_conductivity(pF, x=h, Ksat=Ksat)
    # get KLh at i-1/2, note len(KLh) = N + 1
#    KLh = spatial_average(KLh, method='arithmetic')
    KLh = spatial_average(KLh, method='geometric')

    # potential flux at the soil surface (< 0 infiltration)
    q0 = Evap - Prec - h_pond / t_final

    # maximum infiltration and evaporation rates
    MaxInf = max(-KLh[0]*(- q0 * t_final - h[0] - z[0]) / dzu[0], -Ksat[0])

    # net flow to soil profile during dt
    Qin = (- sum(S * dz) - q0) * dt

    # airvolume available in soil profile after previous time step
    Airvol = max(0.0, sum((poros - W) * dz))
    if Qin < Airvol and q0 < 0 and q0 < MaxInf:
        S[0:10] = S[0:10] - Prec / 10. / grid['dz'][0:10]
        Prec = 0.0
        dt = 30

    """ solve water flow for 0...t_final """
    while t < t_final:

        # old state variables, solution of previous times step
        h_old = h.copy()
        W_old = W.copy()

        # state variables updated during iteration of time step dt
        h_iter = h.copy()
        W_iter = W.copy()

        # hydraulic condictivity based on previous time step
        KLh = hydraulic_conductivity(pF, x=h_iter, Ksat=Ksat)

        # get KLh at i-1/2, note len(KLh) = N + 1
        #KLh = spatial_average(KLh, method='arithmetic')
        KLh = spatial_average(KLh, method='geometric')

        # TEST: upward flow K_unsat, downward flow "K_macro"
        # KLh[1:-1] = np.where((h[:-1] - h[1:] - dzu[1:]) < 0.0, KLh[1:-1], 5.0e-4)

        # initiate iteration
        err1 = 999.0
        err2 = 999.0
        iterNo = 0

        """ iterative solution of time step dt """
        while (err1 > Conv_crit or err2 > Conv_crit2):

            iterNo += 1

            """ lower boundary condition """
            if lbc['type'] == 'free_drain':
                q_bot = - min(KLh[-1] * cosalfa, 1e-6)  # avoid extreme high drainage
                lbcc_type = 'flux'
            elif lbc['type'] == 'impermeable':
                q_bot = 0.0
                lbcc_type = 'flux'
            elif lbc['type'] == 'flux':
                q_bot = np.sign(lbc['value']) * min(lbc['value'], KLh[-1] * cosalfa)
                lbcc_type = 'flux'
            elif lbc['type'] == 'head':
                h_bot = lbc['value']
                lbcc_type = 'head'
                # approximate flux to calculate Qin
                q_bot = -KLh[-1] * (h_iter[-1] - h_bot) / dzl[-1] - KLh[-1] * cosalfa
            elif lbc['type'] == 'head_oneway':
                h_bot = lbc['value']
                # approximate flux to calculate Qin
                q_bot = -KLh[-1] * (h_iter[-1] - h_bot) / dzl[-1] - KLh[-1] * cosalfa
                if q_bot > 0.0:
                    lbcc_type = 'flux'
                    q_bot = 0.0
                else:
                    lbcc_type = 'head'
            else:
                raise ValueError("Unknown lower boundary condition %s"
                     % lbc['type'])

            #--- upper boundary condition, swiching between flux and head as in van Dam and Feddes (2000)

            # potential flux at the soil surface (< 0 infiltration)
            q0 = Evap - Prec - h_pond / dt
            # maximum infiltration and evaporation rates
            MaxInf = max(-KLh[0]*(- q0 * dt - h_iter[0] - z[0]) / dzu[0], -Ksat[0])
            MaxEva = -KLh[0]*(h_atm - h_iter[0] - z[0]) / dzu[0]
            # net flow to soil profile during dt
            Qin = (q_bot - sum(S * dz) - q0) * dt
            # airvolume available in soil profile after previous time step
            Airvol = max(0.0, sum((poros - W_old) * dz))

            if q0 < 0:  # case infiltration
                if Airvol <= EPS:  # initially saturated profile
                    if Qin >= 0:  # inflow exceeds outflow
                        h_sur = min(Qin, h_pond_max)
                        ubc_flag = 'head'
#                        print 'saturated soil ponding water, h_sur = ' + str(h_sur) + ' h = ' + str(h_iter[0])
                    else:  # outflow exceeds inflow
                        q_sur = q0
                        ubc_flag = 'flux'
#                        print 'outflow exceeds inflow' + ' q_sur = ' + str(q0) + ' h_pond = ' + str(h_pond)
                        # saturated soil draining, set better initial guess
                        if iterNo == 1:
                            h_iter -= dz[0]
                            W_iter = h_to_cellmoist(pF, h_iter, dz)
                
                else:  # initially unsaturated profile
                    if Qin >= Airvol:  # only part of inflow fits into profile
                        h_sur = min(Qin - Airvol, h_pond_max)
                        ubc_flag = 'head'
#                        print 'only part fits into profile, h_sur = ' + str(h_sur) + ' h = ' + str(h_iter[0])
                    
                    else:  # all fits into profile
                        # set better initial guess, is this needed here?
                        if iterNo ==1 and Airvol < 1e-3:
                            h_iter -= dz[0]
                            W_iter = h_to_cellmoist(pF, h_iter, dz)
                        if q0 < MaxInf:  # limited by maximum infiltration
                            dt = 30.0
                            h_sur = - q0 * dt
                            ubc_flag = 'head'
#                            print('all fits into profile, h_sur = ' + str(h_sur) + ' MaxInf = ' + str(MaxInf)+ ' K_0 = ' + str(KLh[0])+ ' dt = ' + str(dt))
                        else:
                            q_sur = q0
                            ubc_flag = 'flux'
#                            print 'all fits into profile, q_sur = ' + str(q_sur) + ' Airvol = ' + str(Airvol) + ' MaxInf = ' + str(MaxInf) + ' h = ' + str(h_iter[0]) + ' hpond = ' + str(h_pond)

            else:  # case evaporation
                # if saturated soil draining, set better initial guess
                if iterNo == 1 and Airvol < 1e-3:
                    h_iter -= dz[0]
                    W_iter = h_to_cellmoist(pF, h_iter, dz)
                if q0 > MaxEva:
                    if h_iter[0] + z[0] < h_atm:
                        q_sur = 0.0
                        ubc_flag = 'flux'
                    else:
                        h_sur = h_atm
                        ubc_flag = 'head'
#                    print 'case evaporation, limited by atm demand, q0 = ' + str(q0) + ' MaxEva = ' + str(MaxEva)
                else:
                    q_sur = q0
                    ubc_flag = 'flux'
#                    print 'case evaporation, no limit, q_sur = ' + str(q_sur) + ' Airvol = ' + str(Airvol)

            # differential water capacity [m-1]
            C = diff_wcapa(pF, h_iter, dz)

            """ set up tridiagonal matrix """
            a = np.zeros(N)  # sub diagonal
            b = np.zeros(N)  # diagonal
            g = np.zeros(N)  # super diag
            f = np.zeros(N)  # rhs

            # intermediate nodes i=1...N-1
            b[1:-1] = C[1:-1] + dt / dz[1:-1] * (KLh[1:N-1] / dzu[1:-1] + KLh[2:N] / dzl[1:-1])
            a[1:-1] = - dt / (dz[1:-1] * dzu[1:-1]) * KLh[1:N-1]
            g[1:-1] = - dt / (dz[1:-1] * dzl[1:-1]) * KLh[2:N]
            f[1:-1] = C[1:-1] * h_iter[1:-1] - (W_iter[1:-1] - W_old[1:-1]) + dt / dz[1:-1]\
                        * (KLh[1:N-1] - KLh[2:N]) * cosalfa - S[1:-1] * dt

            # top node i=0
            if ubc_flag != 'head':  # flux bc
                b[0] = C[0] + dt / (dz[0] * dzl[0]) * KLh[1]
                a[0] = 0.0
                g[0] = -dt / (dz[0] * dzl[0]) * KLh[1]
                f[0] = C[0] * h_iter[0] - (W_iter[0] - W_old[0]) + dt / dz[0]\
                        * (-q_sur - KLh[1] * cosalfa) - S[0] * dt
            else:  # head boundary
                b[0] = C[0] + dt / dz[0] * (KLh[0] / dzu[0] + KLh[1] / dzl[0])
                a[0] = 0
                g[0] = -dt / (dz[0] * dzl[0]) * KLh[1]
                f[0] = C[0] * h_iter[0] - (W_iter[0] - W_old[0]) + dt / dz[0]\
                        * ((KLh[0] - KLh[1]) * cosalfa + KLh[0] / dzu[0] * h_sur) - S[0] * dt

            # bottom node i=N
            if lbcc_type == 'flux':  # flux bc
                b[-1] = C[-1] + dt / (dz[-1] * dzu[-1]) * KLh[N-1]
                a[-1] = -dt / (dz[-1] * dzu[-1]) * KLh[N-1]
                g[-1] = 0.0
                f[-1] = C[-1] * h_iter[-1] - (W_iter[-1] - W_old[-1]) + dt / dz[-1]\
                        * (KLh[N-1] * cosalfa + q_bot) - S[-1] * dt
            else:  # head boundary
                b[-1] = C[-1] + dt / dz[-1] * (KLh[N-1] / dzu[-1] + KLh[N] / dzl[-1])
                a[-1] = - dt / (dz[-1] * dzu[-1]) * KLh[N-1]
                g[-1] = 0.0
                f[-1] = C[-1] * h_iter[-1] - (W_iter[-1] - W_old[-1]) + dt / dz[-1]\
                        * ((KLh[N-1] - KLh[N]) * cosalfa + KLh[N] / dzl[-1] * h_bot) - S[-1] * dt

            """ solve triagonal matrix """
            
            # save old iteration values
            h_iterold = h_iter.copy()
            W_iterold = W_iter.copy()

            # solve new pressure head and corresponding moisture
            h_iter = thomas(a, b, g, f)
            W_iter = h_to_cellmoist(pF, h_iter, dz)

            # if problems reaching convergence devide time step and retry
            if iterNo == 20 or any(np.isnan(h_iter)):
                if dt / 3.0 > 11.0:
                    dt = max(dt / 3.0, 30.0)
                    logger.debug('%s (iteration %s) More than 20 iterations, retry with dt = %.1f s',
                                 forcing['date'],
                                 iterNo, dt)
                    iterNo = 0
                    h_iter = h_old.copy()
                    W_iter = W_old.copy()
                    continue
                else:  # convergence not reached with dt=30s, break
                    logger.debug('%s (iteration %s) Solution not converging, h and Wtot set to old values.',
                                 forcing['date'],
                                 iterNo)
                    h_iter = h_old.copy()
                    W_iter = W_old.copy()
                    break

            # errors for determining convergence
            err1 = sum(abs(W_iter - W_iterold)*dz)
            err2 = max(abs(h_iter - h_iterold))

        """ iteration loop ends """

        # new state at t
        h = h_iter.copy()
        W = W_iter.copy()

        # calculate q_sur and q_bot in case of head boundaries
        if ubc_flag == 'head':
            q_sur = -KLh[0] * (h_sur - h[0]) / dzu[0] - KLh[0]  # should be calculated inside iteration not to cause mass balance error
        if lbc['type'] == 'head':
            q_bot = -KLh[-1] * (h[-1] - h_bot) / dzl[-1] - KLh[-1] * cosalfa

        """ cumulative fluxes and h_pond """
        if q_sur < 0.0:  # infiltration dominates, evaporation at potential rate
            h_pond = h_pond - (-Prec + Evap - q_sur) * dt
            rr = max(0, h_pond - h_pond_max)  # surface runoff if h_pond > maxpond
            h_pond = h_pond - rr
            C_roff += rr
            del rr
            C_inf += (q_sur - Evap) * dt
            C_eva += Evap * dt
        else:  # evaporation dominates
            C_eva += Evap * dt
            h_pond = max(0, pond_ini - (-Prec + Evap - q_sur) * dt)
            # C_eva += (q_sur + Prec) * dt + pond_ini
            # h_pond = 0.0

        if abs(h_pond) < EPS:
            h_pond = 0.0

        C_dra += -q_bot * dt + sum(q_drain * dz) * dt  # flux through bottom + net lateral flow
        C_trans += sum(q_sink * dz) * dt  # root water uptake

        """ solution time and new timestep """
        t += dt

        dt_old = dt  # save temporarily
        # select new time step based on convergence
        if iterNo <= 3:
            dt = dt * 2
        elif iterNo >= 6:
            dt = dt / 2
        # limit to minimum of 30s
        dt = max(dt, 30)
        # save dto for output to be used in next run of waterflow1D()
        if dt_old == t_final or t_final > t:
            dto = min(dt, t_final)
        # limit by time left to solve
        dt = min(dt, t_final-t)

    """ time stepping loop ends """

    # get ground water depth
    gwl = get_gwl(h, z)

    # mass balance error [m]
    mbe = forcing['potential_infiltration'] * t_final + (
            sum(W_ini*dz) - sum(W*dz)) + (pond_ini - h_pond) - (
            C_dra + C_trans + C_roff + C_eva)

    fluxes = {'infiltration': C_inf / t_final,
              'evaporation': C_eva / t_final,
              'drainage': C_dra / t_final,
              'transpiration': C_trans / t_final,
              'surface_runoff': C_roff / t_final,
              'water_closure': mbe / t_final}

    state = {'water_potential': h,
             'volumetric_water_content': W,
             'ground_water_level': gwl,
             'pond_storage': h_pond}

    return fluxes, state, dto


def waterStorage1D(t_final: float, grid: Dict, forcing: Dict, initial_state: Dict, 
                   pF: Dict, Ksat: np.ndarray, wsto_gwl: Dict,
                   q_sink: np.ndarray, q_drain: np.ndarray=0.0, 
                   lbc: Dict={'type': 'impermeable', 'value': None},
                   h_pond_max: float=0.0, cosalfa: float=1.0) -> Dict:
    r""" 
    Solves soil water storage in 1D column assuming water potential is at hydrostatic equilibrium.

    Args:
        t_final (float): solution timestep [s]
        grid (dict):
            z (array): node elevations, soil surface = 0.0 [m]
            dz (array): thickness of computational layers [m]
            dzu (array): distance to upper node [m]
            dzl (array): distance to lower node [m]
        forcing (dict):
            potential_infiltration (float): [m s-1]
            potential_evaporation (float): [m s-1]
            atmospheric_pressure_head (float): [m]
        initial_state (dict):
            water_potential (array): initial water potential [m]
            pond_storage (array): initial pond storage [m]
        pF (dict): water retention parameters (van Genuchten)
            ThetaS (array): saturated water content [m3 m-3]
            ThetaR (array):residual water content [m3 m-3]
            alpha (array): air entry suction [cm-1]
            n (array): pore size distribution [-]
        Ksat (array): saturated hydraulic conductivity [m s-1]
        wsto_gwl (dict):
            to_gwl (object): interpolated function for gwl --> Wsto
            to_wsto (object): interpolated function for Wsto --> gwl
        q_sink (array): sink term from layers, e.g. root sink [m3 m-3 s-1]
        q_drain (array): sink due to drainage per layer [m3 m-3 s-1]
        lbc (dict):
                type: 'impermeable', 'flux', 'free_drain', 'head'
                value: give for 'head' [m] and 'flux' [m s-1]
        h_pond_max (float): maximum depth allowed ponding at surface [m]
        cosalfa (float): 1 for vertical water flow, 0 for horizontal transport
    
    Returns:
        fluxes (dict):
            infiltration (float): [m s-1]
            evaporation (float): [m s-1]
            drainage (float): [m s-1]
            transpiration (float): [m s-1]
            surface_runoff (float): [m s-1]
            water_closure (float): [m s-1]
        state (dict):
            water_potential (array): [m]
            volumetric_water_content (array): [m3 m-3]
            ground_water_level (float): [m]
            pond_storage (float): [m]
    """

    # time step
    dt = t_final
    
    # forcing
    Prec = forcing['potential_infiltration']
    Evap = forcing['potential_evaporation']
    if 'atmospheric_pressure_head' in forcing:
        h_atm = forcing['atmospheric_pressure_head']
    else:
        h_atm = -1000.0

    # conversion functions
    GwlToWsto = wsto_gwl['to_wsto']
    WstoToGwl = wsto_gwl['to_gwl']

    # net sink/source term
    S = q_sink + q_drain  # root uptake + lateral flow (e.g. by ditches)

    # cumulative boundary fluxes for 0...t_final
    C_inf, C_eva, C_dra, C_trans, C_roff = 0.0, 0.0, 0.0, 0.0, 0.0

    # grid
    z = grid['z']
    dz = grid['dz']
    dzu = grid['dzu']
    dzl = grid['dzl']

    N = len(z)

    # soil hydraulic conductivity and porosity
    if type(Ksat) is float:
        Ksat = np.zeros(N) * Ksat

    # initial state
    h_ini = initial_state['water_potential']
    gwl = get_gwl(h_ini, z)
    Wsto_ini = GwlToWsto(gwl)
    pond_ini = initial_state['pond_storage'] - forcing['pond_recharge'] * t_final

    # hydraulic condictivity, only used at boundaries at i=-1/2 and i=N+1/2
    KLh = hydraulic_conductivity(pF, x=h_ini, Ksat=Ksat)
    # get KLh at i-1/2, note len(KLh) = N + 1
    KLh = spatial_average(KLh, method='arithmetic')

    """ lower boundary condition """
    if lbc['type'] == 'free_drain':
        q_bot = - min(KLh[-1] * cosalfa, 1e-6)  # avoid extreme high drainage
        lbcc_type = 'flux'
    elif lbc['type'] == 'impermeable':
        q_bot = 0.0
        lbcc_type = 'flux'
    elif lbc['type'] == 'flux':
        q_bot = np.sign(lbc['value']) * min(lbc['value'], KLh[-1] * cosalfa)
        lbcc_type = 'flux'
    elif lbc['type'] == 'head':
        h_bot = lbc['value']
        lbcc_type = 'head'
        # approximate flux to calculate Qin
        q_bot = -KLh[-1] * (h_ini[-1] - h_bot) / dzl[-1] - KLh[-1] * cosalfa
    elif lbc['type'] == 'head_oneway':
        h_bot = lbc['value']
        # approximate flux to calculate Qin
        q_bot = -KLh[-1] * (h_ini[-1] - h_bot) / dzl[-1] - KLh[-1] * cosalfa
        if q_bot > 0.0:
            lbcc_type = 'flux'
            q_bot = 0.0
        else:
            lbcc_type = 'head'
    else:
        raise ValueError("Unknown lower boundary condition %s"
             % lbc['type'])

    """ soil column water balance """
    # potential flux at the soil surface (< 0 infiltration)
    q0 = Evap - Prec - pond_ini / dt
    
    # maximum infiltration and evaporation rates
    MaxInf = -Ksat[0]  #max(-KLh[0]*(pond_ini - h0[0] - z[0]) / dzu[0], -Ksat[0])
    MaxEva = -KLh[0]*(h_atm - h_ini[0] - z[0]) / dzu[0]
    
    # limit flux at the soil surface: MaxInf < q_sur < MaxEvap
    q_sur = min(max(MaxInf, q0), MaxEva)
    #print 'q_sur = ' + str(q_sur) + ' MaxInf = ' + str(MaxInf) + ' MaxEvap = ' + str(MaxEva) + ' KLh = ' + str(KLh[0])

    # net flow to soil profile during dt
    Qin = (q_bot - sum(S * dz) - q_sur) * dt
    # airvolume available in soil profile after previous time step
    Airvol = max(0.0, GwlToWsto(0.0) - Wsto_ini)

    if Qin >= Airvol:  # net inflow does not fit into profile
        Wsto = Wsto_ini + Airvol
        q_sur = - Airvol / dt + q_bot - sum(S * dz)
    else:
        Wsto = Wsto_ini + Qin

    """ new state and cumulative fluxes """
    # ground water depth corresponding to Wsto
    gwl = WstoToGwl(Wsto)
    # water potential and volumetric water content
    h = gwl - z
    W = h_to_cellmoist(pF, h, dz)

    # infitration, evaporation, surface runoff and h_pond
    if q_sur <= EPS:  # infiltration dominates, evaporation at potential rate
        h_pond = pond_ini - (-Prec + Evap - q_sur) * dt
        rr = max(0, h_pond - h_pond_max)  # surface runoff if h_pond > maxpond
        h_pond -= rr
        C_roff += rr
        del rr
        C_inf += (q_sur - Evap) * dt
        C_eva += Evap * dt
    else:  # evaporation dominates
        C_eva += Evap * dt
        h_pond = max(0, pond_ini - (-Prec + Evap - q_sur) * dt)
        # C_eva += (q_sur + Prec) * dt + pond_ini
        # h_pond = 0.0

    if abs(h_pond) < EPS:  # eliminate h_pond caused by numerical innaccuracy (?)
        h_pond = 0.0

    C_dra += -q_bot * dt + sum(q_drain * dz) * dt  # flux through bottom + net lateral flow
    C_trans += sum(q_sink * dz) * dt  # root water uptake

    # mass balance error [m]
    mbe = Prec * t_final + (Wsto_ini - Wsto) + (pond_ini - h_pond)\
            - C_dra - C_trans - C_roff - C_eva

    fluxes = {'infiltration': C_inf / t_final,
              'evaporation': C_eva / t_final,
              'drainage': C_dra / t_final,
              'transpiration': C_trans / t_final,
              'surface_runoff': C_roff / t_final,
              'water_closure': mbe / t_final}

    state = {'water_potential': h,
             'volumetric_water_content': W,
             'ground_water_level': gwl,
             'pond_storage': h_pond}

    return fluxes, state

""" drainage equations """

def drainage_hooghoud(dz, Ksat, gwl, DitchDepth, DitchSpacing, DitchWidth, Zbot=None,
                      below_ditch_drain=False) -> np.ndarray:
    r""" Calculates drainage to open ditch or tile-drain using Hooghoud's drainage equation.
    Accounts for drainage from saturated layers above and below ditch bottom.

    Reference:
       Follows Koivusalo, Lauren et al. FEMMA -document. Ref: El-Sadek et al., 2001. 
       J. Irrig.& Drainage Engineering.
    Args:
       dz (array):  soil conpartment thichness, node in center [m]
       Ksat (array): horizontal saturated hydr. cond. [m s-1]
       gwl (float): ground water level below surface, <0 [m]
       DitchDepth (float): depth of drainage ditch bottom, >0 [m]
       DitchSpacing (float): horizontal spacing of drainage ditches [m]
       DitchWidth (float): ditch bottom width [m]
       Zbot (float): distance to impermeable layer, >0 [m]
    Returns:
       Qz_drain (array): drainage from each soil layer [m3 m-3 s-1]

    """
    z = dz / 2 - np.cumsum(dz)
    N = len(z)
    Qz_drain = np.zeros(N)
    Qa = 0.0
    Qb = 0.0

    if Zbot is None or Zbot > sum(dz):  # Can't be lower than soil profile bottom
        Zbot = sum(dz)

    Hdr = min(max(0, gwl + DitchDepth), DitchDepth)  # depth of saturated layer above ditch bottom

    if Hdr > 0:
        # saturated layer thickness [m]
        dz_sat = np.minimum(np.maximum(gwl - (z - dz / 2), 0), dz)

        """ drainage from saturated layers above ditch base """
        # layers above ditch bottom where drainage is possible
        ix = np.intersect1d(np.where((z - dz / 2)- gwl < 0), np.where(z + dz / 2 > -DitchDepth))

        if ix.size > 0:
            dz_sat[ix[-1]] = dz_sat[ix[-1]] + (z[ix][-1] - dz[ix][-1] / 2 + DitchDepth)
            # transmissivity of layers  [m2 s-1]
            Trans = Ksat * dz_sat
            Ka = sum(Trans[ix]) / sum(dz_sat[ix])  # effective hydraulic conductivity ms-1
            Qa = 4 * Ka * Hdr**2 / (DitchSpacing**2)  # m s-1, total drainage above ditches
            # sink term s-1, partitions Qa by relative transmissivity of layer
            Qz_drain[ix] = Qa * Trans[ix] / sum(Trans[ix]) / dz[ix]

        if below_ditch_drain:
            """ drainage from saturated layers below ditch base """
            # layers below ditch bottom where drainage is possible
            ix = np.where(z - dz / 2 <= -DitchDepth)
            dz_sat = np.where(dz_sat < np.minimum(np.maximum(gwl - (z - dz / 2), 0), dz),
                              np.minimum(np.maximum(gwl - (z - dz / 2), 0), dz) - dz_sat,
                              dz_sat)
            # transmissivity of layers  [m2 s-1]
            Trans = Ksat * dz_sat
            # effective hydraulic conductivity ms-1
            Kb = sum(Trans[ix]) / sum(dz_sat[ix])

            # compute equivalent depth Deq
            Dbt = Zbot - DitchDepth  # distance from impermeable layer to ditch bottom
            A = 3.55 - 1.6 * Dbt / DitchSpacing + 2 * (2.0 / DitchSpacing)**2.0
            Reff = DitchWidth / 2.0  # effective radius of ditch

            if Dbt / DitchSpacing <= 0.3:
                Deq = Dbt / (1.0 + Dbt / DitchSpacing * (8 / np.pi * np.log(Dbt / Reff) - A))  # m
            else:
                Deq = np.pi * DitchSpacing / (8 * (np.log(DitchSpacing / Reff) - 1.15))  # m

            Qb = 8 * Kb * Deq * Hdr / DitchSpacing**2  # m s-1, total drainage below ditches
            Qz_drain[ix] = Qb * Trans[ix] / sum(Trans[ix]) / dz[ix]  # sink term s-1

    return Qz_drain

""" utility functions: vanGenuchten water retention and concuctivity models """

def theta_psi(pF: Dict, x: float):
    r"""
    Converts vol. moisture content to water potential
    
    Args:
        pF (dict) - water retention parameters
        x (float) - vol. water content [m3 m-3]
    Returns:
        water potential [m]
    """
    
    ts = pF['ThetaS']
    tr = pF['ThetaR']
    a = pF['alpha']
    n = pF['n']
    m = 1.0 - np.divide(1.0, n)
    
    x = np.minimum(x, ts)
    x = np.maximum(x, tr)  # checks limits
    s = (ts - tr) / ((x - tr) + EPS)
    Psi = -1e-2 / a * (s**(1.0 / m) - 1.0)**(1.0 / n)  # m

    return Psi

def psi_theta(pF: Dict, x: float):
    r"""
    Converts water potential to vol. moisture content. h_to_cellmoist does the same but
    accounts for partially saturated gridcells.

    Args:
        pF (dict) - water retention parameters
        x (float) - water potential [m]
    Returns:
        vol. water content [m3 m-3]
    """
    # converts water potential (m) to water content (m3m-3)
    x = 100 * np.minimum(x, 0)  # cm
    ts = pF['ThetaS']
    tr = pF['ThetaR']
    a = pF['alpha']
    n = pF['n']
    m = 1.0 - np.divide(1.0, n)
    
    Th = tr + (ts - tr) / (1.0 + abs(a * x)**n)**m
    return Th

def h_to_cellmoist(pF: Dict, h: np.ndarray, dz: np.ndarray) -> np.ndarray:
    r""" 
    Converts water potential to volumetric moisture content based on vanGenuchten-Mualem model.
    Partly saturated cells calculated as thickness-weigthed average of saturated and unsaturated parts.

    Args:
        pF (dict):
            ThetaS (array): saturated water content [m3 m-3]
            ThetaR (array): residual water content [m3 m-3]
            alpha (array): air entry suction [cm-1]
            n (array): pore size distribution [-]
        h (array): pressure head [m]
        dz (array): soil conpartment thichness, node in center [m]
    Returns:
        theta (array): volumetric water content of cell  [m3 m-3]

    """
    # water retention parameters
    Ts = np.array(pF['ThetaS'])
    Tr = np.array(pF['ThetaR'])
    alfa = np.array(pF['alpha'])
    n = np.array(pF['n'])
    m = 1.0 - np.divide(1.0, n)

    # moisture based on cell-center head
    x = np.minimum(h, 0)
    theta = Tr + (Ts - Tr) / (1 + abs(alfa * 100 * x)**n)**m

    # correct moisture of partly saturated cells
    ix = np.where(abs(h) < dz/2)
    # moisture of unsaturated part
    x[ix] = -(dz[ix]/2 - h[ix]) / 2
    theta[ix] = Tr[ix] + (Ts[ix] - Tr[ix]) / (1 + abs(alfa[ix] * 100 * x[ix])**n[ix])**m[ix]
    # total moisture as weighted average
    theta[ix] = (theta[ix] * (dz[ix]/2 - h[ix]) + Ts[ix] * (dz[ix]/2 + h[ix])) / (dz[ix])

    return theta

def diff_wcapa(pF: Dict, h: np.ndarray, dz: np.ndarray) -> np.ndarray:
    r""" Differential water capacity (dW/dh) calculated numerically.

    Args:
        pF (dict):
            ThetaS (array): saturated water content [m3 m-3]
            ThetaR (array): residual water content [m3 m-3]
            alpha' (array): air entry suction [cm-1]
            n (array): pore size distribution [-]
        h (array): pressure head [m]
        dz (array): soil conpartment thichness, node in center [m]
    Returns:
        dwcapa (array): differential water capacity dTheta/dhead [m-1]

    """

    dh = 1e-5
    theta_plus = h_to_cellmoist(pF, h + dh, dz)
    theta_minus = h_to_cellmoist(pF, h - dh, dz)

    # differential water capacity
    dwcapa = (theta_plus - theta_minus) / (2 * dh)

    return dwcapa

def hydraulic_conductivity(pF: Dict, x: np.ndarray, Ksat: np.ndarray=1) -> np.ndarray:
    r""" 
    Unsaturated hydraulic conductivity following vanGenuchten-Mualem -model.

    Args:
        pF (dict):
            ThetaS (float|array): saturated water content [m3 m-3]
            ThetaR (float|array): residual water content [m3 m-3]
            alpha (float|array): air entry suction [cm-1]
            n (float|array): pore size distribution [-]
        h (float|array): pressure head [m]
        Ksat (float|array): saturated hydraulic conductivity [units]
    Returns:
        Kh (float|array): hydraulic conductivity (if Ksat ~=1 then in [units], else relative [-])

    """

    x = np.array(x)

    # water retention parameters
    alfa = np.array(pF['alpha'])
    n = np.array(pF['n'])
    m = 1.0 - np.divide(1.0, n)

    def relcond(x):
        Seff = 1.0 / (1.0 + abs(alfa*x)**n)**m
        r = Seff**0.5 * (1.0 - (1.0 - Seff**(1/m))**m)**2.0
        return r

    Kh = Ksat * relcond(100.0 * np.minimum(x, 0.0))

    return Kh

def gwl_Wsto(dz: np.ndarray, pF: Dict) -> Dict:
    r""" 
    Relationship (interpolation functions) between soil column ground water depth and
    column water storage [m] and vice versa.

    Args:
        dz (array): soil compartment thichness, node in center [m]
        pF (dict):
            ThetaS (float|array): saturated water content [m3 m-3]
            ThetaR (float|array): residual water content [m3 m-3]
            alpha (float|array): air entry suction [cm-1]
            n (float|array): pore size distribution [-]
    Returns (dict):
            to_gwl: interpolated function to convert gwl --> Wsto
            to_wsto: interpolated function to convert Wsto --> gwl

    """

    z = dz / 2 - np.cumsum(dz)

    # --------- connection between gwl and water storage------------
    # gwl from ground surface gwl = 0 to gwl = -5
    gwl = np.arange(0.0, -5, -1e-3)

    # solve water storage corresponding to gwls
    Wsto = [sum(h_to_cellmoist(pF, g - z, dz) * dz) for g in gwl]

    # interpolate functions
    WstoToGwl = interp1d(np.array(Wsto), np.array(gwl), fill_value='extrapolate')
    GwlToWsto = interp1d(np.array(gwl), np.array(Wsto), fill_value='extrapolate')
#    plt.figure(1)
#    plt.plot(WstoToGwl(Wsto), Wsto)

    del gwl, Wsto

    return {'to_gwl': WstoToGwl, 'to_wsto': GwlToWsto}

def get_gwl(head, x):
    r""" Finds ground water level based on pressure head.

    Args:
        head (array): heads in nodes [m]
        x (array): grid, < 0, monotonically decreasing [m]
    Returns:
        gwl (float): ground water level in column [m]
    """
    # indices of unsaturatd nodes
    sid = np.where(head <= 0)[0]

    if len(sid) < len(head):
        # gwl above profile bottom
        if len(sid) > 0:  # gwl below first node
            # finding head from bottom to top to avoid returning perched gwl
            gwl = x[sid[-1]+1] + head[sid[-1]+1]
        else:  # gwl in or above first node
            gwl = x[0] + head[0]
    else:
        # gwl not in profile, assume hydr. equilibrium between last node and gwl
        gwl = head[-1] + x[-1]

    return gwl

def rew(theta: np.ndarray, theta_wp: np.ndarray, theta_fc: np.ndarray) -> np.ndarray:
    r"""
    Relative plant available water
    Rew = min[1.0, (theta – theta_wp)/(theta_fc – theta_wp)

    Args:
        theta (array|float): vol. water content [m3 m-3]
        theta_wp (array|float): vol. water content at wilting point [m3 m-3]
        theta_fc (array|float): vol. water content at field capacity [m3 m-3]
    Returns (array):
        Rew(array|float): relative plant available water [-]
    """

    Rew = np.minimum(1.0, (theta - theta_wp)/(theta_fc - theta_wp))

    return Rew


""" NOT IN USE IN pyAPES_MLM"""

def wrc(pF: Dict, theta: np.ndarray=None, psi: np.ndarray=None, draw_pF: bool=False) -> np.ndarray:
    """
    vanGenuchten-Mualem soil water retention model 

    References:
        Schaap and van Genuchten (2005). Vadose Zone 5:27-34
        van Genuchten, (1980). Soil Science Society of America Journal 44:892-898

    Args:
        pF (dict):
            ThetaS (float|array): saturated water content [m3 m-3]
            ThetaR (float|array): residual water content [m3 m-3]
            alpha (float|array): air entry suction [cm-1]
            n (float|array): pore size distribution [-]
        theta (float|array): vol. water content [m3 m-3]
        psi (float|array): water potential [m]
        draw_pF (bool): Draw pF-curve.
    Returns:
        y (float|array): water potential [m] or vol. water content [m3 m-3]. Returns None if only curve is drawn.

    """

    Ts = np.array(pF['ThetaS'])
    Tr = np.array(pF['ThetaR'])
    alfa = np.array(pF['alpha'])
    n = np.array(pF['n'])
    m = 1.0 - np.divide(1.0, n)

    def theta_psi(x):
        # converts water content [m3 m-3] to potential [m]]
        x = np.minimum(x, Ts)
        x = np.maximum(x, Tr)  # checks limits
        s = (Ts - Tr) / ((x - Tr) + EPS)
        Psi = -1e-2 / alfa*(s**(1.0 / m) - 1.0)**(1.0 / n)  # m
        Psi[np.isnan(Psi)] = 0.0
        return Psi

    def psi_theta(x):
        # converts water potential [m] to water content [m3 m-3]
        x = 100*np.minimum(x, 0)  # cm
        Th = Tr + (Ts - Tr) / (1 + abs(alfa*x)**n)**m
        return Th

    # --- convert between theta <-- --> psi
    if theta:
        y = theta_psi(theta)  # 'Theta-->Psi'
    elif psi:
        y = psi_theta(psi)  # 'Psi-->Theta'

    # draws pf-curve
    if draw_pF:
        Ts = Ts[0]; Tr = Tr[0]; alpha = alpha[0]; n = n[0]  
        xx = -np.logspace(-4, 5, 100)  # cm
        yy = psi_theta(xx)

        #  field capacity and wilting point
        fc = psi_theta(-1.0)
        wp = psi_theta(-150.0)

        fig = plt.figure(99)
        fig.suptitle('vanGenuchten-Mualem WRC', fontsize=16)
        ttext = r'$\theta_s=$' + str(Ts) + r', $\theta_r=$' + str(Tr) +\
                r', $\alpha=$' + str(alfa) + ',n=' + str(n)

        plt.title(ttext, fontsize=14)
        plt.semilogx(-xx, yy, 'g-')
        plt.semilogx(1, fc, 'ro', 150, wp, 'ro')  # fc, wp
        plt.text(1, 1.1*fc, 'FC'), plt.text(150, 1.2*wp, 'WP')
        plt.ylabel(r'$\theta$  $(m^3m^{-3})$', fontsize=14)
        plt.xlabel('$\psi$ $(m)$', fontsize=14)
        plt.ylim(0.8*Tr, min(1, 1.1*Ts))

        del xx, yy
        y = None
    
    return y

# EOF