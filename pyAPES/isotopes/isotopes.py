# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 15:06:01 2024

@author: Kersti Leppa
"""

import numpy as np

from pyAPES.utils.constants import EPS, DEG_TO_KELVIN, GAS_CONSTANT

# VPDB standard for 13C/12C ratio
D13C_STD = 0.01123720
# VSMOW standard for 18O/16O ratio
D18O_STD = 2.0052e-3
# H2O molar density [mol m-3]
H2O_MOLARDENSITY = 55.5e3
# H2 18O diffusivity [m2 s-1]
H2_18O_DIFFYSIVITY = 2.66e-9

TN = 25.0 + DEG_TO_KELVIN  # reference temperature [K]

class Isotopes(object):

    def __init__(self, p: dict, pt_masks: list):
        r""" 
        Initializes isotope model.
        
        Args:
            p (dict):
                'd13C' (dict):
                    'solve' (bool): True/False for solving d13C prosesses
                    'a_b' (float): fractionation during diffusion through the boundary layer [-]
                    'a_s' (float): fractionation during diffusion through the stomata [-]
                    'a_m' (float): fractionation during transfer through mesophyll [-]
                    'b' (float): fractionation by Rubisco [-]
                    'e' (float): fractionation during mitochondrial respiration [-]
                    'f' (float): fractionation during photorespiration [-]
                    'init_d13C_leaf' (float): initial leaf sugar store d13c [permil]
                    },
                'd18O': {
                    'solve' (bool): True/False for solving d18O processes
                    'e_kb' (float): fractionation during diffusion of water vapor through boundary layer [-]
                    'e_k' (float):  fractionation during diffusion of water vapor through stomata [-]
                    'e_wc' (dict): biochemical fractionation [-]
                        'constant' (bool): True/False
                        'value' (float): value if constant otherwise temperature dependent [-]
                    'nonsteadystate' (bool): if true applies non-steady state for leaf water modeling, else steady-state approach
                    'Vlw' (float): leaf mesophyll water volume [mol m-2]  ! one-sided leaf area
                    'peclet' (bool): if true applies peclet model (L_eff), else two-pool model (f1)
                    'L_eff' (float): leaf mesophyll effective mixing length [m]  ! one-sided leaf area
                    'f1' (float): parameter of two-pool model [-], if f1=1 simplifies to Craig-Gordon model
                    'init_d18O_leaf' (float): initial leaf sugar store d18O [permil].
                    },
                'C_leaf' (float): size of leaf sugar store [umol C m-2] ! one-sided leaf area
            pt_masks (list[np.ndarray]): list of arrays for each planttype with 1.0 where lad > 0, nan elsewhere
        
        Returns:
            self (object)
        """

        # switches
        self.solve_d13C = p['d13C']['solve']
        self.solve_d18O = p['d18O']['solve']
        
        # parameters
        self.d13C_para = p['d13C']
        self.d18O_para = p['d18O']
        self.C_leaf = p['C_leaf']
        
        # number of planttypes
        self.n_pt = len(pt_masks)

        # initial conditions (dict of list of arrays for each planttype)
        self.d13C_leaf_sugar = {'sunlit':[mask * p['d13C']['init_d13C_leaf'] for mask in pt_masks],
                                'shaded':[mask * p['d13C']['init_d13C_leaf'] for mask in pt_masks]}
        self.d18O_leaf_sugar = {'sunlit':[mask * p['d18O']['init_d18O_leaf'] for mask in pt_masks],
                                'shaded':[mask * p['d18O']['init_d18O_leaf'] for mask in pt_masks]}
        self.d18O_leaf_water = {'sunlit':[mask * np.nan for mask in pt_masks],
                                'shaded':[mask * np.nan for mask in pt_masks]}  # initialized with steady state result at first time step

    def run(self, dt: float, forcing: dict) -> dict:
        r"""
        Runs isotope model for one timestep.
        (list[np.ndarray] type inputs/outputs contain canopy profiles for each planttype)
        
        Args:
            dt (float): time step [s]
            forcing (dict):
                'leaf_temperature_sunlit' (list[np.ndarray]): leaf temperature [degC].
                'leaf_temperature_shaded' (list[np.ndarray]): leaf temperature [degC].
                'net_co2_sunlit' (list[np.ndarray]): net phosynthesis [umol m-2 s-1].
                'net_co2_shaded' (list[np.ndarray]): net phosynthesis [umol m-2 s-1].
                'dark_respiration_sunlit' (list[np.ndarray]): mitochondrial respiration [umol m-2 s-1].
                'dark_respiration_shaded' (list[np.ndarray]): mitochondrial respiration [umol m-2 s-1].
                'transpiration_sunlit' (list[np.ndarray]): transpiraiton [mol m-2 s-1].
                'transpiration_shaded' (list[np.ndarray]): transpiraiton [mol m-2 s-1].
                'stomatal_conductance_h2o_sunlit' (list[np.ndarray]): stomatal conductance for water vapor [mol m-2 s-1].
                'stomatal_conductance_h2o_shaded' (list[np.ndarray]): stomatal conductance for water vapor [mol m-2 s-1].
                'boundary_conductance_h2o_sunlit' (list[np.ndarray]): boundary-layer conductance for water vapor [mol m-2 s-1].
                'boundary_conductance_h2o_shaded' (list[np.ndarray]): boundary-layer conductance for water vapor [mol m-2 s-1].
                'leaf_internal_co2_sunlit' (list[np.ndarray]): CO2 mole fraction in the intercellular spaces [ppm].
                'leaf_internal_co2_shaded' (list[np.ndarray]): CO2 mole fraction in the intercellular spaces [ppm].
                'leaf_surface_co2_sunlit' (list[np.ndarray]): CO2 mole fraction at leaf surface [ppm].
                'leaf_surface_co2_shaded' (list[np.ndarray]): CO2 mole fraction at leaf surface [ppm].
                'co2' (np.ndarray): CO2 mole fraction in ambient air [ppm].
                'h2o' (np.ndarray): mole fraction of water vapor in atmosphere [mol mol-1].
                'air_pressure' (float): atmospheric pressure [Pa].
                'd13Ca' (float): carbon isotope composition in CO2 of ambient air [permil].
                'd18Ov' (float): d18O of water vapor [permil].
                'd18O_sw' (float): d18O of source water [permil].

        Returns:
            results (dict): 
                '13c_discrimination_sunlit' (list[np.ndarray]): discrimination of net photosynthesis [permil].
                '13c_discrimination_shaded' (list[np.ndarray]): discrimination of net photosynthesis [permil].
                'd13c_net_co2_flux_sunlit' (list[np.ndarray]): d13C of net CO2 flux [permil].
                'd13c_net_co2_flux_shaded' (list[np.ndarray]): d13C of net CO2 flux [permil].
                'd13c_leaf_sugar_sunlit' (list[np.ndarray]): d13C of needle sugar pool [permil].
                'd13c_leaf_sugar_shaded' (list[np.ndarray]): d13C of needle sugar pool [permil].
                'd18o_evaporative_sites_sunlit' (list[np.ndarray]): d18o at evaporative sites [permil].
                'd18o_evaporative_sites' (list[np.ndarray]): d18o at evaporative sites [permil].
                'd18o_leaf_water_sunlit' (list[np.ndarray]): d18o of leaf water [permil].
                'd18o_leaf_water_shaded' (list[np.ndarray]): d18o of leaf water [permil].
                'd18o_leaf_sugar_sunlit' (list[np.ndarray]): d18o of needle sugar pool [permil].
                'd18o_leaf_sugar_shaded' (list[np.ndarray]): d18o of needle sugar pool [permil].
        """
        pt_results = [{} for i in range(self.n_pt)]  # list of n_pt dicts
        
        # d13C
        if self.solve_d13C:
            for n in range(self.n_pt):
                # at leaf-level
                for sl_sh in ['sunlit','shaded']:
                    Tk = forcing['leaf_temperature_'+sl_sh][n] + DEG_TO_KELVIN
                    tau_c = 42.75 * np.exp(37830*(Tk - TN) / (TN * GAS_CONSTANT * Tk))
                    pt_res = d13C_leaf(dt=dt,
                                       ca=forcing['co2'],
                                       cs=forcing['leaf_surface_co2_'+sl_sh][n],
                                       ci=forcing['leaf_internal_co2_'+sl_sh][n], 
                                       cc=forcing['leaf_internal_co2_'+sl_sh][n],  # for now neglects mesophyll resistance
                                       An=forcing['net_co2_'+sl_sh][n],
                                       Rd=forcing['dark_respiration_'+sl_sh][n],
                                       tau_c=tau_c,
                                       d13ca=forcing['d13Ca'],
                                       a_b=self.d13C_para['a_b'], a_s=self.d13C_para['a_s'], 
                                       a_m=self.d13C_para['a_m'], b=self.d13C_para['b'], 
                                       e=self.d13C_para['e'], f=self.d13C_para['f'], 
                                       sugar_sto=self.C_leaf,
                                       init_d13c=self.d13C_leaf_sugar[sl_sh][n])
                    # update state
                    self.d13C_leaf_sugar[sl_sh][n] = pt_res['d13c_leaf_sugar']
                    # add to results and add sunlit/shaded to dict keys
                    pt_results[n].update({k+'_'+sl_sh: v for k, v in pt_res.items()})
                # at plant-level
                #
                #
        
        # d18O
        if self.solve_d18O:
            for n in range(self.n_pt):
                # at leaf-level
                for sl_sh in ['sunlit','shaded']:
                    pt_res = d18O_leaf(dt=dt,
                                       E=forcing['transpiration_'+sl_sh][n],
                                       An=forcing['net_co2_'+sl_sh][n], 
                                       Rd=forcing['dark_respiration_'+sl_sh][n], 
                                       w_a=forcing['h2o'],
                                       gb=forcing['boundary_conductance_h2o_'+sl_sh][n],
                                       gs=forcing['stomatal_conductance_h2o_'+sl_sh][n],
                                       d18o_xylem=forcing['d18O_sw'],
                                       d18o_vapor=forcing['d18Ov'],
                                       T=forcing['leaf_temperature_'+sl_sh][n],
                                       P=forcing['air_pressure'],
                                       e_kb=self.d18O_para['e_kb'], 
                                       e_k=self.d18O_para['e_k'], 
                                       peclet=self.d18O_para['peclet'], 
                                       L_eff=self.d18O_para['L_eff'], 
                                       e_wc=self.d18O_para['e_wc'], 
                                       nonsteadystate=self.d18O_para['nonsteadystate'], 
                                       Vlw=self.d18O_para['Vlw'], 
                                       f1=self.d18O_para['f1'], 
                                       sugar_sto=self.C_leaf, 
                                       init_d18Olw=self.d18O_leaf_water[sl_sh][n],
                                       init_d18Ols=self.d18O_leaf_sugar[sl_sh][n])
                    # update state
                    self.d18O_leaf_water[sl_sh][n] = pt_res['d18o_leaf_water']
                    self.d18O_leaf_sugar[sl_sh][n] = pt_res['d18o_leaf_sugar']
                    # add to results and add sunlit/shaded to dict keys
                    pt_results[n].update({k+'_'+sl_sh: v for k, v in pt_res.items()})
                # at plant-level
                #
                #

        # convert list of dicts to dict of lists and append
        results = {}
        for k, v in pt_results[0].items():
            results[k] = [x[k] for x in pt_results]

        return results

def d13C_leaf(dt: float, ca: np.ndarray, cs: np.ndarray, ci: np.ndarray, cc: np.ndarray, 
              An: np.ndarray, Rd: np.ndarray, tau_c: np.ndarray, d13ca: float,
              init_d13c: np.ndarray, a_b: float=2.9e-3, a_s: float=4.4e-3, a_m: float=1.8e-3, 
              b: float=29.0e-3, e: float=-6.0e-3, f: float=8.0e-3, sugar_sto: float=4e5) -> dict:
    """
    Solves on timestep of net photosynthetic discrimination following Wingate et al. (2007)
    and the accumulation of the d13C signal in the leaf sugar pool. See also Leppä et al (2022).

    Args:
        dt (float): timestep [s].
        ca (np.ndarray): CO2 mole fraction in ambient air [ppm].
        cs (np.ndarray): CO2 mole fraction at leaf surface [ppm].
        ci (np.ndarray): CO2 mole fraction in the intercellular spaces [ppm].
        cc (np.ndarray): CO2 mole fraction in the chloroplast [ppm].
        An (np.ndarray): net phosynthesis [umol m-2 s-1].
        Rd (np.ndarray): mitochondrial respiration [umol m-2 s-1].
        tau_c (np.ndarray): CO2 compensation point [ppm].
        d13ca (float): carbon isotope composition in CO2 of ambient air [permil].
        init_d13c (np.ndarray): initial sugar store d13c [permil].
        a_b (float, optional): fractionation during diffusion through the
                boundary layer [-]. Defaults to 2.8e-3.
        a_s (float, optional): fractionation during diffusion through the
                stomata [-]. Defaults to 4.4e-3.
        a_m (float, optional): fractionation during transfer through mesophyll [-].
                Defaults to 1.8e-3.
        b (float, optional): fractionation by Rubisco [-]. Defaults to 29.0e-3.
        e (float, optional): fractionation during mitochondrial respiration [-].
                Defaults to -6.0e-3.
        f (float, optional): fractionation during photorespiration [-]. Defaults
                to 8.0e-3.
        sugar_sto (float, optional): size of sugar store [umol C m-2].
                Defaults to 4e5.

    Returns:
        dict: 
            '13c_discrimination' (np.ndarray): discrimination of net photosynthesis [permil].
            'd13c_net_co2_flux' (np.ndarray): d13C of net CO2 flux [permil].
            'd13c_leaf_sugar' (np.ndarray): d13C of needle sugar pool [permil].
    """
    # carboxylation efﬁciency
    k = (An + Rd)/(cc - tau_c)
    
    R_ca = (d13ca/1000 + 1) * D13C_STD
    
    # initial state
    R_sugar_init = (init_d13c/1000 + 1) * D13C_STD
    
    # discrimination of net photosynthesis.
    Delta = 1 / (k * ca - Rd) * (
        (a_b * (ca - cs) / ca + a_s * (cs - ci) / ca + a_m * (ci - cc) / ca
        + b * cc / ca - f * tau_c / ca) * (k * ca)
        + (1 - R_ca / R_sugar_init * (1 + e)) * Rd)

    # weird values when An + Rd > 0 and An < 0
    Delta = np.minimum(np.maximum(Delta, -0.5), 0.5)

    # sugar discharge (assuming constant sugar_sto)
    discharge = An
    
    # sugar isotope ratio (implicit)
    R_sugar = ((sugar_sto * R_sugar_init + An * dt * R_ca / (1 + Delta)) / (
                sugar_sto + discharge * dt))
    
    # in permil
    d13c_sugar = (R_sugar / D13C_STD - 1) * 1000
    d13c_flux = (R_ca / (1 + Delta)/ D13C_STD - 1) * 1000

    return {'13c_discrimination': 1000 * Delta,
            'd13c_net_co2_flux': d13c_flux,
            'd13c_leaf_sugar': d13c_sugar}

def d18O_leaf(dt: float, E: np.ndarray, An: np.ndarray, Rd: np.ndarray, w_a: np.ndarray, 
              gb: np.ndarray, gs: np.ndarray, d18o_xylem: float, d18o_vapor: float, 
              T: np.ndarray, P: np.ndarray, init_d18Olw: np.ndarray, init_d18Ols: np.ndarray,
              e_kb: float=19e-3, e_k: float=28e-3, peclet: bool=True, L_eff: float=30e-3, 
              e_wc: dict={'constant':True, 'value':27e-3}, nonsteadystate: bool=False, 
              Vlw: float=10, f1: float=1.0, sugar_sto: float=4e5) -> dict:
    """
    Solves one timestep of oxygen fractionation in leaf water, new assimilates and the
    accumulation of the d18O signal in the needle sugar and WSC pool. See Leppä et al (2022).

    Args:
        dt (float): DESCRIPTION.
        E (np.ndarray): transpiraiton [mol m-2 s-1].
        An (np.ndarray): net phosynthesis [umol m-2 s-1].
        Rd (np.ndarray): mitochondrial respiration [umol m-2 s-1].
        w_a (np.ndarray): mole fraction of water vapor in atmosphere [mol mol-1].
        gb (np.ndarray): boundary-layer conductance for water vapor [mol m-2 s-1].
        gs (np.ndarray): stomatal conductance for water vapor [mol m-2 s-1].
        d18o_xylem (float): d18O of source water [permil].
        d18o_vapor (float): d18O of water vapor [permil].
        T (np.ndarray): leaf temperature [degC].
        P (np.ndarray): atmospheric pressure [Pa].
        init_d18Olw (np.ndarray): initial leaf water d18O [permil].
        init_d18Ols (np.ndarray): initial sugar store d18O [permil].
        e_kb (float, optional): fractionation during diffusion of water vapor
                through boundary layer. Defaults to 19e-3.
        e_k (float, optional): fractionation during diffusion of water vapor
                through stomata. Defaults to 28e-3.
        peclet (boolean, optional): if true applies peclet model, else
                two-pool model. Defaults to True.
        L_eff (float, optional): leaf mesophyll effective mixing length (m).
                Defaults to 30e-3.
        e_wc (dict, optional): biochemical fractionation factor (constant or 
                temperature dependent). Defaults to {'constant':True, 'value':27e-3}.
        nonsteadystate (bool, optional): if true applies non-steady state
                for leaf water modeling, else steady-state approach. Defaults
                to False.
        Vlw (float, optional): leaf mesophyll water volume [mol m-2]. Defaults to 10.
        f1 (float, optional): parameter of two-pool model [-]. Defaults to 1.0.
        sugar_sto (float, optional): size of sugar store [umol C m-2].
                Defaults to 4e5.

    Returns:
        dict: 
            'd18o_evaporative_sites' (np.ndarray): d18o at evaporative sites [permil].
            'd18o_leaf_water' (np.ndarray): d18o of leaf water [permil].
            'd18o_leaf_sugar' (np.ndarray): d18o of needle sugar pool [permil].
    """
    
    esat = 611.0 * np.exp((17.502 * T) / (T + 240.97))  # Pa
    w_i = esat / P

    e_star = np.exp(1137/(T + DEG_TO_KELVIN)**2 - 0.4156 /(T + DEG_TO_KELVIN) - 0.0020667) - 1

    e_kkb = (e_k * gb + e_kb * gs) / (gb + gs)
    
    # temperature-dependent biochemical fractionation factor (sternberg et al. 2011)
    e_wc = 1e-3*(0.0084*T**2 - 0.51*T + 33.172)

    # isotopic ratios in xylem and ambient vapor
    R_x = (d18o_xylem/1000 + 1) * D18O_STD
    R_v = (d18o_vapor/1000 + 1) * D18O_STD

    # isotopic ratio at evaporative sites
    R_e = (1 + e_star)*((1+e_kkb)*R_x*(1 - w_a / w_i) + R_v *(w_a / w_i))

    # peclet effect
    if peclet:
        p = np.maximum(EPS, L_eff * E / (H2O_MOLARDENSITY * H2_18O_DIFFYSIVITY))
        f1 = (1 - np.exp(-p)) / p

    # steady state 
    R_lw_ss = (R_e - R_x) * f1 + R_x

    if nonsteadystate:
        # initial state
        if np.isnan(init_d18Olw).all():  # if no initial state use steady state
            R_lw_init = R_lw_ss
        else:
            R_lw_init = (init_d18Olw/1000 + 1) * D18O_STD

        a = dt / Vlw * E * w_i / (w_i - w_a) / ((1 + e_star)*(1+e_kkb)) * 1 / f1
        R_lw = (a * R_lw_ss + R_lw_init) / (1 + a)
        R_e = (R_lw - R_x) / f1 + R_x
    else:
        R_lw = R_lw_ss

    # initial state
    R_sugar_init = (init_d18Ols/1000 + 1) * D18O_STD

    # isotope ratio of assimilates
    R_CO2 = (1 + e_wc) * R_lw

    # sugar discharge
    discharge = An

    # sugar isotope ratio (implicit)
    R_sugar = ((sugar_sto * R_sugar_init + (An + Rd) * dt * R_CO2) / (
        sugar_sto + (Rd + discharge) * dt))

    # in permil
    d18o_e = (R_e / D18O_STD - 1) * 1000
    d18o_lw = (R_lw / D18O_STD - 1) * 1000
    d18o_sugar = (R_sugar / D18O_STD - 1) * 1000

    return {'d18o_evaporative_sites': d18o_e,
            'd18o_leaf_water': d18o_lw,
            'd18o_leaf_sugar': d18o_sugar}