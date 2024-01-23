# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 15:06:01 2024

@author: Kersti Leppa
"""

import numpy as np
import pandas as pd

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

    def __init__(self, p: dict, pt_masks: list, pt_LAImax: list):
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
                    'e_stem' (float): fractionation by wood respiration [-]
                    'x' (float): fractionation by biomass synthesis [-]
                    'init_d13C_leaf' (float): initial leaf sugar store d13c [permil]
                    'init_d13C_C_pool' (float): initial C_pool sugar d13c [permil]
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
                    'init_d18O_leaf' (float): initial leaf sugar store d18O [permil]
                    'init_d18O_C_pool' (float): initial C_pool sugar d18O [permil]
                    },
                'C_leaf' (float): size of leaf sugar store [umol C m-2] ! one-sided leaf area
                'init_C_pool' (float): initial C_pool size [umol C m-2] ! one-sided leaf area       
                'k_pool' (float): trunover rate of C_pool [s-1]
                'woodsections_filename' (str):  filename for file containing 'date_formation' and 'date_maturation' for wood sections of interest
            pt_masks (list[np.ndarray]): 1.0 where lad > 0, nan elsewhere for planttypes
            pt_LAImax (list[float]): maximum leaf area of planttypes
    
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
        self.k_pool = p['k_pool']
        
        # number of planttypes
        self.n_pt = len(pt_masks)

        # initial conditions 
        # leaf level: dict (sunlit/shaded) of list of arrays for each planttype
        self.d13C_leaf_sugar = {'sunlit':[mask * p['d13C']['init_d13C_leaf'] for mask in pt_masks],
                                'shaded':[mask * p['d13C']['init_d13C_leaf'] for mask in pt_masks]}
        self.d18O_leaf_sugar = {'sunlit':[mask * p['d18O']['init_d18O_leaf'] for mask in pt_masks],
                                'shaded':[mask * p['d18O']['init_d18O_leaf'] for mask in pt_masks]}
        self.d18O_leaf_water = {'sunlit':[mask * np.nan for mask in pt_masks],
                                'shaded':[mask * np.nan for mask in pt_masks]}  # initialized with steady state result at first time step
        # plant level: list with floats for each planttype
        self.C_pool = [p['init_C_pool'] * LAImax for LAImax in pt_LAImax]  # per ground area
        self.d13C_C_pool = [p['d13C']['init_d13C_C_pool'] for n in range(self.n_pt)]
        self.d18O_C_pool = [p['d18O']['init_d18O_C_pool'] for n in range(self.n_pt)]
        
        # formation and maturation dates of wood sections of interest
        df = pd.read_csv(p['woodsections_filename'], header='infer', sep=';')
        self.date_formation = pd.to_datetime(df['date_formation'].values)
        self.date_maturation = pd.to_datetime(df['date_maturation'].values)
        
        # initialize arrays for calculation of woodsections d13C and d18O (final values concentration-weighted means)
        self.sum_C_times_d13C = [np.zeros_like(self.date_formation, float) for n in range(self.n_pt)]
        self.sum_C_times_d18O = [np.zeros_like(self.date_formation, float) for n in range(self.n_pt)]
        self.sum_C = [np.zeros_like(self.date_formation, float) for n in range(self.n_pt)]
    
    def run(self, dt: float, forcing: dict) -> dict:
        r"""
        Runs isotope model for one timestep.
        (list[np.ndarray] type inputs/outputs contain canopy profiles for each planttype)
        
        Args:
            dt (float): time step [s]
            forcing (dict):
                'leaf_temperature_sunlit' (list[np.ndarray]): leaf temperature, sunlit leaves [degC].
                'leaf_temperature_shaded' (list[np.ndarray]): leaf temperature, shaded leaves [degC].
                'net_co2_sunlit' (list[np.ndarray]): net phosynthesis, sunlit leaves [umol m-2 s-1].
                'net_co2_shaded' (list[np.ndarray]): net phosynthesis, shaded leaves [umol m-2 s-1].
                'dark_respiration_sunlit' (list[np.ndarray]): mitochondrial respiration, sunlit leaves [umol m-2 s-1].
                'dark_respiration_shaded' (list[np.ndarray]): mitochondrial respiration, shaded leaves [umol m-2 s-1].
                'transpiration_sunlit' (list[np.ndarray]): transpiraiton, sunlit leaves [mol m-2 s-1].
                'transpiration_shaded' (list[np.ndarray]): transpiraiton, shaded leaves [mol m-2 s-1].
                'stomatal_conductance_h2o_sunlit' (list[np.ndarray]): stomatal conductance for water vapor, sunlit leaves [mol m-2 s-1].
                'stomatal_conductance_h2o_shaded' (list[np.ndarray]): stomatal conductance for water vapor, shaded leaves [mol m-2 s-1].
                'boundary_conductance_h2o_sunlit' (list[np.ndarray]): boundary-layer conductance for water vapor, sunlit leaves [mol m-2 s-1].
                'boundary_conductance_h2o_shaded' (list[np.ndarray]): boundary-layer conductance for water vapor, shaded leaves [mol m-2 s-1].
                'leaf_internal_co2_sunlit' (list[np.ndarray]): CO2 mole fraction in the intercellular spaces, sunlit leaves [ppm].
                'leaf_internal_co2_shaded' (list[np.ndarray]): CO2 mole fraction in the intercellular spaces, shaded leaves [ppm].
                'leaf_surface_co2_sunlit' (list[np.ndarray]): CO2 mole fraction at leaf surface, sunlit leaves [ppm].
                'leaf_surface_co2_shaded' (list[np.ndarray]): CO2 mole fraction at leaf surface, shaded leaves [ppm].
                'co2' (np.ndarray): CO2 mole fraction in ambient air [ppm].
                'h2o' (np.ndarray): mole fraction of water vapor in atmosphere [mol mol-1].
                'air_pressure' (float): atmospheric pressure [Pa].
                'air_temperature' (float): air temperature above canopy [degC].
                'LAIz' (list[np.ndarray]): layerwise one-sided leaf-area index [m2 m-2].
                'f_sl' (np.ndarray): sunlit canopy fraction [-].
                'd13Ca' (float): carbon isotope composition in CO2 of ambient air [permil].
                'd18Ov' (float): d18O of water vapor [permil].
                'd18O_sw' (float): d18O of source water [permil].
                'datetime': datetime of current timestep.

        Returns:
            results (dict): 
                '13c_discrimination_sunlit' (list[np.ndarray]): discrimination of net photosynthesis, sunlit leaves [permil].
                '13c_discrimination_shaded' (list[np.ndarray]): discrimination of net photosynthesis, shaded leaves [permil].
                'd13c_net_co2_flux_sunlit' (list[np.ndarray]): d13C of net CO2 flux, sunlit leaves [permil].
                'd13c_net_co2_flux_shaded' (list[np.ndarray]): d13C of net CO2 flux, shaded leaves [permil].
                'd13c_leaf_sugar_sunlit' (list[np.ndarray]): d13C of needle sugar pool, sunlit leaves [permil].
                'd13c_leaf_sugar_shaded' (list[np.ndarray]): d13C of needle sugar pool, shaded leaves [permil].
                'd18o_evaporative_sites_sunlit' (list[np.ndarray]): d18o at evaporative sites, sunlit leaves [permil].
                'd18o_evaporative_sites_shaded' (list[np.ndarray]): d18o at evaporative sites, shaded leaves [permil].
                'd18o_leaf_water_sunlit' (list[np.ndarray]): d18o of leaf wate, sunlit leavesr [permil].
                'd18o_leaf_water_shaded' (list[np.ndarray]): d18o of leaf water, shaded leaves [permil].
                'd18o_leaf_sugar_sunlit' (list[np.ndarray]): d18o of needle sugar pool, sunlit leaves [permil].
                'd18o_leaf_sugar_shaded' (list[np.ndarray]): d18o of needle sugar pool, shaded leaves [permil].
        """
        pt_results = [{} for i in range(self.n_pt)]  # list of n_pt dicts
        
        # mask for woodsections that this time step is relevant for
        ix = (forcing['datetime'] >= self.date_formation) & (forcing['datetime'] <= self.date_maturation)
        
        for n in range(self.n_pt):
            
            # --- d13C ---
            if self.solve_d13C:
                
                # at leaf-level
                for sl_sh in ['sunlit','shaded']:
                    Tk = forcing['leaf_temperature_'+sl_sh][n] + DEG_TO_KELVIN
                    tau_c = 42.75 * np.exp(37830*(Tk - TN) / (TN * GAS_CONSTANT * Tk))
                    # run timestep
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
                # total net co2 flux from leaves per ground area (umol m-2)
                F_leaf = np.nansum(
                    forcing['net_co2_sunlit'][n] * forcing['f_sl'] * forcing['LAIz'][n] +
                    forcing['net_co2_shaded'][n] * (1 - forcing['f_sl']) * forcing['LAIz'][n]
                    )
                # net co2 flux -weighted mean d13C of sugars from leaves
                d13C_leaf_sugar = np.nansum(
                    self.d13C_leaf_sugar['sunlit'][n] * forcing['net_co2_sunlit'][n] * forcing['f_sl'] * forcing['LAIz'][n] +
                    self.d13C_leaf_sugar['shaded'][n] * forcing['net_co2_shaded'][n] * (1 - forcing['f_sl']) * forcing['LAIz'][n]
                    ) / F_leaf
                # run timestep
                pt_res = plant_d13C(dt=dt,
                                    d13C_leaf_sugar=d13C_leaf_sugar,
                                    F_leaf=F_leaf,
                                    F_wood=0.0,
                                    k_pool=self.k_pool, 
                                    e=self.d13C_para['e_stem'], x=self.d13C_para['x'],
                                    d13C_pool_init=self.d13C_C_pool[n],
                                    C_pool_init=self.C_pool[n])
                # sum for calculating concentration weighted average for woodsections cellulose d13C
                self.sum_C_times_d13C[n][ix] += pt_res['C_pool'] * pt_res['d13c_cellulose']
                # update sate
                self.d13C_C_pool[n] = pt_res['d13c_C_pool']
                # add to results
                pt_results[n].update(pt_res)
        
            # --- d18O ---
            if self.solve_d18O:
                
                # at leaf-level
                for sl_sh in ['sunlit','shaded']:
                    # run timestep
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
                # total net co2 flux from leaves per ground area (umol m-2)
                F_leaf = np.nansum(
                    forcing['net_co2_sunlit'][n] * forcing['f_sl'] * forcing['LAIz'][n] +
                    forcing['net_co2_shaded'][n] * (1 - forcing['f_sl']) * forcing['LAIz'][n]
                    )
                # net co2 flux -weighted mean d18O of sugars from leaves
                d18O_leaf_sugar = np.nansum(
                    self.d18O_leaf_sugar['sunlit'][n] * forcing['net_co2_sunlit'][n] * forcing['f_sl'] * forcing['LAIz'][n] +
                    self.d18O_leaf_sugar['shaded'][n] * forcing['net_co2_shaded'][n] * (1 - forcing['f_sl']) * forcing['LAIz'][n]
                    ) / F_leaf
                # biochemical fractionation factor 
                if self.d18O_para['e_wc']['constant']:
                    e_wc = self.d18O_para['e_wc']['value']
                else:  # temperature-dependent (sternberg et al. 2011)
                    e_wc = 1e-3*(0.0084*forcing['air_temperature']**2 - 0.51*forcing['air_temperature'] + 33.172)
                # run timestep
                pt_res = plant_d18O(dt=dt,
                                    d18O_leaf_sugar=d18O_leaf_sugar,
                                    F_leaf=F_leaf,
                                    F_wood=0.0,
                                    k_pool=self.k_pool, 
                                    pex=self.d18O_para['pex'],
                                    e_wc=e_wc,
                                    d18O_xylem=forcing['d18O_sw'],
                                    d18O_pool_init=self.d18O_C_pool[n],
                                    C_pool_init=self.C_pool[n])
                # sum for calculating concentration weighted average for woodsections cellulose d18O
                self.sum_C_times_d18O[n][ix] += pt_res['C_pool'] * pt_res['d18o_cellulose']
                # update sate
                self.d18O_C_pool[n] = pt_res['d18o_C_pool']
                # add to results
                pt_results[n].update(pt_res)
            
            if self.solve_d13C or self.solve_d18O: # do these only once!
                # sum for calculating concentration weighted average for woodsections cellulose d13C/d18O
                self.sum_C[n][ix] = self.sum_C[n][ix] + pt_res['C_pool']
                # update sate
                self.C_pool[n] = pt_res['C_pool']

        # convert list of dicts to dict of lists and append
        results = {}
        for k, v in pt_results[0].items():
            results[k] = [x[k] for x in pt_results]

        return results
    
    def calculate_woodsections_values(self, dates: pd.DatetimeIndex) -> tuple:
        """
        Calculates concentration weighted average for woodsections cellulose d13C/d18O.

        Args:
            dates (pd.DatetimeIndex): datetime for all timesteps.

        Returns:            
            results (list[dict]): own dict for each woodsection
                'd13c_treering_celluose' (list): tree ring cellulose d13C for each planttype
                'd18o_treering_celluose' (list): tree ring cellulose d18O for each planttype
            indices (list): indices corresponding to woodsections formation period mid date in dates
        """
        
        # wood section isotope values will be saved to date in middle of corresponding formation period
        if self.solve_d13C or self.solve_d18O:
            date_mid = self.date_formation + (self.date_maturation - self.date_formation)/2
            date_mid = date_mid.floor('H')  # minutes to zero to mach forcing indices
            ix = dates.isin(date_mid)
            indices = np.arange(len(dates))[ix]
            ixx = date_mid.isin(dates)
        else:
            indices = []
        
        pt_results = [{} for i in range(self.n_pt)]  # list of n_pt dicts
        
        for n in range(self.n_pt):
            if self.solve_d13C:
                d13C_cellulose = self.sum_C_times_d13C[n][ixx] / self.sum_C[n][ixx]
                pt_results[n].update(
                    {'d13c_treering_celluose': d13C_cellulose})
                
            if self.solve_d18O:
                d18O_cellulose = self.sum_C_times_d18O[n][ixx] / self.sum_C[n][ixx]
                pt_results[n].update(
                    {'d18o_treering_celluose': d18O_cellulose})
            
        # convert list of dicts to list (for each timestep with values) of dicts of lists for each planttype
        results = [{} for i in range(len(indices))]
        for k, v in pt_results[0].items():
            for i in range(len(indices)):
                results[i][k] = [x[k][i] for x in pt_results]

        return results, indices

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
    
    # keep vpd positive
    w_a = np.minimum(w_i - EPS, w_a)

    e_star = np.exp(1137/(T + DEG_TO_KELVIN)**2 - 0.4156 /(T + DEG_TO_KELVIN) - 0.0020667) - 1

    e_kkb = (e_k * gb + e_kb * gs) / (gb + gs)
    
    if e_wc['constant']:
        e_wc = e_wc['value']
    else:
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

def plant_d13C(dt: float, d13C_leaf_sugar: float, F_leaf: float, 
               F_wood: float, d13C_pool_init: float, C_pool_init: float,
               k_pool: float=9.51e-07, e: float=0.0, x: float=0.0) -> dict:
    """
    Solves one timestep for d13C in the plant carbon pool (can be considered phloem?)
    and corresponding value for tree ring cellulose d13C. (see Ogee et al., 2009)

    Args:
        dt (float): timestep [s].
        d13C_leaf_sugar (float): d13C of leaf sugar [permil].
        F_leaf (float): flux from leaves per ground area [umol m-2 s-1].
        F_wood (float): woody respiration per ground area [umol m-2 s-1].
        d13C_pool_init (float): initial C pool d13C [permil].
        C_pool_init (float): initial C pool size per ground area [umol C m-2].
        k_pool (float, optional): trunover rate of C_pool [s-1]. Defaults to 9.51e-07.
        e (float, optional): fractionation by wood respiration. Defaults to 0.0.
        x (float, optional): fractionation by biomass synthesis. Defaults to 0.0.

    Returns:
        dict: 
            'C_pool' (float): C pool size [umol C m-2]
            'd13c_C_pool' (float): C pool d13C [permil]
            'd13c_cellulose' (float): d13C in cellulose [permil]
    """
    R_sugar = (d13C_leaf_sugar/1000 + 1) * D13C_STD
    R_pool_init = (d13C_pool_init/1000 + 1) * D13C_STD
    
    # Calculate size and isotopic ratio of non-structural carbon pool: C_pool and R_pool
    C_pool = ((F_leaf - F_wood) * dt + C_pool_init) / (1 + k_pool * dt)
    R_pool = ((F_leaf * R_sugar * dt + C_pool_init * R_pool_init) /
              (C_pool + ((1-x) * k_pool * C_pool + (1-e) * F_wood) * dt))
    
    d13C_pool = (R_pool / D13C_STD - 1) * 1000
    
    return {'C_pool': C_pool, 
            'd13c_C_pool': d13C_pool, 
            'd13c_cellulose': d13C_pool} # for 13C isotopic ratio in cellulose same as d13C_pool

def plant_d18O(dt: float, d18O_leaf_sugar: float, F_leaf: float, 
               F_wood: float, d18O_xylem: float, d18O_pool_init: float, 
               C_pool_init: float, k_pool: float=9.51e-07, pex: float=0.42,
               e_wc: float=27e-3) -> dict:
    """
    Solves one timestep for d18O in the plant carbon pool (can be considered phloem?)
    and corresponding value for tree ring cellulose d18O. (see Ogee et al., 2009)

    Args:
        dt (float): timestep [s].
        d18O_leaf_sugar (float): d18O of leaf sugar [permil].
        F_leaf (float): flux from leaves per ground area [umol m-2 s-1].
        F_wood (float): woody respiration per ground area [umol m-2 s-1].
        d18O_xylem (float): d18O of source water [permil].
        d18O_pool_init (float): initial C pool d18O [permil].
        C_pool_init (float): initial C pool size per ground area [umol C m-2].
        k_pool (float, optional): trunover rate of C_pool [s-1]. Defaults to 9.51e-07.
        pex (float, optional): proportion of oxygen atoms exchanged with source 
                water during cellulose synthesis. Defaults to 0.42.
        e_wc (float, optional): biochemical fractionation factor. Defaults to 27e-3.

    Returns:
        dict:
            'C_pool' (float): C pool size [umol C m-2]
            'd18o_C_pool' (float): C pool d18O [permil]
            'd18o_cellulose' (float): d18O in cellulose [permil]
    """
    R_sugar = (d18O_leaf_sugar/1000 + 1) * D18O_STD
    R_sw = (d18O_xylem/1000 + 1) * D18O_STD
    R_pool_init = (d18O_pool_init/1000 + 1) * D18O_STD
    
    # Calculate size and isotopic ratio of non-structural carbon pool: C_pool and R_pool
    C_pool = ((F_leaf - F_wood) * dt + C_pool_init) / (1 + k_pool * dt)
    R_pool = ((F_leaf * R_sugar * dt + C_pool_init * R_pool_init) /
              (C_pool + (k_pool * C_pool + F_wood) * dt))
    
    # isotopic ratio in cellulose affected by exchange with source water during cellulose synthesis
    R_cellulose = pex * (1 + e_wc) * R_sw  + (1 - pex) * R_pool
                       
    d18O_pool = (R_pool / D18O_STD - 1) * 1000
    d18O_cellulose = (R_cellulose / D18O_STD - 1) * 1000
    
    return {'C_pool': C_pool, 
            'd18o_C_pool': d18O_pool, 
            'd18o_cellulose': d18O_cellulose}