#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. module: bottomlayer.carbon
    :synopsis: pyAPES-model component to calculate carbon exchange in moss / litter layer and soil
.. moduleauthor:: Samuli Launiainen & Antti-Jussi Kieloaho

"""

import numpy as np
from typing import Dict, Tuple, List

from pyAPES.utils.constants import EPS

class BryophyteFarquhar(object):
    def __init__(self, para: Dict, carbon_pool: float=0):
        r"""
        Farquhar-model for moss community photosynthesis and dark respiration.
        Returns fluxes per unit ground area; code can be, however, 
        be used also mass-basis if appropriate units are used.
        
        Args:
            - para' (dict)
            - Vcmax [umol m-2 (ground) s-1], maximum carboxylation velocity at 25C
            - Jmax [umol m-2 (ground) s-1], maximim electron transport rate at 25C
            - Rd [umol m-2 (ground) s-1], dark respiration rate at 25C
            - alpha [-], quantum efficiency
            - theta [-], curvature parameter
            - beta [-], co-limitation parameter
            - gmax [mol m-2 (ground) s-1], conductance for CO2 at water content 'wref'
            - wref [g g-1], parameter of conductance - water content relationship
            - a0 [-], parameter of conductance - water content curve
            - a1 [-], parameter of conductance - water content curve
            - CAP_desic (list) [-], parameters of desiccation curve 
            - tresp' (dict): parameters of photosynthetic temperature response curve
                - Vcmax (list): [activation energy [kJ mol-1], 
                                 deactivation energy [kJ mol-1]
                                 entropy factor [kJ mol-1]
                                ]
                - Jmax (list): [activation energy [kJ mol-1], 
                                 deactivation energy [kJ mol-1]
                                 entropy factor [kJ mol-1]
                                ]
                - Rd (list): [activation energy [kJ mol-1]
            - carbon_pool [g C m-2 (ground)], initial carbon pool
        """
        self.photopara = para
        self.carbon_pool = carbon_pool

    def co2_exchange(self, Qp: float, Ca: float, T: float, w: float, wstar: float=None) -> Dict:
        """
        Computes net CO2 exchange of moss community.
        Args (float):
            - self (object)
            - Qp [umol m-2 s-1], incident PAR
            - Ca [ppm] ambient CO2
            - T [degC] moss temperature
            - w [g g-1] moss gravimetric water content
            - wstar [g g-1] delayed water content (for desiccation recovery, not implemented)
        Returns (dict of floats):
            - An [umol m-2 s-1] net CO2 exchange rate, An = -A + Rd, <0 is uptake
            - A [umol m-2 s-1] photosynthesis rate 
            - Rd [umol m-2 s-1] dark respiration rate 
            - Cc [ppm] intercellular CO2 
            - g [mol m-2 s-1] conductance for CO2
        """
        p = self.photopara.copy()
        cap, rcap = relative_capacity(p['CAP_desic'], w, wstar)
        p['Vcmax'] *= cap
        p['Jmax'] *= cap
        p['alpha'] *= cap
        p['Rd'] *= rcap

        # conductance (mol m-2 s-1)
        g = conductance(p, w)

        # solve Anet and Cc iteratively until Cc converges
        err = 10^9
        Cc = 0.8*Ca

        iterNo = 0
        while err > 1e-3:
            Cco = Cc
            An, Rd, _, _ = photo_farquhar(p, Qp, Cc, T)

            Cc = Ca - An / g  # new Cc
            Cc = 0.5*(Cco + Cc)
            err = np.nanmax(abs(Cc - Cco))

            if iterNo > 50:
                Cc = 0.8*Ca
                An, Rd, _, _ = photo_farquhar(p, Qp, Cc, T)
                break
            iterNo += 1

        return {'net_co2': -An,
                'photosynthesis': An + Rd,
                'respiration': Rd,
                'internal_co2': Cc,
                'conductance_co2': g
                }

def conductance(para: Dict, w: float) -> float:
    """
    Conductance for CO2 diffusion from bulk air to chloroplast in bryophyte.
    Assumes g = gref * fw, where gref is species-specific internal conductance, 
    occurring at w == wref, and fw [-] describes decay of conductance due to
    external water (w > wref).
    
    gref and wopt are bryophyte traits, while the shape of fw is fixed based on
    data of Williams & Flanagan, 1998 (Oecologia). We normalized maximum conductance for Pleurozium and
    Sphagnum to unity, and determined respective wref's. Then a decreasing exponential 
    function was fitted to data.
    
    Args:
        - para (dict of floats):
            - gref [mol m-2 (ground) s-1], conductance for CO2 at optimum water content 'wref'
            - wref [g g-1], parameter of conductance - water content relationship
            - a0 [-], parameter of conductance - water content curve
            - a1 [-], parameter of conductance - water content curve
        - w [g g-1] gravimetric water content

    Returns:
        -- g [mol m\ :sup:`-2`\ (ground) s\ :sup:`-1`\] conductance for CO2
    """
 
    gmax = para['gmax']
    a0 = para['a0']
    a1 = para['a1']
    wopt = para['wopt']
    
    #g = gmax * np.minimum(1.0, a0*np.exp(a1*(w-wopt)) + (1.0 - a0))
    g = gmax * (a0*np.exp(a1*(w-wopt)) + (1.0 - a0)) # this allows g to go above gmax in dry moss
    return g


def relative_capacity(para: Dict, w: float, wstar: float=None) -> Tuple(float, float):
    """
    Relative photosynthetic capacity and dark respiration rates as a function of water content.
    
    Args:
        - para (list),parameters of desiccation curve 
        - w [g g-1], current gravimetric water content
        - wstar [g g-1], delayed effective water content (for desiccation recovery, not implemented)
    Returns:
         - rphoto [-], photosynthetic capacity relative to well-hydrated state
         - rrd [-], dark respiration rate relative to well-hydrated state
    Note:
        - currently no hysteresis (drying-wetting effects equal) and no difference 
        between photosynthetic capacity and respiration sensitivity to water content- 
    """
    p = para
    #p = para['CAP_desic']
    #r = para['CAP_rewet']
    
    # drying phase is function of w
    cap_dec = np.maximum(0.0, np.minimum(1.0 + p[0]*np.log(w/p[1]), 1.0)) # [0...cap_dec...1]
    
    # recovery from desiccation is a function of wstar; now assume reversible
    cap_rw = 1.0
    
    
    rphoto = np.minimum(cap_dec, cap_rw)
    del p #, r
    
    # respiration
    # p = para['Rd_desic']
    # r = para['Rd_rewet]
    
    #rrd = 1.0
    rrd = rphoto # assume to behave as photo
    return rphoto, rrd

def photo_farquhar(photop: Dict, Qp: float, Ci: float, T: float) -> Tuple(float, float, float, float):
    """
    Calculates moss photosynthesis, dark respiration and net CO2 exhange using Farquhar-model.
    
    Args:
        - photop (dict)
            - Vcmax [umol m-2 s-1], maximum carboxylation velocity at 25 degC
            - Jmax [umol m-2 s-1], maximum electron transport rate at 25 degC
            - Rd [umol m-2 s-1], dark respiration rate at 25 degC
            - alpha [-], quantum efficiency
            - theta [-], curvature parameter [-]
            - beta [-],  co-limitation parameter [-]
            - tresp' (dict): parameters of photosynthetic temperature response curve
                - Vcmax (list): [activation energy [kJ mol-1], 
                              deactivation energy [kJ mol-1], 
                              entropy factor [kJ mol-1]
                             ]
                - Jmax (list): [activation energy [kJ mol-1], 
                              deactivation energy [kJ mol-1], 
                              entropy factor [kJ mol-1]
                             ]
                - Rd (list): [activation energy [kJ mol-1]]
                                  
        - Qp [umol m-2 s-1], incident PAR
        - Ci [ppm] intercellular CO2
        - T [degC] temperature

    Returns:
       - An [umol m-2 s-1], net CO2 exchange rate, An = A - R, >0 is uptake
       - Rd [umol m-2 s-1], dark respiration rate 
       - Av [umol m-2 s-1], rubisco limited rate
       - Aj [umol m-2 s-1], RuBP -regeneration limited rate
        
    """
    Tk = T + 273.15  # K

    # --- constants
    Coa = 2.10e5  # O2 in air (umol/mol)
    TN = 298.15  # reference temperature 298.15 K = 25degC
    R = 8.314427  # gas constant, J mol-1 K-1

    # --- params ----
    # lai = photop['LAI']
    Vcmax = photop['Vcmax'] 
    Jmax = photop['Jmax'] 
    Rd = photop['Rd'] 
    alpha = photop['alpha']
    theta = photop['theta']
    beta = photop['beta']  # co-limitation parameter

    # --- CO2 compensation point (in absence of mitochondrial respiration)-------
    Tau_c = 42.75 * np.exp(37830*(Tk - TN) / (TN * R * Tk))

    # ---- Kc & Ko (umol/mol), Rubisco activity for CO2 & O2 ------
    Kc = 404.9 * np.exp(79430.0*(Tk - TN) / (TN * R * Tk))
    Ko = 2.784e5 * np.exp(36380.0*(Tk - TN) / (TN * R * Tk))

    if 'tresp' in photop:  # adjust parameters for temperature
        tresp = photop['tresp']
        Vcmax_T = tresp['Vcmax']
        Jmax_T = tresp['Jmax']
        Rd_T = tresp['Rd']
        Vcmax, Jmax, Rd, Tau_c = photo_temperature_response(Vcmax, Jmax, Rd, Vcmax_T, Jmax_T, Rd_T, Tk)
    
    Km = Kc*(1.0 + Coa / Ko)
    J = (Jmax + alpha*Qp -((Jmax + alpha*Qp)**2.0 - (4.0*theta*Jmax*alpha*Qp))**0.5) / (2.0*theta)

    # -- rubisco -limited rate (CO2 limitation)
    Av = Vcmax * (Ci - Tau_c) / (Ci + Km)
    # -- RuBP -regeneration limited rate (light limitation)
    Aj = J/4 * (Ci - Tau_c) / (Ci + 2.0*Tau_c)

    # An = np.minimum(Av, Aj) - Rd  # single limiting rate
    x = Av + Aj
    y = Av * Aj
    An = np.maximum((x - (x**2 - 4*beta*y)**0.5) / (2*beta), 0) - Rd  # co-limitation
    
    return An, Rd, Av, Aj


def photo_temperature_response(Vcmax25: float, Jmax25: float, Rd25: float,
                               Vcmax_T: List, Jmax_T: List, Rd_T: List, T: float) -> Tuple(float, float, float, float):
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

    # ---constants
    #NT = 273.15
    T0 = 298.15  # reference temperature 298.15 K = 25degC
    R = 8.314427  # gas constant, J mol-1 K-1

    # --- CO2 compensation point -------
    Gamma_star = 42.75 * np.exp(37830*(T - T0) / (T0 * R * T))

    # # ---- Kc & Ko (umol/mol), Rubisco activity for CO2 & O2 ------
    # Kc = 404.9 * np.exp(79430.0*(T - TN) / (TN * R * T))
    # Ko = 2.784e5 * np.exp(36380.0*(T - TN) / (TN * R * T))

    # ------  Vcmax (umol m-2(leaf)s-1) ------------
    Ha = 1e3 * Vcmax_T[0]  # J mol-1, activation energy Vcmax
    Hd = 1e3 * Vcmax_T[1]  # J mol-1, deactivation energy Vcmax
    Sd = Vcmax_T[2]  # entropy factor J mol-1 K-1

    NOM = np.exp(Ha * (T - T0) / (R*T0*T)) * (1.0 + np.exp((T0*Sd - Hd) / (T0*R)))
    DENOM = (1.0 + np.exp((T*Sd - Hd) / (T*R)))
    Vcmax = Vcmax25 * NOM / DENOM

    del Ha, Hd, Sd, DENOM, NOM

    # ----  Jmax (umol m-2(leaf)s-1) ------------
    Ha = 1e3 * Jmax_T[0]  # J mol-1, activation energy Vcmax
    Hd = 1e3 * Jmax_T[1]  # J mol-1, deactivation energy Vcmax
    Sd = Jmax_T[2]  # entropy factor J mol-1 K-1

    NOM = np.exp(Ha * (T - T0) / (R*T0*T)) * (1.0 + np.exp((T0*Sd - Hd) / (T0*R)))
    DENOM = (1.0 + np.exp((T*Sd - Hd) / (T*R)))
    Jmax = Jmax25*NOM / DENOM

    del Ha, Hd, Sd, DENOM, NOM

    # --- Rd (umol m-2(leaf)s-1) -------
    Ha = 1e3 * Rd_T[0]  # J mol-1, activation energy dark respiration
    Rd = Rd25 * np.exp(Ha*(T - T0) / (T0 * R * T))

    return Vcmax, Jmax, Rd, Gamma_star

def topt_deltaS_conversion(Ha: float, Hd: float, dS: float=None, Topt: float=None) -> float:
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

# ***************************************************************
class BryophyteCarbon(object):
    
    def __init__(self, para: Dict, carbon_pool: float=0):
        r"""
        Empirical photosynthesis and dark respiration model for moss community.
        Based on empirical light-response with temperature and moisture effects included.

        Gives fluxes per unit ground area.
        
        Based on Launiainen et al. 2015. Ecol. Mod. NOTE: Not currently used in pyAPES-MLM (SL 14.3.2023)
        Args:
            - para (dict):
                - amax [umol m-2 s-1], light-saturated photosynthetic rate
                - beta [umol m-2 s-1],  half-saturation PAR
                - moisture coeff (list of floats), moisture response parameters
                - temperature_coeff (list of floats), temperature response parameters
                - q10 [-], temperature sensitivity of respiration rate
                - r10 [g g-1], base rate of respiration at 10 degC 
                - max_water_content [g g-1], saturated gravimetric water content 
            - carbon_pool [g C m-2 (ground)], initial state of moss carbon pool
        """
        self.amax = para['amax']
        self.beta = para['beta']
        self.moisture_coeff = ['moisture_coeff']
        self.temperature_coeff = ['temperature_coeff']

        self.r10 = para['r10']
        self.q10 = para['q10']

        self.max_water_content = para['max_water_content'] # g g-1
        self.carbon_pool = carbon_pool

    def carbon_exchange(self, temperature: float, water_content: float, incident_par: float) -> Dict:
        r"""
        Estimates photosynthesis and respiration rates of bryophyte layer.

        Photosynthesis is restricted by both tissue water content
        (dry conditions) and excess water film on leaves (diffusion limitation)
        as in Williams and Flanagan (1996, oecologia). Water content
        coefficients are 3rd order polynomial fitted to the data represented by
        WF96 and used to calculate effect of water content on photosynthesis.

        Empirical modifier of photosynthesis due to water content assumes that
        both light-limited and Rubisco-limited assimilation of carbon are
        affected similarly. This seems to apply for Pleurozium and Sphagnum
        when normalized water content is used as scaling. Assumes that
        there is always 5 percents left in photosynthetic capacity.Empirical
        modifier of photosynthesis due to temperature is based on
        late growing season presented in Fig. 2 in Frolking et al. (1996).

        References:
            Frolking et al. (1996). Global Change Biology 2:343-366
            Williams and Flanagan (1996). Oecologia 108:38-46

        Args:
            - water_content [g g-1]
            - temperature [degC]
            - incident_par [umol m-2 s-1], incident photosynthetically active radiation

        Returns (dict):
            - photosynthesis_rate [umol m-2 (ground) s-1]
            - respiration_rate [umol m-2 (ground) s-1]
            - co_flux [umol m-2 (ground) s-1], net co2 exchange, <0 uptake 
        """

        incident_par = np.maximum(EPS, incident_par)

        normalized_water_content = water_content / self.max_water_content

        # hyperbolic light response at community level [umol m-2(ground) s-1]
        light_response = self.amax * incident_par / (self.beta + incident_par )

        # moisture and temperature responses [-]
        water_modifier = (self.moisture_coeff[3]
                          + self.moisture_coeff[2] * normalized_water_content
                          + self.moisture_coeff[1] * normalized_water_content ** 2.0
                          + self.moisture_coeff[0] * normalized_water_content ** 3.0)

        water_modifier = np.maximum(0.05, water_modifier)

        temperature_modifier = (self.temperature_coeff[3]
                                + self.temperature_coeff[2] * temperature
                                + self.temperature_coeff[1] * temperature ** 2.0
                                + self.temperature_coeff[0] * temperature ** 3.0)

        temperature_modifier = np.maximum(0.01, temperature_modifier)

        temperature_modifier = np.minimum(1.0, temperature_modifier)

        # [umol m-2 (ground) s-1]
        photosynthetic_rate = light_response * water_modifier * temperature_modifier

        # respiration rate [umol m-2 (ground) s-1]

        """ replace with smooth function and move as parameter """
        if water_content < 7.0:
            water_modifier_respiration = (
                -0.45 + 0.4 * water_content
                - 0.0273 * water_content ** 2)
        else:
            water_modifier_respiration = (
                -0.04 * water_content + 1.38)

        # effect of water content is set to range 0.01 to 1.0
        water_modifier_respiration = np.maximum(0.01, np.minimum(1.0, water_modifier_respiration))

        # r = r10 * Q10^((T-10) / 10) [umol m-2 s-1]
        respiration_rate = (
            self.r10 * self.q10**((temperature - 10.0) / 10.0) * water_modifier_respiration
            )

        # umol m-2 (ground) s-1
        return {
            'photosynthesis': photosynthetic_rate,
            'respiration': respiration_rate,
            'net_co2': -photosynthetic_rate + respiration_rate
            }

class OrganicRespiration(object):
    """
    Model for litter layer respiration. Values per unit ground area
    """
    def __init__(self, para: Dict, carbon_pool: float=0.0):
        """
        Args:
            - q10 [-], temperature sensitivity of respiration rate
            - r10  [umol m-2 s-1], base rate of respiration at 10 degC
            - carbon_pool [g C m-2 (ground)], initial state of litter carbon pool
        """
        self.r10 = para['r10']
        self.q10 = para['q10']
        #self.moisture_coeff = para['respiration']['moisture_coeff']
        self.carbon_pool = 0.0

    def respiration(self, temperature: float, volumetric_water: float) -> Dict:
        """
        Litter layer respiration rate from temperature and moisture.
        Args:
            - 'temperature' (float): [degC]
            - 'water_content' (float): gravimetric water content [g g-1]
        Returns (dict):
            - photosynthesis_rate, always zero
            - respiration_rate [umol m-2 s-1]
            - co_flux [umol m-2 s-1], net co2 exchange, <0 uptake
        
        Note: SL 14.3.2023: Implement moisture response!
        """
        r = self.r10 * np.power(self.q10, (temperature - 10.0) / 10.0)

        # add moisture response
        fW = 1.0
        r = r * fW
        return {'photosynthesis': 0.0,
                'respiration': r,
                'net_co2': r
               }


class SoilRespiration(object):
    """
    Soil respiration model. Lumps heterotrophic and autotrophic respiration.
    Rsoil = R10 * Q10*[(T-10)/10] * fmoisture
    """
    def __init__(self, para: Dict, weights: float=1):
        """

        Args:
            para (Dict): 
                - 'r10' (float): [umol m-2 s-1], base rate at 10 degC
                - 'Q10' (float): [-], temperature sensitivity
                - 'moisture_coeff' (list): [-], moisture response parameters

            weights (int, optional): weight factors for soil layers
        """
        # base rate [umol m-2 s-1] and temperature sensitivity [-]
        self.r10 = para['r10']
        self.q10 = para['q10']

        # moisture response of Skopp et al. 1990
        self.moisture_coeff = para['moisture_coeff']

        if weights is not None:
            # soil respiration computed in layers and weighted
            self.weights = weights
            self.Nlayers = len(weights)

    def respiration(self, soil_temperature: np.array, volumetric_water: np.array, volumetric_air: np.array) -> float:
        """ Soil respiration beneath litter/moss.

        Heterotrophic and autotrophic respiration rate (CO2-flux) for mineral forest soil, 
        based on Pumpanen et al. (2003) Soil.Sci.Soc.Am

        Restricts respiration by soil moisuture as in Skopp et al. (1990), Soil.Sci.Soc.Am

        Args:

            soil_temperature (float|array) [degC]
            volumetric_water (float|array) [m3 m-3]
            volumetric_air float|array) [m3 m-3]
            
        Returns:
            soil respiration rate (float): [umol m-2 (ground) s-1]

        """
        # Skopp limitparam [a,b,d,g] for two soil types
        # sp = {'Yolo':[3.83, 4.43, 1.25, 0.854],
        #       'Valentine': [1.65,6.15,0.385,1.03]}

        # unrestricted respiration rate
        x = self.r10 * np.power(self.q10, (soil_temperature - 10.0) / 10.0)

        # moisture response (substrate diffusion, oxygen limitation)
        f = np.minimum(self.moisture_coeff[0] * volumetric_water**self.moisture_coeff[2],
                       self.moisture_coeff[1] * volumetric_air**self.moisture_coeff[3])
        f = np.minimum(f, 1.0)

        respiration = x * f

        if hasattr(self, 'weights'):
            respiration = sum(self.weights * respiration[0:self.Nlayers])

        return respiration
    
# EOF
