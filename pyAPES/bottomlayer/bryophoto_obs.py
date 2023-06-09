# -*- coding: utf-8 -*-
"""

NOTE: OBSOLETE - NOW ALL MOVED INTO bottomlayer.carbon! SL 7.6.23

.. module: forestfloor
    :synopsis: pyAPES-model bottomlayer component
.. moduleauthor:: Samuli Launiainen

Module for moss community photosynthesis. Process-based (Farquhar-type) & empirical model
to compute moss photosynthesis, dark respiration and net CO2 exchange as a function of light, temperature and water content.

Last edit: 15.3.2023 Samuli 
"""
import matplotlib.pyplot as plt
import numpy as np
#from scipy.integrate import odeint
EPS = np.finfo(float).eps  # machine epsilon

# Constants used in the model calculations.
#: [J mol\ :sup:`-1`\ ], latent heat of vaporization at 20\ :math:`^{\circ}`\ C
LATENT_HEAT = 44100.0
# LATENT_HEAT = 2,501e6 #  J kg
#: [kg mol\ :sup:`-1`\ ], molar mass of H\ :sub:`2`\ O
MOLAR_MASS_H2O = 18.015e-3
#: [kg mol\ :sup:`-1`\ ], molar mass of CO\ :sub:`2`\
MOLAR_MASS_CO2 = 44.01e-3
#: [kg mol\ :sup:`-1`\ ], molar mass of C
MOLAR_MASS_C = 12.01e-3
#: [J kg\ :sup:`-1` K\ :sup:`-1`\ ], specific heat of H\ :sub:`2`\ O
SPECIFIC_HEAT_H2O = 4.18e3
#: [J kg\ :sup:`-1` K\ :sup:`-1`\ ], specific heat of organic matter
SPECIFIC_HEAT_ORGANIC_MATTER = 1.92e3
#: [J mol\ :sup:`-1` K\ :sup:`-1`\ ], heat capacity of air at constant pressure
SPECIFIC_HEAT_AIR = 29.3
#: [W m\ :sup:`-2` K\ :sup:`-4`\ ], Stefan-Boltzmann constant
STEFAN_BOLTZMANN = 5.6697e-8
#: [K], zero degrees celsius in Kelvin
DEG_TO_KELVIN = 273.15

#: [K], zero degrees celsius in Kelvin
NORMAL_TEMPERATURE = 273.15
#: [mol m\ :sup:`-3`\ ], density of air at 20\ :math:`^{\circ}`\ C
AIR_DENSITY = 41.6
#: [m\ :sup:`2` s\ :sup:`-1`\ ], kinematic viscosity of air at 20\ :math:`^{\circ}`\ C
AIR_VISCOSITY = 15.1e-6
#: [m\ :sup:`2` s\ :sup:`-1`\ ], thermal diffusivity of air at 20\ :math:`^{\circ}`\ C
THERMAL_DIFFUSIVITY_AIR = 21.4e-6
#: [m\ :sup:`2` s\ :sup:`-1`\ ], molecular diffusvity of CO\ :sub:`2` at 20\ :math:`^{\circ}`\ C
MOLECULAR_DIFFUSIVITY_CO2 = 15.7e-6
#: [m\ :sup:`2` s\ :sup:`-1`\ ], molecular diffusvity of H\ :sub:`2`\ at 20\ :math:`^{\circ}`\ C
MOLECULAR_DIFFUSIVITY_H2O = 24.0e-6
#: [J mol\ :sup:`-1` K\ :sup:``-1], universal gas constant
GAS_CONSTANT = 8.314
#: [kg m\ :sup:`2` s\ :sup:`-1`\ ], standard gravity
GRAVITY = 9.81
#: [kg m\ :sup:`-3`\ ], water density
WATER_DENSITY = 1.0e3


def net_co2_exchange(para, Qp, Ca, T, w, wstar):
    """
    Computes net CO2 exchange of moss community.
    Args:
        - 'para' (dict): parameters
            - 'Vcmax' (float): maximum carboxylation velocity at 25 C [mol m\ :sup:`-2` (ground) s\ :sup:`-1`]
            - 'Jmax' (float): maximim electron transport rate at 25 C [mol m\ :sup:`-2` (ground) s\ :sup:`-1`]
            - 'Rd' (float): dark respiration rate at 25 C [mol m\ :sup:`-2` (ground) s\ :sup:`-1`]
            - 'alpha' (float): quantum efficiency [-]
            - 'theta' (float): curvature parameter [-]
            - 'beta' (float): co-limitation parameter [-]
            
            - 'gmax' (float): conductance for CO2 at optimum water content [mol m\ :sup:`-2` s\ :sup:`-1`] 
            - 'wopt' (flot): parameter of conductance - water content relationship [g g\ :sup:`-1`\ ]
            - 'a0' (float): parameter of conductance - water content curve [-]
            - 'a1' (float): parameter of conductance - water content curve [-]
            - 'CAP_desic' (list): parameters (float) of desiccation curve [-]
            - 'tresp' (dict): parameters of photosynthetic temperature response curve
                - Vcmax (list): [activation energy [kJ  mol\ :sup:`-1`], 
                                 deactivation energy [kJ  mol\ :sup:`-1`]
                                 entropy factor [kJ  mol\ :sup:`-1`]]
                                ]
                - Jmax (list): [activation energy [kJ  mol\ :sup:`-1`], 
                                 deactivation energy [kJ  mol\ :sup:`-1`]
                                 entropy factor [kJ  mol\ :sup:`-1`]]
                                ]
                - Rd (list): [activation energy [kJ  mol\ :sup:`-1`]]
                
        - 'Qp' (float): incident PAR [umol m-2 s-1]
        - 'Ca' (float): ambient CO2 [ppm]
        - 'T' (float): temperature [degC]
        - 'w' (float): gravimetric water content [g/g]
        - 'wstar' (float): delayed water content [g/g] for desiccation recovery
    Returns:
        - 'An' (float): net CO2 exchange rate, An = -A + Rd, <0 is uptake [umol m\ :sup:`-2`\ (ground) s\ :sup:`-1`\]
        - 'A' (float): photosynthesis rate [umol m\ :sup:`-2`\ (ground) s\ :sup:`-1`\]
        - 'Rd' (float): dark respiration rate [umol m\ :sup:`-2`\ (ground) s\ :sup:`-1`\]
        - 'Cc' (float): intercellular CO2 (ppm)
        - 'g' (float): conductance for CO2 [umol m\ :sup:`-2`\ (ground) s\ :sup:`-1`\]
    """
    p = para.copy()
    
    # moisture effect on parameters
    cap, rcap = relative_capacity(p, w, wstar)
    p['Vcmax'] *= cap 
    p['Jmax'] *= cap
    p['alpha'] *= cap
    p['Rd'] *= rcap
    
    # conductance (mol m-2 s-1)
    g = conductance(p, w)
    
    # solve Anet and Cc iteratively until Cc converges
    err = 10^9
    Cc = 0.8*Ca

    while err > 1e-3:
        Cco = Cc
        An, Rd, Av, Aj = photo_farquhar(p, Qp, Cc, T)
        
        Cc = Ca - An / g  # new Cc
        Cc = 0.5*(Cco + Cc)
        err = np.nanmax(abs(Cc - Cco))
    return -An, An - Rd, Rd, Cc, g
    

def conductance(para, w):
    """
    Conductance for CO2 diffusion from bulk air to chloroplast in bryophyte.
    Assumes g = gmax * fw, where gmax is species-specific maximum internal conductance, 
    occurring at w <= wopt, and fw [-] describes decay of conductance due to
    external water (w > wopt).
    
    gmax and wopt are bryophyte traits, while the shape of fw is fixed based on
    data of Williams & Flanagan, 1998. We normalized maximum conductance for Pleurozium and
    Sphagnum to unity, and determined respective wopt's. Then a decreasing exponential 
    function was fitted to data.
    
    Args:
        - 'para' (dict):
            - 'gmax' (float): conductance [mol m\ :sup:`-2` (ground) s\ :sup:`-1`] at wopt
            - 'wopt' (float): scaling water content [g g\ :sup:`-1`]
            - 'a0' (float): shape parameter [-]
            - 'a1' (float): shape parameter [-]
        - 'w' (float): gravimetric water content [g g\ :sup:`-1`]
    Returns:
        - 'g' (float): air - reaction site conductance for CO2 [molm\ :sup:`-2` (ground) s\ :sup:`-1`]
    """
 
    gmax = para['gmax']
    a0 = para['a0']
    a1 = para['a1']
    wopt = para['wopt']
    
    #g = gmax * np.minimum(1.0, a0*np.exp(a1*(w-wopt)) + (1.0 - a0))
    g = gmax * (a0*np.exp(a1*(w-wopt)) + (1.0 - a0)) # this allows g to go above gmax in dry moss
    return g


def relative_capacity(para, w, wstar):
    """
    Relative photosynthetic capacity and dark respiration rate as a function of water content.
    
    Args:
        - 'para' (dict): parameter dictionary
        - 'w' (float): current gravimetric water content [g g\ :sup:`-1`]
        ' 'wstar' - delayed effective water content for desiccation recovery  [g g\ :sup:`-1`]
    Returns:
         - 'rphoto' (float): photosynthetic capacity relative to well-hydrated state [-]
         ' 'rrd' (float): dark respiration rate relative to well-hydrated state [-]
    """
    
    p = para['CAP_desic']
    
    # r = para['CAP_rewet']
    
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

def photo_farquhar(photop, Qp, Ci, T, co_limi=False):
    """
    Calculates moss photosynthesis, dark respiration and net CO2 exhange.
    
    Args:
        - 'photop' (dict):
        - 'Vcmax' (float): maximum carboxylation velocity at 25 C [mol m\ :sup:`-2` (ground) s\ :sup:`-1`]
        - 'Jmax' (float): maximim electron transport rate at 25 C [mol m\ :sup:`-2` (ground) s\ :sup:`-1`]
        - 'Rd' (float): dark respiration rate at 25 C [mol m\ :sup:`-2` (ground) s\ :sup:`-1`]
        - 'alpha' (float): quantum efficiency [-]
        - 'theta' (float): curvature parameter [-]
        - 'beta' (float): co-limitation parameter [-]
        - 'tresp' (dict): parameters of photosynthetic temperature response curve
            - 'Vcmax' (list): [activation energy [kJ  mol\ :sup:`-1`], 
                             deactivation energy [kJ  mol\ :sup:`-1`]
                             entropy factor [kJ  mol\ :sup:`-1`]]
                            ]
            - 'Jmax' (list): [activation energy [kJ  mol\ :sup:`-1`], 
                             deactivation energy [kJ  mol\ :sup:`-1`]
                             entropy factor [kJ  mol\ :sup:`-1`]]
                            ]
            - 'Rd' (list): [activation energy [kJ  mol\ :sup:`-1`]]
                
        - 'Qp' (float): incident PAR [umol m-2 s-1]
        - 'Ci' (float): intercellular CO2 [ppm]
        - 'T' (float): temperature [degC]

    Returns:
       - 'An' (float): net CO2 exchange rate, An = A - Rd [umol m\ :sup:`-2`\ (ground) s\ :sup:`-1`\] >0 is uptake
       - 'Rd' (float): dark respiration rate [umol m\ :sup:`-2`\ (ground) s\ :sup:`-1`\]
       - 'Av' (float): rubisco limited rate [umol m\ :sup:`-2`\ (ground) s\ :sup:`-1`\] 
       - 'Aj' (float): RuBP -regeneration limited rate [umol m\ :sup:`-2`\ (ground) s\ :sup:`-1`\] 
        
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


def photo_temperature_response(Vcmax25, Jmax25, Rd25, Vcmax_T, Jmax_T, Rd_T, T):
    """
    Adjusts Farquhar-parameters for temperature
    
    Args:
        - 'Vcmax25' (float): maximum carboxylation velocity at 25 degC (298 K)  [umol m\ :sup:`-2`\ (ground) s\ :sup:`-1`\] 
        - 'Jmax25' (float): maximum electron transport rate at 25 degC  [umol m\ :sup:`-2`\ (ground) s\ :sup:`-1`\] 
        - 'Rd25' (float): dark respiration rate at 25 degC  [umol m\ :sup:`-2`\ (ground) s\ :sup:`-1`\] 
        - 'Vcmax_T' (list): [activation energy [kJ  mol\ :sup:`-1`], 
                           deactivation energy [kJ  mol\ :sup:`-1`]
                           entropy factor [kJ  mol\ :sup:`-1`]]
                         ]
        - 'Jmax_T' (list): [activation energy [kJ  mol\ :sup:`-1`], 
                          deactivation energy [kJ  mol\ :sup:`-1`]
                          entropy factor [kJ  mol\ :sup:`-1`]]
                         ]
       - 'Rd_T' (list): [activation energy [kJ  mol\ :sup:`-1`]]
       - 'T' (float or array): temperature [K]
    Returns:
        - 'Vcmax' (float or array): at temperature T [umol m\ :sup:`-2`\ (ground) s\ :sup:`-1`\] 
        - 'Jmax' (float or array): at temperature T  [umol m\ :sup:`-2`\ (ground) s\ :sup:`-1`\] 
        - 'Rd' (float or array): at temperature T  [umol m\ :sup:`-2`\ (ground) s\ :sup:`-1`\] 
        - 'Gamma_star' (float or array): CO2 compensation point at T [ppm]
   
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



def topt_deltaS_conversion(Ha, Hd, dS=None, Topt=None):
    """
    Converts between entropy factor Sd [kJ mol \ :sup:`-1`\] and temperature optimum
    Topt [k]. Medlyn et al. 2002 PCE 25, 1167-1179 eq.19.
    
    Args:
        - 'Ha' (float): activation energy [kJ  mol\ :sup:`-1`]
        - 'Hd' (float): deactivation energy [kJ  mol\ :sup:`-1`]
        - 'dS' (float): entropy factor [kJ  mol\ :sup:`-1`]
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


def draw_farquhar_curves():
    Vcmax = 30.0
    Jmax = 1.9*Vcmax
    Rd = 0.15*Vcmax
    tresp = {'Vcmax': [69.83, 200.0, 650.],
             'Jmax': [100.28, 147.92,650],
             'Rd': [33.0], 'include': 'y'}
    
    photop = {'Vcmax': Vcmax, 'Jmax': Jmax, 'Rd': Rd, # umolm-2s-1
              'alpha': 0.24, 'theta': 0.8, 'beta': 0.95, # quantum yield, curvature, co-limitation
              'gmax': 0.06, 'wopt': 7.0, 'a0': 0.7, 'a1': -0.263, 'CAP_desic': [0.53, 7.0],
              'tresp': tresp}

    Qp = np.linspace(0.0, 1600.0, 100)
    cc = 0.8*400
    T = 25.0
    
    An, Rd, Av, Aj = photo_farquhar(photop, Qp, cc, T, co_limi=False)
    photop['beta'] = 1.0
    An1, Rd1, _, _ = photo_farquhar(photop, Qp, cc, T, co_limi=False)

    plt.figure(1)
    # plt.subplot(121)
    plt.plot(Qp, An+Rd, 'k-', Qp, Aj, 'b--', Qp[[0,-1]], [Av,Av], 'r--', Qp, An1+Rd1, 'g--')
    plt.legend(['A', 'A_j', 'A_v', 'A ($\\beta$=1.0)'])
    plt.ylabel('A ($\mu$mol m$^{-2}$ s$^{-1}$)')
    plt.xlabel('absorbed PAR ($\mu$mol m$^{-2}$ s$^{-1}$)')
    plt.title('$C_c=320\, ppm, V_{c,max}=30, J_{max}=1.9\,V_{c,max}, \gamma=0.24, \\theta=0.8, \\beta=0.98, T=25degC$', fontsize=10)


def test():
    
    Vcmax = 15.0
    Jmax = 1.9*Vcmax
    Rd = 0.05*Vcmax
    tresp = {'Vcmax': [69.83, 200.0, 650.],
             'Jmax': [100.28, 147.92, 650.],
             'Rd': [33.0], 'include': 'y'}
    
    # pleurozium type
    photop_p = {'Vcmax': Vcmax, 'Jmax': Jmax, 'Rd': Rd, # umolm-2s-1
              'alpha': 0.3, 'theta': 0.8, 'beta': 0.9, # quantum yield, curvature, co-limitation
              'gmax': 0.02, 'wopt': 7.0, 'a0': 0.7, 'a1': -0.263, 'CAP_desic': [0.44, 7.0],
              'tresp': tresp}

    # sphagnum type
    Vcmax = 45.0
    photop_s = {'Vcmax': Vcmax, 'Jmax': 1.9*Vcmax, 'Rd': 0.03*Vcmax, # umolm-2s-1
              'alpha': 0.3, 'theta': 0.8, 'beta': 0.9, # quantum yield, curvature, co-limitation
              'gmax': 0.04, 'wopt': 7.0, 'a0': 0.7, 'a1': -0.263, 'CAP_desic': [0.58, 10.0],
              'tresp': tresp}        
    
    # moisture levels
    w = np.linspace(1, 20 , 50)
    wstar = w.copy()
    
    # make few response plots
    
    # moisture response at 400 and 600 ppm and at light-saturated cond.
    #Qp = np.linspace(0.0, 100, 100)
    Qp = 500.0
    T = 20.0
    fig, ax = plt.subplots(ncols=2, nrows=2)
    
    style=['-', '--', '-.']
    k = 0
    for Ca in [400.0, 800.0]:        
        for gmax in [0.03, 0.06]:
            photop_p['gmax'] = gmax
            photop_s['gmax'] = gmax
            A, An, Rd, Cc, g = net_co2_exchange(photop_p, Qp, Ca, T, w, wstar)
            ax[0,0].plot(w, An, linestyle=style[k], label='Ca=' + str(Ca) + ', gmax=' + str(gmax))
            ax[1,0].plot(w, Cc/Ca, linestyle=style[k])
            
            A, An, Rd, Cc, g = net_co2_exchange(photop_s, Qp, Ca, T, w, wstar)
            ax[0,1].plot(w, An, linestyle=style[k], label='Ca=' + str(Ca) + ', gmax=' + str(gmax))
            ax[1,1].plot(w, Cc/Ca, linestyle=style[k])
        k += 1

    ax[0,0].set_title('"Pleurozium"'); 
    ax[0,0].set_ylabel('A umolm-2s-1'); ax[0,0].set_xlabel('w (g/g)')
    ax[1,0].set_ylabel('Cc/Ca'); ax[1,0].set_xlabel('w (g/g)')
    
    ax[0,1].set_title('"Sphagnum"');
    ax[0,1].set_xlabel('w (g/g)')
    ax[1,1].set_xlabel('w (g/g)')
    ax[0,0].legend()
    ax[0,1].legend()
    