import numpy as np

from pyAPES.utils.constants import DEG_TO_KELVIN, GAS_CONSTANT
# reference temperature [K]
TN = 25.0 + DEG_TO_KELVIN


def carbon_discrimination(leaf_gas_exchange, forcing,
                          a_b=2.9 ,a_s=4.4, a_m=1.8, b_simple=27.,
                          b_classical=29., f=8., e=-6., gm=0.1):
    """
    Calculates 13C-discrimination by photosynthesis based on 3 alternative approaches.

    Args:
        leaf_gas_exchange (dict):
            'net_co2': net CO2 flux (umol m-2 leaf s-1)
            'dark_respiration': CO2 respiration (umol m-2 leaf s-1)
            'leaf_temperature': leaf temperature (degC)
            'leaf_internal_co2': leaf internal CO2 mixing ratio (umol/mol)
            'leaf_surface_co2': leaf surface CO2 mixing ratio (umol/mol)
            'leaf_chloropast_co2': leaf chloroplast CO2 mixing ratio (umol/mol)
        forcing (dict):
            'co2': ambient CO2 mixing ratio (umol/mol)
        a_b (float, optional): fractionation during diffusion through the
            boundary layer (permil). Defaults to 2.9.
        a_s (float, optional): fractionation during diffusion through the
            stomata (permil). Defaults to 4.4.
        a_m (float, optional): fractionation during transfer through mesophyll
            (permil). Defaults to 1.8.
        b_simple (float, optional): fractionation during carboxylation (permil).
            Defaults to 27.
        b_classical (float, optional): fractionation during carboxylation
            (permil). Defaults to 29.
        f (float, optional): fractionation during mitochondrial respiration
            (permil). Defaults to 8.
        e (float, optional): fractionation during photorespiration (permil).
            Defaults to -6.
        gm (float, optional): mesophyll conductance for CO2 (mol m-2 s-1).
            Defaults to 0.25.

    Returns:
        leaf_13C_discrimination (dict):
            'D13C_simple': (permil)
            'D13C_classical': (permil)
            'D13C_classical_gm': (permil)
    """

    An = leaf_gas_exchange['net_co2']
    Rd = leaf_gas_exchange['dark_respiration']

    ca = forcing['co2']
    cs = leaf_gas_exchange['leaf_surface_co2']
    ci = leaf_gas_exchange['leaf_internal_co2']

    T = leaf_gas_exchange['leaf_temperature']
    Tk = T + DEG_TO_KELVIN
    
    # CO2 compensation point
    Tau_c = 42.75 * np.exp(37830*(Tk - TN) / (TN * GAS_CONSTANT * Tk))

    D13C_simple = a_s * (ca - ci)/ca + b_simple * ci/ca

    k = (An + Rd) / (ci - Tau_c)
    k[k==0] = np.nan # For non leaf elements An+Rd == 0, remove these to suppress divide by zero warning
    D13C_classical = (a_b * (ca - cs)/ca + a_s * (cs - ci)/ca
                    + b_classical * ci/ca - f * Tau_c/ca - e * Rd / (k * ca))

    if leaf_gas_exchange['leaf_chloroplast_co2'] is None:
        gm = 0.1 # Assume mesophyll conductance of 0.1 mol/m2 (leaf)/s to calculate D13c classical gm
        cc = ci - An / gm
    else:
        cc = leaf_gas_exchange['leaf_chloroplast_co2']

    # print(cc)
    k = (An + Rd) / (cc - Tau_c)
    k[k==0] = np.nan
    D13C_classical_gm = (a_b * (ca - cs)/ca + a_s * (cs - ci)/ca + a_m * (ci - cc)/ca
                       + b_classical * cc/ca - f * Tau_c/ca - e * Rd / (k * ca))

    leaf_13C_discrimination = {
        'D13C_simple': D13C_simple,
        'D13C_classical': D13C_classical,
        'D13C_classical_gm': D13C_classical_gm
        }

    return leaf_13C_discrimination