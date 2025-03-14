import numpy as np

from pyAPES.utils.constants import H2_18O_DIFFYSIVITY, H2O_MOLARDENSITY, EPS, DEG_TO_KELVIN

TN = 25.0 + DEG_TO_KELVIN

def oxygen_enrichment(leaf_gas_exchange, forcing,
                      e_kb=19., e_k=28., L_eff=30e-3, f1=0.9):
    """
    Calculates leaf water 18O-enrichment
    calculated with 3 alternative approaches.

    Assumes vapor-source water isotopic equilibrium, i.e. D18O_vapor = -e_star

    Args:
        leaf_gas_exchange (dict):
            'transpiration': H2O flux (transpiration) (mol m-2 leaf s-1)
            'leaf_temperature': leaf temperature (degC)
            'stomatal_conductance': stomatal conductance for H2O (mol m-2 leaf s-1)
            'boundary_conductance': boundary layer conductance for H2O (mol m-2 leaf s-1)
        forcing (dict):
            'h2o' (array): water vapor mixing ratio (mol mol-1)
            'air_pressure' (float): ambient pressure (Pa)
        e_kb (float, optional): fractionation during diffusion of water vapor
            through boundary layer (permil). Defaults to 19.
        e_k (float, optional): fractionation during diffusion of water vapor
            through stomata (permil). Defaults to 28.
        L_eff (float, optional): leaf mesophyll effective mixing length for
            peclet model (m). Defaults to 30e-3.
        f1 (float, optional): ratio of enriched to total needle water for
            two-pool model. Defaults to 0.9.

    Returns:
        leaf_18O_enrichment  (dict):
            'D18O_CG': (permil)
            'D18O_peclet': (permil)
            'D18O_2pool': (permil)

    """

    E = leaf_gas_exchange['transpiration']
    gb = leaf_gas_exchange['boundary_conductance']
    gs = leaf_gas_exchange['stomatal_conductance']

    wa = forcing['h2o']
    P = forcing['air_pressure']
    T = leaf_gas_exchange['leaf_temperature']

    # mole fraction of water vapor inside leaf, assuming fully saturated
    esat = 611.0 * np.exp((17.502 * T) / (T + 240.97))  # Pa
    wi = esat / P

    e_star = 1e3*(np.exp(1137/(T + DEG_TO_KELVIN)**2 - 0.4156 /(T + DEG_TO_KELVIN) - 0.0020667) - 1)

    e_kkb = (e_k * gb + e_kb * gs) / (gb + gs)

    # leaf water enrichment
    # evaporative sites (Craig-Gordon model)
    D18O_e = e_star + e_kkb - (e_star + e_kkb) * wa / wi
    # peclet model
    p = p = np.maximum(EPS, L_eff * 0.5 * E /  # 0.5E is for all-sided leaf area
                       (H2O_MOLARDENSITY * H2_18O_DIFFYSIVITY))
    D18O_lw_peclet = D18O_e * (1 - np.exp(-p)) / p
    # two pool model
    D18O_lw_2pool = D18O_e * f1

    #Collect model results to dictionary
    leaf_18O_enrichment = {
        'D18O_CG': D18O_e,
        'D18O_peclet': D18O_lw_peclet,
        'D18O_2pool': D18O_lw_2pool
        }

    return leaf_18O_enrichment
