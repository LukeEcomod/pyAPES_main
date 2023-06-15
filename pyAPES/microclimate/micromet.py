# -*- coding: utf-8 -*-
"""
.. module: micromet
    :synopsis: pyAPES-model microclimate component
.. moduleauthor:: Samuli Launiainen, Kersti Leppä, Gaby Katul

Describes turbulent flow and momentum & scalar transport within canopy.

References:
    Launiainen, S., Katul, G.G., Lauren, A. and Kolari, P., 2015. Coupling boreal
    forest CO2, H2O and energy flows by a vertically structured forest canopy –
    Soil model with separate bryophyte layer. Ecological modelling, 312, pp.385-405.

    Juang, J.-Y., Katul, G.G., Siqueira, M.B., Stoy, P.C., McCarthy, H.R., 2008.
    Investigating a hierarchy of Eulerian closure models for scalar transfer inside
    forested canopies. Boundary-Layer Meteorology 128, 1–32.
    
    Campbell, S.C., and J.M. Norman. 1998. An introduction to Environmental Biophysics, 
    Springer, 2nd edition.
"""
import numpy as np
from typing import Dict, List, Tuple
import logging

from pyAPES.utils.utilities import central_diff, forward_diff, tridiag, smooth, spatial_average
from pyAPES.utils.constants import EPS, VON_KARMAN, GRAVITY, MOLECULAR_DIFFUSIVITY_CO2, MOLECULAR_DIFFUSIVITY_H2O, \
THERMAL_DIFFUSIVITY_AIR, AIR_VISCOSITY, MOLAR_MASS_AIR, SPECIFIC_HEAT_AIR, DEG_TO_KELVIN

logger = logging.getLogger(__name__)

class Micromet(object):
    r""" 
    Horizontal mean flow and scalar profiles within horizontally homogeneous multi-layer canopy
    """
    def __init__(self, z: np.ndarray, lad: np.ndarray, hc: float, p: Dict):
        r""" 
        Initializes micromet object

        Args:
            z (array): canopy model nodes, equidistance, height from soil surface (= 0.0) [m]
            lad (array): leaf area density [m2 m-3]
            hc (float): canopy heigth [m]
            p (dict):
                'zos': forest floor roughness length [m]
                'dPdx': horizontal pressure gradient
                'Cd': drag coefficient
                'Utop': ensemble U/ustar [-]
                'Ubot': U or U/ustar at the lower boundary
                'Sc' (dict): {'T','H2O','CO2'}, turbulent Schmidt numbers [-]
        Returns:
            self (object)
        """

        # parameters
        self.zos = p['zos']  # forest floor roughness length [m]

        self.dPdx = p['dPdx']  # horizontal pressure gradient
        self.Cd = p['Cd']  # drag coefficient
        self.Utop = p['Utop']  # ensemble U/ustar
        self.Ubot = p['Ubot']  # lower boundary
        self.Sc = p['Sc']  # Schmidt numbers
        self.dz = z[1] - z[0]

        # initialize state variables
        self.tau, self.U_n, self.Km_n, _, _, _ = closure_1_model_U(
                z, self.Cd, lad, hc, self.Utop + EPS, self.Ubot, dPdx=self.dPdx)

    def normalized_flow_stats(self, z: np.ndarray, lad: np.ndarray, hc: float, Utop: float=None):
        r"""
        Computes normalized mean velocity, shear stress and eddy diffusivity profiles within and above 
        horizontally homogenous plant canopies using 1st order closure schemes.

        Args:
            z (array): canopy model nodes, height from soil surface (= 0.0) [m]
            lad (array): leaf area density [m2 m-3]
            hc (float): canopy heigth [m]
            Utop (float): U/ustar [-], if None, set to self.Utop
        Returns:
            None: updates self.U, self.Km_n, self.tau
        """
        if Utop is None:
            Utop = self.Utop

        tau, U_n, Km_n, _, _, _ = closure_1_model_U(
                z, self.Cd, lad, hc, Utop + EPS, self.Ubot, dPdx=self.dPdx, U_ini=self.U_n)

        if any(U_n < 0.0):
            logger.debug('Negative U_n, set to previous profile.')
        else:
            self.U_n = U_n.copy()
            self.Km_n = Km_n.copy()
            self.tau = tau.copy()

    def update_state(self, ustaro: float) -> Tuple:
        r""" 
        Updates mean wind speed, ustar and eddy-diffusivity profile.
        Args:
            ustaro (float): friction velocity at uppermost grid-point [m s-1]
        Returns:
            U (array): mean wind speed [m s-1]
            ustar (array): friction velocity [m s-1]
        """

        U = self.U_n * ustaro + EPS
        U[0] = U[1]

        Km = self.Km_n * ustaro + EPS
        Km[0] = Km[1]
        self.Km = Km

        ustar = np.sqrt(abs(self.tau)) * ustaro

        return U, ustar

    def scalar_profiles(self, gam: float, H2O: np.ndarray, CO2: np.ndarray, T: np.ndarray, 
                        P: float, source: Dict, lbc: Dict, Ebal: bool) -> Tuple:
        r""" 
        Solves scalar profiles (H2O, CO2 and T) within the canopy using 1st order closure scheme.

        Args:
            gam (float): weight for new value in iterations
            H2O (array): water vapor mixing ratio [mol mol-1]
            CO2 (array): carbon dioxide mixing ratio [ppm]
            T (array): ambient air temperature [degC]
            P (float): ambient pressure [Pa]
            source (dict):
                'H2O' (array): water vapor source [mol m-3 s-1]
                'CO2' (array): carbon dioxide source [umol m-3 s-1]
                'T' (array): heat source [W m-3]
            lbc (dict):
                'H2O' (float): water vapor lower boundary [mol m-2 s-1]
                'CO2' (float): carbon dioxide lower boundary [umol m-2 s-1]
                'T' (float): heat lower boundary [W m-2]
        Returns:
            H2O (array): water vapor mixing ratio [mol mol-1]
            CO2 (array): carbon dioxide mixing ratio [ppm]
            T (array): ambient air temperature [degC]
            err_h2o (float): maximum error for H2O
            err_co2 (float): -"- CO2
            err_t (float): -"- T
        """

        # previous guess, not values of previous time step!
        H2O_prev = H2O.copy()
        CO2_prev = CO2.copy()
        T_prev = T.copy()

        # --- H2O ---
        H2O = closure_1_model_scalar(dz=self.dz,
                                     Ks=self.Km * self.Sc['H2O'],
                                     source=source['h2o'],
                                     ubc=H2O[-1],
                                     lbc=lbc['H2O'],
                                     scalar='H2O',
                                     T=T[-1], P=P)
        # new H2O
        H2O = (1 - gam) * H2O_prev + gam * H2O
        # limit change to +/- 10%
        if all(~np.isnan(H2O)):
            H2O[H2O > H2O_prev] = np.minimum(H2O_prev[H2O > H2O_prev] * 1.1, H2O[H2O > H2O_prev])
            H2O[H2O < H2O_prev] = np.maximum(H2O_prev[H2O < H2O_prev] * 0.9, H2O[H2O < H2O_prev])
        
        # relative error
        err_h2o = max(abs((H2O - H2O_prev) / H2O_prev))

        # --- CO2 ---
        CO2 = closure_1_model_scalar(dz=self.dz,
                                     Ks=self.Km * self.Sc['CO2'],
                                     source=source['co2'],
                                     ubc=CO2[-1],
                                     lbc=lbc['CO2'],
                                     scalar='CO2',
                                     T=T[-1], P=P)
        # new CO2
        CO2 = (1 - gam) * CO2_prev + gam * CO2
        # limit change to +/- 10%
        if all(~np.isnan(CO2)):
            CO2[CO2 > CO2_prev] = np.minimum(CO2_prev[CO2 > CO2_prev] * 1.1, CO2[CO2 > CO2_prev])
            CO2[CO2 < CO2_prev] = np.maximum(CO2_prev[CO2 < CO2_prev] * 0.9, CO2[CO2 < CO2_prev])
        
        # relative error
        err_co2 = max(abs((CO2 - CO2_prev) / CO2_prev))

        if Ebal:
            # --- T ---
            T = closure_1_model_scalar(dz=self.dz,
                                       Ks=self.Km * self.Sc['T'],
                                       source=source['sensible_heat'],
                                       ubc=T[-1],
                                       lbc=lbc['T'],
                                       scalar='T',
                                       T=T[-1], P=P)
            # new T
            T = (1 - gam) * T_prev + gam * T
            # limit change to T_prev +/- 2degC
            if all(~np.isnan(T)):
                T[T > T_prev] = np.minimum(T_prev[T > T_prev] + 2.0, T[T > T_prev])
                T[T < T_prev] = np.maximum(T_prev[T < T_prev] - 2.0, T[T < T_prev])

            # absolute error
            err_t = max(abs(T - T_prev))
        else:
            err_t = 0.0

        return H2O, CO2, T, err_h2o, err_co2, err_t

def closure_1_model_U(z: np.ndarray, Cd: float, lad: np.ndarray, hc: float, 
                      Utop: float, Ubot: float, dPdx: float=0.0, lbc_flux: bool=None, 
                      U_ini: np.array=None) -> Tuple:
    """
    Computes mean velocity profile, shear stress and eddy diffusivity
    within and above horizontally homogenous plant canopies using 1st order closure.
    Accounts for horizontal pressure gradient force dPdx, assumes neutral diabatic stability.
    Solves displacement height as centroid of drag force.
    
    Args:
       z - height [m]], constant increments
       Cd - drag coefficient (typical range 0.1 - 0.3) [-]]
       lad - plant area density, 1-sided [m2 m-3]
       hc - canopy height [m]
       Utop - U /u* [-] upper boundary
       Uhi - U /u* [-]at ground (0.0 for no-slip)
       dPdx - u* -normalized horizontal pressure gradient
    
    Returns:
       tau (array): u* -normalized momentum flux
       U (array): u* normalized mean wind speed [-]]
       Km (array): eddy diffusivity for momentum [m2 s-1]
       l_mix (array): mixing length [m]]
       d (float): zero-plane displacement height [m]
       zo (float): roughness lenght for momentum [m]]
    """

    lad = 0.5*lad  # frontal plant-area density is half of one-sided
    dz = z[1] - z[2]
    N = len(z)
    if U_ini is None:
        U = np.linspace(Ubot, Utop, N)
    else:
        U = U_ini.copy()

    nn1 = max(2, np.floor(N/20))  # window for moving average smoothing

    # --- Start iterative solution
    err = 999.9
    iter_max = 20
    eps1 = 0.5
    dPdx_m = 0.0

    iter_no = 0.0

    while err > 0.01 and iter_no < iter_max:
        iter_no += 1
        Fd = Cd*lad*U**2  # drag force
        d = sum(z*Fd) / (sum(Fd) + EPS)  # displacement height
        l_mix = mixing_length(z, hc, d)  # m

        # --- dU/dz [m-1]
        y = central_diff(U, dz)

        # --- eddy diffusivity & shear stress
        Km = l_mix**2*abs(y)
        tau = -Km * y

        # ------ Set the elements of the Tri-diagonal Matrix
        a1 = -Km
        a2 = central_diff(-Km, dz)
        a3 = Cd*lad*U

        upd = (a1 / (dz*dz) + a2 / (2*dz))  # upper diagonal
        dia = (-a1*2 / (dz*dz) + a3)  # diagonal
        lod = (a1 / (dz*dz) - a2 / (2*dz))  # subdiagonal
        rhs = np.ones(N) * dPdx  #_m ???

        # upper BC
        upd[-1] = 0.
        dia[-1] = 1.
        lod[-1] = 0.
        rhs[-1] = Utop

        if not lbc_flux:  # --- lower BC, fixed Ubot
            upd[0] = 0.
            dia[0] = 1.
            lod[0] = 0.
            rhs[0] = Ubot
        else:  # --- lower BC, flux-based
            upd[0] = -1.
            dia[0] = 1.
            lod[0] = 0.
            rhs[0] = 0.  # zero-flux bc
            # rhs[0] = lbc_flux

        # --- call tridiagonal solver
        Un = tridiag(lod, dia, upd, rhs)

        err = max(abs(Un - U))

        # --- Use successive relaxations in iterations
        U = eps1*Un + (1.0 - eps1)*U
        dPdx_m = eps1*dPdx + (1.0 - eps1)*dPdx_m  # ???
        if iter_no == iter_max:
            logger.debug('Maximum number of iterations reached: U_n = %.2f, err = %.2f',
                         np.mean(U), err)

    # ---- return values
    tau = tau / tau[-1]  # normalized shear stress
    zo = (z[-1] - d)*np.exp(-0.4*U[-1])  # roughness length

    y = forward_diff(U, dz)
    Kmr = l_mix**2 * abs(y)  # eddy diffusivity
    Km = smooth(Kmr, nn1)

    # --- for testing ----
#    plt.figure(101)
#    plt.subplot(221); plt.plot(Un, z, 'r-'); plt.title('U')
#    plt.subplot(222); plt.plot(y, z, 'b-'); plt.title('dUdz')
#    plt.subplot(223); plt.plot(l_mix, z, 'r-'); plt.title('l mix')
#    plt.subplot(224); plt.plot(Km, z, 'r-', Kmr, z, 'b-'); plt.title('Km')

    return tau, U, Km, l_mix, d, zo

def closure_1_model_scalar(dz: float, Ks: np.ndarray, source: np.ndarray, ubc: float, lbc: float, 
                           scalar: str, T: float=20.0, P: float=101300.0, lbc_dirchlet=False) -> np.ndarray:
    r""" 
    Solves stedy-state scalar profiles in 1-D grid using 1st order closure.
    Args:
        dz (float): grid size (m)
        Ks (array): eddy diffusivity [m2 s-1]
        source (array): sink/source term
            CO2 [umol m-3 s-1], H2O [mol m-3 s-1], T [W m-3]
        ubc (float):upper boundary condition
            value: CO2 [ppm], H2O [mol mol-1], T [degC]
        lbc (float): lower boundary condition, flux or value:
            flux: CO2 [umol m-2 s-1], H2O [mol m-2 s-1], T [W m-2].
            value: CO2 [ppm], H2O [mol mol-1], T [degC]
        scalar (str): 'CO2', 'H2O', 'T'
        T (float): air temperature [degC], for computing air molar density
        P (float): pressure [Pa]
        lbc_dirchlet - True for Dirchlet (fixed value) lower boundary condition
    OUT:
        x (array): mixing ratio profile
            CO2 [ppm], H2O [mol mol-1], T [degC]
    References:
        Juang, J.-Y., Katul, G.G., Siqueira, M.B., Stoy, P.C., McCarthy, H.R., 2008.
        Investigating a hierarchy of Eulerian closure models for scalar transfer inside
        forested canopies. Boundary-Layer Meteorology 128, 1–32.
    Code:
        Gaby Katul & Samuli Launiainen, 2009 - 2017
        Kersti: code condensed, discretization simplified (?)
    Note: assumes constant dz and Dirchlet upper boundary condition
    """

    dz = float(dz)
    N = len(Ks)
    rho_a = P / (287.05 * (T + 273.15))  # [kg m-3], air density

    CF = rho_a / MOLAR_MASS_AIR  # [mol m-3], molar conc. of air

    Ks = spatial_average(Ks, method='arithmetic')  # length N+1

    if scalar.upper() == 'CO2':  # [umol] -> [mol]
        ubc = 1e-6 * ubc
        source = 1e-6 * source
        lbc = 1e-6 * lbc

    if scalar.upper() == 'T':
        CF = CF * SPECIFIC_HEAT_AIR  # [J m-3 K-1], volumetric heat capacity of air

    # --- Set elements of tridiagonal matrix ---
    a = np.zeros(N)  # sub diagonal
    b = np.zeros(N)  # diagonal
    g = np.zeros(N)  # super diag
    f = np.zeros(N)  # rhs

    # intermediate nodes
    a[1:-1] = Ks[1:-2]
    b[1:-1] = - (Ks[1:-2] + Ks[2:-1])
    g[1:-1] = Ks[2:-1]
    f[1:-1] = - source[1:-1] / CF * dz**2

    # uppermost node, Dirchlet boundary
    a[-1] = 0.0
    b[-1] = 1.0
    g[-1] = 0.0
    f[-1] = ubc

    # lowermost node
    if not lbc_dirchlet:  # flux-based
        a[0] = 0.0
        b[0] = 1.
        g[0] = -1.
        f[0] = (lbc / CF)*dz / (Ks[1] + EPS)

    else:  #  fixed concentration/temperature
        a[0] = 0.0
        b[0] = 1.
        g[0] = 0.0
        f[0] = lbc

    x = tridiag(a, b, g, f)

    if scalar.upper() == 'CO2':  # [mol] -> [umol]
        x = 1e6*x

    return x

def mixing_length(z: np.ndarray, h: float, d: float, l_min: float=None) -> np.ndarray:
    """
    Computes turbulend mixing length.
    l_mix is assumed linear above the canopy, constant within and
    decreases linearly close the ground (below z< alpha*h/VON_KARMAN)
    Args:
        z (array): [m], computation grid, constant increment
        h (float): [m], canopy height
        d (float): [m], displacement height
        l_min (float): [m], set to finite value at ground
    OUT:
        lmix (array): [m], turbulent mixing length
    """
    dz = z[1] - z[0]

    if not l_min:
        l_min = dz / 2.0

    alpha = (h - d)*VON_KARMAN / (h + EPS)
    I_F = np.sign(z - h) + 1.0
    l_mix = alpha*h*(1 - I_F / 2) + (I_F / 2) * (VON_KARMAN*(z - d))

    sc = (alpha*h) / VON_KARMAN
    ix = np.where(z < sc)
    l_mix[ix] = VON_KARMAN*(z[ix] + dz / 2)
    l_mix = l_mix + l_min

    return l_mix

def leaf_boundary_layer_conductance(u: np.ndarray, d: float, Ta: np.ndarray, 
                                    dT:np.ndarray, P: float=101300.0) -> Tuple:
    """
    Computes 2-sided leaf boundary layer conductance assuming mixed forced and free
    convection form two parallel transport mechanisma through the leaf boundary layer.
    
    Args: u (float|array): [m s-1], mean velocity
           d (float|array): [m], characteristic dimension of the leaf
           Ta (float|array): [degC], ambient temperature
           dT (float|array): [degC], leaf-air temperature difference, for free convection
           P (float): [Pa], air pressure

    Returns: boundary-layer conductances (mol m-2 s-1)
        gb_h (float|array): [mol m-2 (leaf) s-1] boundary-layer conductance for heat
        gb_c (float|array): [mol m-2 (leaf) s-1] boundary-layer conductance for CO2
        gb_v (float|array): [mol m-2 (leaf) s-1] boundary-layer conductance for H2O
    
    Reference: 
        Campbell, S.C., and J.M. Norman (1998), An introduction to Environmental Biophysics, 
        Springer, 2nd edition, Ch. 7
    Note: the factor of 1.4 is adopted for outdoor environment, see Campbell and Norman, 1998
        p. 89, 101.
    """

    u = np.maximum(u, EPS)

    factor1 = 1.4*2  # forced conv. both sides, 1.4 is correction for turbulent flow
    factor2 = 1.5  # free conv.; 0.5 comes from cooler surface up or warmer down

    # -- Adjust diffusivity, viscosity, and air density to current pressure/temp.
    t_adj = (101300.0 / P)*((Ta + 273.15) / 293.16)**1.75

    Da_v = MOLECULAR_DIFFUSIVITY_H2O*t_adj
    Da_c = MOLECULAR_DIFFUSIVITY_CO2*t_adj
    Da_T = THERMAL_DIFFUSIVITY_AIR*t_adj
    va = AIR_VISCOSITY*t_adj
    rho_air = 44.6*(P / 101300.0)*(273.15 / (Ta + 273.13))  # [mol/m3]

    # ----- Compute the leaf-level dimensionless groups
    Re = u*d / va  # Reynolds number
    Sc_v = va / Da_v  # Schmid numbers for water
    Sc_c = va / Da_c  # Schmid numbers for CO2
    Pr = va / Da_T  # Prandtl number
    Gr = GRAVITY*(d**3)*abs(dT) / (Ta + 273.15) / (va**2)  # Grashoff number

    # ----- aerodynamic conductance for "forced convection"
    gb_T = (0.664*rho_air*Da_T*Re**0.5*(Pr)**0.33) / d  # [mol/m2/s]
    gb_c=(0.664*rho_air*Da_c*Re**0.5*(Sc_c)**0.33) / d  # [mol/m2/s]
    gb_v=(0.664*rho_air*Da_v*Re**0.5*(Sc_v)**0.33) / d  # [mol/m2/s]

    # ----- Compute the aerodynamic conductance for "free convection"
    gbf_T = (0.54*rho_air*Da_T*(Gr*Pr)**0.25) / d  # [mol/m2/s]
    gbf_c = 0.75*gbf_T  # [mol/m2/s]
    gbf_v = 1.09*gbf_T  # [mol/m2/s]

    # --- aerodynamic conductance: "forced convection"+"free convection"
    gb_h = factor1*gb_T + factor2*gbf_T
    gb_c = factor1*gb_c + factor2*gbf_c
    gb_v = factor1*gb_v + factor2*gbf_v
    # gb_o3=factor1*gb_o3+factor2*gbf_o3

    #r = Gr / (Re**2)  # ratio of free/forced convection

    return gb_h, gb_c, gb_v#, r


def e_sat(T: float) -> Tuple:
    """
    Computes saturation vapor pressure [Pa]and the slope of vapor pressure curve
    [Pa K-1]
    Args:
        T (float|array): [degC], air temperature
    Returns:
        esa (float|array): [Pa], saturation vapor pressure over water film
        s (float|array): [Pa K-1], slope of saturation vapor pressure curve
    Reference:
        Campbell & Norman, 1998. Introduction to Environmental Biophysics. Springer
    """

    esa = 611.0 * np.exp((17.502 * T) / (T + 240.97))  # Pa
    s = 17.502 * 240.97 * esa / ((240.97 + T)**2)

    return esa, s

def latent_heat(T: float) -> float:
    """
    Computes latent heat of vaporization or sublimation.
    Args:
        T (float|array): [degC], temperature
    Returns:
        L (float|array): [J kg-1], latent heat of vaporization or sublimation depending
    """
    # latent heat of vaporizati [J/kg]
    Lv = 1e3 * (3147.5 - 2.37 * (T + DEG_TO_KELVIN))
    # latent heat sublimation [J/kg]
    Ls = Lv + 3.3e5

    L = np.where(T < 0, Ls, Lv)
    return L

# EOF