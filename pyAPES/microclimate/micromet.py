# -*- coding: utf-8 -*-
"""
.. module: micromet
    :synopsis: pyAPES-model microclimate component
.. moduleauthor:: Samuli Launiainen, Kersti Leppä, Gaby Katul

#Turbulent and laminar flow, momentum & scalar transport within 1D multi-layer canopies#

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
    """
    Horizontal mean flow and scalar profiles within horizontally homogeneous multi-layer canopy
    """

    def __init__(self, z: np.ndarray, lad: np.ndarray, hc: float, p: Dict):
        """ 
        Args:
            z (array): canopy model nodes, equidistance, height from soil surface (= 0.0) [m]
            lad (array): leaf area density [m2 m-3]
            hc (float): canopy heigth [m]
            p (dict):
                zos: forest floor roughness length [m]
                dPdx: horizontal pressure gradient
                Cd: drag coefficient
                Utop: ensemble U/ustar [-]
                Ubot: U or U/ustar at the lower boundary
                Sc (dict): {'T','H2O','CO2'}, turbulent Schmidt numbers [-]
        Returns:
            (object):
                self (object)

        """

        # parameters
        self.zos = p['zos']  # forest floor roughness length [m]

        self.dPdx = p['dPdx']  # horizontal pressure gradient
        self.Cd = p['Cd']  # drag coefficient
        self.Utop = p['Utop']  # ensemble U/ustar
        self.Ubot = p['Ubot']  # lower boundary
        self.Sc = p['Sc']  # turbulent Schmidt numbers
        self.dz = z[1] - z[0]

        # initialize state variables
        self.tau, self.U_n, self.Km_n, _, _, _ = closure_1_model_U(
            z, self.Cd, lad, hc, self.Utop + EPS, self.Ubot, dPdx=self.dPdx)

    def normalized_flow_stats(self, z: np.ndarray, lad: np.ndarray, hc: float, Utop: float = None) -> None:
        """
        Computes normalized mean velocity, shear stress and eddy diffusivity profiles within and above 
        horizontally homogenous plant canopies using 1st order closure schemes.

        Args:
            z (array): canopy model nodes, height from soil surface (= 0.0) [m]
            lad (array): leaf area density [m2 m-3]
            hc (float): canopy heigth [m]
            Utop (float): U/ustar [-], if None, set to self.Utop
        Returns:
            (none): updates self.U, self.Km_n, self.tau

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
        """ 
        Updates mean wind speed, ustar and eddy-diffusivity profile.
        Args:
            ustaro (float): friction velocity at uppermost grid-point [m s-1]
        Returns:
            (tuple):
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
        """ 
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
            (tuple):
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
            H2O[H2O > H2O_prev] = np.minimum(
                H2O_prev[H2O > H2O_prev] * 1.1, H2O[H2O > H2O_prev])
            H2O[H2O < H2O_prev] = np.maximum(
                H2O_prev[H2O < H2O_prev] * 0.9, H2O[H2O < H2O_prev])

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
            CO2[CO2 > CO2_prev] = np.minimum(
                CO2_prev[CO2 > CO2_prev] * 1.1, CO2[CO2 > CO2_prev])
            CO2[CO2 < CO2_prev] = np.maximum(
                CO2_prev[CO2 < CO2_prev] * 0.9, CO2[CO2 < CO2_prev])

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
                T[T > T_prev] = np.minimum(
                    T_prev[T > T_prev] + 2.0, T[T > T_prev])
                T[T < T_prev] = np.maximum(
                    T_prev[T < T_prev] - 2.0, T[T < T_prev])

            # absolute error
            err_t = max(abs(T - T_prev))
        else:
            err_t = 0.0

        return H2O, CO2, T, err_h2o, err_co2, err_t


def closure_1_model_U(z: np.ndarray, Cd: float, lad: np.ndarray, hc: float,
                      Utop: float, Ubot: float, dPdx: float = 0.0, lbc_flux: bool = None,
                      U_ini: np.array = None) -> Tuple:
    """
    Mean velocity profile, shear stress and eddy diffusivity within and above 
    horizontally homogenous plant canopies using 1st order closure. Accounts 
    for horizontal pressure gradient force dPdx, assumes neutral diabatic stability.
    Solves displacement height as centroid of drag force.

    Args:
       z - height [m]], constant increments
       Cd - drag coefficient (typical range 0.1 - 0.3) [-]
       lad - plant area density, 1-sided [m2 m-3]
       hc - canopy height [m]
       Utop - U /u* [-] upper boundary
       Ubot - U /u* [-] at ground (0.0 for no-slip)
       dPdx - u* -normalized horizontal pressure gradient
       lbc_flux - True sets lower BC to zero flux

    Returns:
        (tuple):
            tau (array): u* -normalized momentum flux
            U (array): u* normalized mean wind speed [-]]
            Km (array): eddy diffusivity for momentum [m2 s-1]
            l_mix (array): mixing length [m]
            d (float): zero-plane displacement height [m]
            zo (float): roughness lenght for momentum [m]

    """

    lad = 0.5*lad  # frontal plant-area density is half of one-sided
    # dz = z[1] - z[2]
    dz = z[2] - z[1]
    N = len(z)
    if U_ini is None:
        U = np.linspace(Ubot, Utop, N)
    else:
        U = U_ini.copy()

    nn1 = max(2, np.floor(N/20))  # window for moving average smoothing

    # --- Start iterative solution
    err = 999.9
    iter_max = 200
    eps1 = 0.5
    dPdx_m = 0.0

    iter_no = 0.0

    while err > 0.001 and iter_no < iter_max:
        iter_no += 1
        Fd = Cd*lad*U**2  # drag force
        d = sum(z*Fd) / (sum(Fd) + EPS)  # displacement height
        l_mix = mixing_length(z, hc, d, l_min=0.01)  # m

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
        rhs = np.ones(N) * dPdx  # _m ???

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
    y_orig = y
    # ---- return values
    tau_orig = tau
    tau = tau / tau[-1]  # normalized shear stress
    zo = (z[-1] - d)*np.exp(-0.4*U[-1])  # roughness length

    y = central_diff(U, dz)
    Kmr = l_mix**2 * abs(y)  # eddy diffusivity
    Km = smooth(Kmr, nn1)

    # --- for testing ----
#    plt.figure(101)
#    plt.subplot(221); plt.plot(Un, z, 'r-'); plt.title('U')
#    plt.subplot(222); plt.plot(y, z, 'b-'); plt.title('dUdz')
#    plt.subplot(223); plt.plot(l_mix, z, 'r-'); plt.title('l mix')
#    plt.subplot(224); plt.plot(Km, z, 'r-', Kmr, z, 'b-'); plt.title('Km')

    return tau, U, Km, l_mix, d, zo, tau_orig, y_orig


def _compute_zo(z: np.ndarray, U: np.ndarray, d: float, n_fit: int = 10) -> float:
    """Estimate roughness length by fitting a log-profile to the upper n_fit nodes."""
    z_fit = z[-n_fit:] - d
    U_fit = U[-n_fit:]
    valid = z_fit > 0.0
    if valid.sum() < 2:
        return float((z[-1] - d) * np.exp(-VON_KARMAN * U[-1]))
    coeffs = np.polyfit(np.log(z_fit[valid]), U_fit[valid], deg=1)
    a, b = coeffs
    if a <= 0.0:
        return float((z[-1] - d) * np.exp(-VON_KARMAN * U[-1]))
    zo = float(np.exp(-b / a))
    return zo if zo > 0.0 else float((z[-1] - d) * np.exp(-VON_KARMAN * U[-1]))


def closure_1_model_U_fdm(z: np.ndarray, Cd: float, lad: np.ndarray, hc: float,
                           Utop: float, Ubot: float, dPdx: float = 0.0,
                           lbc_flux: bool = None, U_ini: np.ndarray = None,
                           l_min: float = None) -> Tuple:
    """
    Mean velocity profile, shear stress and eddy diffusivity — improved FDM.

    Improvements over closure_1_model_U:
        - lbc_flux checked with identity test (is None) instead of bool cast
        - l_min forwarded to mixing_length; pass zos for open surfaces
        - tau normalisation guarded against near-zero tau[-1]
        - zo estimated by multi-point log-profile fit (_compute_zo)

    Args:
        z        : height grid [m], uniform spacing, increasing upward
        Cd       : drag coefficient [-]
        lad      : one-sided plant area density [m2 m-3]
        hc       : canopy height [m]
        Utop     : U/u* at upper boundary [-]
        Ubot     : U/u* at lower boundary [-]  (used when lbc_flux is None)
        dPdx     : u*-normalised horizontal pressure gradient [-]
        lbc_flux : if not None, apply zero-flux Neumann BC at lower boundary
        U_ini    : initial guess; linear profile used if None
        l_min    : minimum mixing length [m]; pass zos for open surfaces

    Returns:
        tau      : u*-normalised momentum flux [-]
        U        : u*-normalised mean wind speed [-]
        Km       : eddy diffusivity [m2 s-1] (smoothed)
        l_mix    : mixing length [m]
        d        : zero-plane displacement height [m]
        zo       : aerodynamic roughness length [m]
        tau_orig : unsmoothed, unnormalised shear stress
    """
    lad = 0.5 * lad
    dz = z[1] - z[0]
    N = len(z)

    U = np.linspace(Ubot, Utop, N) if U_ini is None else U_ini.copy()

    nn1 = max(2, int(np.floor(N / 20)))
    iter_max = 200
    eps1 = 0.5
    iter_no = 0
    err = 999.9

    while err > 0.001 and iter_no < iter_max:
        iter_no += 1
        Fd = Cd * lad * U ** 2
        d = np.sum(z * Fd) / (np.sum(Fd) + EPS)
        l_mix = mixing_length(z, hc, d, l_min=l_min)

        y = central_diff(U, dz)
        Km = l_mix ** 2 * np.abs(y)
        tau = -Km * y

        a1 = -Km
        a2 = central_diff(-Km, dz)
        a3 = Cd * lad * U

        upd = a1 / dz ** 2 + a2 / (2 * dz)
        dia = -a1 * 2 / dz ** 2 + a3
        lod = a1 / dz ** 2 - a2 / (2 * dz)
        rhs = np.full(N, dPdx)

        upd[-1] = 0.0
        dia[-1] = 1.0
        lod[-1] = 0.0
        rhs[-1] = Utop

        if lbc_flux is None:
            upd[0] = 0.0
            dia[0] = 1.0
            lod[0] = 0.0
            rhs[0] = Ubot
        else:
            upd[0] = -1.0
            dia[0] = 1.0
            lod[0] = 0.0
            rhs[0] = 0.0

        Un = tridiag(lod, dia, upd, rhs)
        err = np.max(np.abs(Un - U))
        U = eps1 * Un + (1.0 - eps1) * U

        if iter_no == iter_max:
            logger.debug('Maximum iterations reached: U_n = %.2f, err = %.2f', np.mean(U), err)

    tau_orig = tau
    tau_top = tau[-1]
    if np.abs(tau_top) > EPS:
        tau = tau / tau_top
    else:
        logger.warning('tau[-1] near zero; normalisation skipped.')

    zo = _compute_zo(z, U, d)

    y = central_diff(U, dz)
    Kmr = l_mix ** 2 * np.abs(y)
    Km = smooth(Kmr, nn1)

    return tau, U, Km, l_mix, d, zo, tau_orig


def closure_1_model_scalar(dz: float, Ks: np.ndarray, source: np.ndarray, ubc: float, lbc: float,
                           scalar: str, T: float = 20.0, P: float = 101300.0, lbc_dirchlet=False) -> np.ndarray:
    r""" 
    Solves stedy-state scalar profiles in 1-D grid using 1st order closure

    Note: assumes constant dz and Dirchlet upper boundary condition

    References:
        Juang, J.-Y., Katul, G.G., Siqueira, M.B., Stoy, P.C., McCarthy, H.R., 2008.
        Investigating a hierarchy of Eulerian closure models for scalar transfer inside
        forested canopies. Boundary-Layer Meteorology 128, 1–32.

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
    Returns:
        (np.ndarray): scalar profile, CO2 [ppm] | H2O [mol mol-1] | T [degC]

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
        # [J m-3 K-1], volumetric heat capacity of air
        CF = CF * SPECIFIC_HEAT_AIR

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

    else:  # fixed concentration/temperature
        a[0] = 0.0
        b[0] = 1.
        g[0] = 0.0
        f[0] = lbc

    x = tridiag(a, b, g, f)

    if scalar.upper() == 'CO2':  # [mol] -> [umol]
        x = 1e6*x

    return x

def mixing_length(z: np.ndarray, h: float, d: float,
                  l_min: float = None) -> np.ndarray:
    """
    Turbulent mixing length.
 
    Linear above canopy (von Karman), constant within canopy, linear near
    ground.  Open land (h < dz) handled explicitly to avoid grid dependence.
 
    References:
        Juang et al. (2008) Boundary-Layer Meteorology 128, 1-32.
 
    Args:
        z:     height grid [m], constant increment, increasing upward
        h:     canopy height [m]
        d:     zero-plane displacement height [m]
        l_min: minimum mixing length [m]; defaults to dz/2 unless h < dz
 
    Returns:
        l_mix: turbulent mixing length [m]
    """
    dz = z[1] - z[0]
 
    if l_min is None:           
        l_min = dz / 2.0
 
    # --- open land: bypass canopy logic entirely ---
    if h < dz:
        l_mix = VON_KARMAN * (z - d)
        # use a physically meaningful minimum instead of grid-dependent dz/2
        l_mix = np.maximum(l_mix, VON_KARMAN * l_min)
        return l_mix
 
    # --- canopy case ---
    alpha = (h - d) * VON_KARMAN / (h + EPS)
 
    # step function: 0 inside canopy, 1 above
    above = (np.sign(z - h) + 1.0) / 2.0
 
    l_mix = alpha * h * (1.0 - above) + above * VON_KARMAN * (z - d)
 
    # near-ground ramp: below z < alpha*h/kappa
    sc = alpha * h / VON_KARMAN
    near_ground = z < sc
    l_mix[near_ground] = VON_KARMAN * (z[near_ground] + dz / 2.0)
 
    l_mix += l_min
    return l_mix

def _km_faces(Km_cell: np.ndarray) -> np.ndarray:
    """
    Interpolate cell-centre Km to N+1 cell faces using arithmetic mean.
 
    Face layout:
        face 0        : lower boundary (z = z[0] - dz/2)
        face i        : between cell i-1 and cell i   (i = 1 … N-1)
        face N        : upper boundary (z = z[-1] + dz/2)
 
    Returns:
        Km_f: shape (N+1,)
    """
    Km_f = np.empty(len(Km_cell) + 1)
    Km_f[1:-1] = 0.5 * (Km_cell[:-1] + Km_cell[1:])
    Km_f[0] = Km_cell[0]       # ghost extrapolation at lower boundary
    Km_f[-1] = Km_cell[-1]     # ghost extrapolation at upper boundary
    return Km_f

def closure_1_model_U_fvm(
    z: np.ndarray,
    Cd: float,
    lad: np.ndarray,
    hc: float,
    Utop: float,
    Ubot: float,
    dPdx: float = 0.0,
    lbc_flux: bool = None,
    U_ini: np.ndarray = None,
    l_min: float = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Mean velocity profile, shear stress and eddy diffusivity within and above
    horizontally homogeneous plant canopies — FVM formulation.
 
    Solves:
        d/dz [ Km * dU/dz ] - Cd * a(z) * U^2 = -dPdx
 
    The diffusion term is discretised with Km evaluated at cell faces
    (i ± 1/2), guaranteeing local flux conservation without post-hoc
    smoothing.  The drag term is linearised as (Cd * a * U^(n)) * U^(n+1)
    and the nonlinear system is solved by Picard iteration.
 
    Args:
        z        : height grid [m], uniform spacing, increasing upward
        Cd       : drag coefficient [-]
        lad      : one-sided plant area density [m2 m-3]
        hc       : canopy height [m]
        Utop     : U/u* at upper boundary [-]
        Ubot     : U/u* at lower boundary [-]  (used when lbc_flux is None)
        dPdx     : u*-normalised horizontal pressure gradient [-]
        lbc_flux : if True, apply zero-flux (Neumann) BC at lower boundary;
                   if None, apply Dirichlet BC = Ubot
        U_ini    : initial guess for U; linear profile used if None
        l_min    : minimum mixing length [m]; pass zos for open surfaces

    Returns:
        tau    : u*-normalised momentum flux (shear stress) [-]
        U      : u*-normalised mean wind speed [-]
        Km     : eddy diffusivity for momentum [m2 s-1]
        l_mix  : mixing length [m]
        d      : zero-plane displacement height [m]
        zo     : aerodynamic roughness length [m]
    """
    # frontal area density is half of one-sided
    lad = 0.5 * lad
 
    dz = z[1] - z[0]        # fixed: was z[1] - z[2]
    N = len(z)
 
    # --- initial guess ---
    U = np.linspace(Ubot, Utop, N) if U_ini is None else U_ini.copy()
 
    iter_max = 50
    eps_relax = 0.5
    conv_tol = 0.01
 
    for iter_no in range(1, iter_max + 1):
 
        # --- drag and displacement height ---
        Fd = Cd * lad * U ** 2
        d = np.sum(z * Fd) / (np.sum(Fd) + EPS)
 
        # --- mixing length and cell-centre Km ---
        l_mix = mixing_length(z, hc, d, l_min=l_min)
        dUdz = forward_diff(U, dz)
        Km_cell = l_mix ** 2 * np.abs(dUdz)
 
        # --- face Km (FVM key step) ---
        Km_f = _km_faces(Km_cell)
        Kp = Km_f[1:]    # K at face i+1/2,  shape (N,)
        Km_ = Km_f[:-1]  # K at face i-1/2,  shape (N,)
 
        # --- assemble tridiagonal system ---
        #
        # FVM balance for cell i:
        #   [Km_{i+1/2} (U_{i+1} - U_i) - Km_{i-1/2} (U_i - U_{i-1})] / dz^2
        #   - Cd * a_i * U_i * U_i^(n+1) = -dPdx
        #
        lod = -Km_ / dz ** 2                        # coefficient of U_{i-1}
        dia = (Km_ + Kp) / dz ** 2 + Cd * lad * U  # coefficient of U_i
        upd = -Kp / dz ** 2                         # coefficient of U_{i+1}
        rhs = np.full(N, dPdx)
 
        # --- boundary conditions ---
        # upper: Dirichlet U = Utop
        upd[-1] = 0.0
        dia[-1] = 1.0
        lod[-1] = 0.0
        rhs[-1] = Utop
 
        # lower
        if lbc_flux is None:        # fixed: was `if not lbc_flux`
            # Dirichlet U = Ubot
            upd[0] = 0.0
            dia[0] = 1.0
            lod[0] = 0.0
            rhs[0] = Ubot
        else:
            # Neumann: zero flux  →  U[0] = U[1]  →  dU/dz = 0
            upd[0] = -1.0
            dia[0] = 1.0
            lod[0] = 0.0
            rhs[0] = 0.0
 
        # --- solve ---
        Un = tridiag(lod, dia, upd, rhs)
 
        err = np.max(np.abs(Un - U))
        U = eps_relax * Un + (1.0 - eps_relax) * U
 
        if err < conv_tol:
            break
    else:
        logger.debug(
            "Maximum iterations reached: mean(U) = %.3f, err = %.4f",
            np.mean(U), err,
        )
 
    # --- diagnostics ---
    # shear stress from face fluxes (conservative)
    tau = -Km_f[:-1] * central_diff(U, dz)   # at cell centres, shape (N,)
    tau_orig = tau
    tau_top = tau[-1]
    if np.abs(tau_top) > EPS:
        tau = tau / tau_top
    else:
        logger.warning("tau[-1] near zero; normalisation skipped.")
 
    zo = _compute_zo(z, U, d)

    return tau, U, Km_cell, l_mix, d, zo, tau_orig

# def mixing_length(z: np.ndarray, h: float, d: float, l_min: float = None) -> np.ndarray:
#     """
#     Computes turbulend mixing length. The l_mix is assumed linear above the canopy, constant within and
#     decreases linearly close the ground (below z< alpha*h/VON_KARMAN)

#     References:
#         Juang, J.-Y., Katul, G.G., Siqueira, M.B., Stoy, P.C., McCarthy, H.R., 2008.
#         Investigating a hierarchy of Eulerian closure models for scalar transfer inside
#         forested canopies. Boundary-Layer Meteorology 128, 1–32.    
#     Args:
#         z (array): [m], computation grid, constant increment
#         h (float): [m], canopy height
#         d (float): [m], displacement height
#         l_min (float): [m], set to finite value at ground

#     Returns:
#         (np.ndarray):
#             lmix (array): [m], turbulent mixing length

#     """
#     dz = z[1] - z[0]
    
#     if not l_min:
#         l_min = dz / 2.0

#     alpha = (h - d)*VON_KARMAN / (h + EPS)
#     I_F = np.sign(z - h) + 1.0
#     l_mix = alpha*h*(1 - I_F / 2) + (I_F / 2) * (VON_KARMAN*(z - d))

#     sc = (alpha*h) / VON_KARMAN
#     ix = np.where(z < sc)
#     l_mix[ix] = VON_KARMAN*(z[ix] + dz / 2)
#     l_mix = l_mix + l_min
# #    l_mix[ix] = VON_KARMAN*z[ix]
# #    l_mix = np.maximum(l_mix, l_min)

#     return l_mix


def e_sat(T: float) -> Tuple:
    """
    Saturation vapor pressure and slope of vapor pressure curve

    Reference:
        Campbell & Norman, 1998. Introduction to Environmental Biophysics. Springer

    Args:
        T (float|array): [degC], air temperature

    Returns:
        (tuple):
            esa (float|array): [Pa], saturation vapor pressure over water film
            s (float|array): [Pa K-1], slope of saturation vapor pressure curve

    """

    esa = 611.0 * np.exp((17.502 * T) / (T + 240.97))  # Pa
    s = 17.502 * 240.97 * esa / ((240.97 + T)**2)

    return esa, s


def latent_heat(T: float) -> float:
    """
    Latent heat of vaporization or sublimation.

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
