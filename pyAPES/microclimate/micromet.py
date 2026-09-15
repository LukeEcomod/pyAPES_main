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
import logging
from typing import Dict, List, Tuple
from scipy.linalg import solve_banded

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
        self.U_solver = p.get('U_solver')  # method for solving the flow field
        if self.U_solver is None:
            self.U_solver = 'fdm'  # default method for solving the flow field
        if self.U_solver not in ['fdm', 'fvm']:
            raise ValueError("U_solver must be either 'fdm' or 'fvm'")
        # initialize state variables
        self.tau, self.U_n, self.Km_n, self.l_mix, self.d, self.zo, self.g_tau = \
            self._solve_U(z, lad, hc, self.Utop + EPS)

    def _solve_U(self, z, lad, hc, Utop, U_ini=None):
      """
      Dispatches to the momentum solver selected by U_solver, normalizing the two
      solvers' return values to one signature: (tau, U, Km, l_mix, d, zo, g_tau).
      g_tau is None for 'fdm', which has no surface-conductance boundary condition.
      """
      if self.U_solver == 'fvm':
          tau, U, Km, l_mix, d, zo, g_tau = closure_1_model_U_fvm(
              z, self.Cd, lad, hc, Utop, z0=self.zos, dPdx=self.dPdx)
      elif self.U_solver == 'fdm':
          tau, U, Km, l_mix, d, zo = closure_1_model_U(
              z, self.Cd, lad, hc, Utop, self.Ubot, dPdx=self.dPdx, U_ini=U_ini)
          g_tau = None
      return tau, U, Km, l_mix, d, zo, g_tau

    def normalized_flow_stats(self, z: np.ndarray, lad: np.ndarray, hc: float, Utop: float = None, z0=None) -> None:
        """
        Computes normalized mean velocity, shear stress and eddy diffusivity profiles within and above 
        horizontally homogenous plant canopies using 1st order closure schemes.

        Args:
            z (array): canopy model nodes, height from soil surface (= 0.0) [m]
            lad (array): leaf area density [m2 m-3]
            hc (float): canopy heigth [m]
            Utop (float): U/ustar [-], if None, set to self.Utop
            z0 (float): surface roughness length [m]
        Returns:
            (none): updates self.U, self.Km_n, self.tau

        """
        if Utop is None:
            Utop = self.Utop
        if z0 is not None:
            self.zos = z0
        tau, U_n, Km_n, l_mix, d, zo, g_tau = self._solve_U(
            z, lad, hc, Utop + EPS, U_ini=self.U_n)

        if any(U_n < 0.0):
            logger.debug('Negative U_n, set to previous profile.')
        else:
            self.U_n = U_n.copy()
            self.Km_n = Km_n.copy()
            self.l_mix = l_mix.copy()
            self.d = d
            self.zo = zo
            self.g_tau = g_tau
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
        Km = self.Km_n * ustaro + EPS
        if self.U_solver == 'fdm':
            # Old boundary condition for 'fdm' solver
            U[0] = U[1]
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


def closure_1_model_U_fvm(z: np.ndarray, Cd: float, lad: np.ndarray, hc: float, Utop: float, Ubot: float = 0.01,
                          z0: float = 0.01, dPdx: float = 0.0, lbc: str = 'conductance', gamma: float = 0.5,
                          l_min: float = None, max_iter: int = 200):
    """ 
    Mean velocity profile, shear stress and eddy diffusivity within and above 
    horizontally homogenous plant canopies using 1st order closure. Accounts 
    for horizontal pressure gradient force dPdx, assumes neutral diabatic stability.
    Solves displacement height as centroid of drag force. Uses finite volume method.
    Best used for open canopies with relatively low leaf area density but works for
    canopy with higher leaf area density as well.

    Args:
        z (np.ndarray): height [m], constant increments
        Cd (float): drag coefficient (typical range 0.1 - 0.3) [-]
        lad (np.ndarray): plant area density, 2-sided [m2 m-3]
        hc (float): canopy height [m]
        Utop (float): U /u* [-] upper boundary
        Ubot (float, optional): U /u* [-] at ground. Defaults to 0.01.
        z0 (float, optional): surface roughness length. Defaults to 0.01.
        dPdx (float, optional): u* -normalized horizontal pressure gradient. Defaults to 0.0.
        lbc (str, optional): lower boundary condition type. Defaults to 'conductance'.
        gamma (float, optional): weighting factor for iterative solution. Defaults to 0.5.
        l_min (float, optional): minimum mixing length. Defaults to None.
        max_iter (int, optional): maximum number of iterations. Defaults to 200.

    Raises:
        NotImplementedError: only lbc='conductance' is implemented. If lbc='flux' or other types are used, this error will be raised.
        lbc='Dirichlet' user closure_1_model_U.

    Returns:
        (tuple):
            tau (array): u* -normalized momentum flux
            U (array): u* normalized mean wind speed [-]
            Km (array): u* normalized eddy diffusivity for momentum [m]
            l_mix (array): mixing length [m]
            d (float): zero-plane displacement height [m]
            zo (float): roughness lenght for momentum [m]
    """
    z_faces = z.copy()  # z input is faces, e.g., 0,dz,2*dz...
    # z_midpoint has shape (N-1,).
    z_midpoint = 0.5 * (z_faces[:-1] + z_faces[1:])

    lad_faces = 0.5 * np.asarray(lad, dtype=float)
    lad = 0.5 * (lad_faces[:-1] + lad_faces[1:])
    dz = z_midpoint[1] - z_midpoint[0]

    N = len(z_midpoint)

    if l_min is None:
        l_min = VON_KARMAN * z0

    if lbc == 'conductance':
        if z0 >= 0.5*dz:
            raise ValueError(
                "Surface roughness length z0 is too large compared to the grid spacing dz.")
        else:
            g_tau = (VON_KARMAN / np.log(z_midpoint[0] / z0)) ** 2
    else:
        raise NotImplementedError(
            "only lbc='conductance' is implemented, use closure_1_model_U")

    # Initial guess for U profile
    U = np.linspace(max(Ubot, EPS), Utop, N)
    err = 1e6
    iter_no = 0

    # Define RHS of the matrix equation outside iteration loop as this never changes
    # This needs to be minus if we have vertical pressure gradient. In OPT notes we only deal with system where dP/dx=0
    rhs = np.full(N, -1.0*dPdx)
    rhs[-1] = Utop
    while err > 1e-6 and iter_no < max_iter:
        iter_no += 1

        Fd = Cd * lad * U ** 2  # drag force from last iteration
        # displacement height as centroid of drag force
        d = np.sum(z_midpoint * Fd) / (np.sum(Fd) + EPS)
        # mixing length at cell faces (not midpoints since flux is at faces)
        l_mix = mixing_length_fvm(z_faces[1:-1], z0, hc, d, l_min=l_min)

        # calculate dUdz at elements which have element upwards and downwards from them
        dUdz = (U[1:] - U[:-1]) / dz
        # dUdz[0] = (U[0]) / dz0 # calculate dUdz at the bottom flux. Here we assume U(z0) = 0
        # dUdz[-1] = (Utop - U[-1]) / dz # calculate dUdz at the top flux
        Km = l_mix ** 2 * np.abs(dUdz)

        # Check OPT notes from 25.8.2026 for naming of these matrix elements, shape (N-2,)
        A_plus = Km[1:] / dz ** 2
        A_minus = Km[:-1] / dz ** 2  # shape (N-2,)
        B = -A_plus - A_minus - Cd*lad[1:-1]*U[1:-1]  # shape (N-2)
        C = -Km[0]/dz**2 - g_tau/dz*U[0]-Cd*lad[0]*U[0]  # shape (1,)

        # Build tridiagonal matrix for solve_banded
        ab = np.zeros((3, N))
        ab[0, 2:] = A_plus  # upper diagonal
        ab[0, 1] = Km[0]/dz**2  # upper element for the conductance BC
        ab[1, 1:-1] = B  # main diagonal
        ab[1, 0] = C  # conductance BC at bottom
        ab[1, -1] = 1  # Dirichlet BC at top
        ab[2, :-2] = A_minus  # lower diagonal

        # update RHS[-1] with Utop computed at z_midpoint[-1]
        # Utop is given at z_faces[-1] and here we calculate
        # U(z_midpoint[-1]) = Utop - (Utop - U(z_midpoint[-1]))
        # and use u* normalized log profile to estimate Utop and U(z_midpoint[-1])
        rhs[-1] = Utop - np.log((z_faces[-1]-d) /
                                (z_midpoint[-1]-d)) / VON_KARMAN

        U_new = solve_banded((1, 1), ab, rhs)
        err = np.max(np.abs(U_new - U))
        U = gamma * U_new + (1.0 - gamma) * U
        if iter_no == max_iter:
            logger.debug('Maximum number of iterations reached: U_n = %.2f, err = %.2f',
                         np.mean(U), err)

    l_mix = mixing_length_fvm(z_faces[1:-1], z0, hc, d, l_min=l_min)
    dUdz = (U[1:] - U[:-1]) / dz
    Km_int = l_mix ** 2 * np.abs(dUdz)  # Km at interior faces

    # Create face arrays at original z which is what we want to return
    tau_f = np.zeros(N+1)
    Km_f = np.zeros_like(tau_f)
    U_f = np.zeros_like(tau_f)
    lmix_f = np.zeros_like(tau_f)

    tau_f[1:-1] = Km_int * dUdz
    Km_f[1:-1] = Km_int
    U_f[1:-1] = 0.5*(U[1:] + U[:-1])  # mean between two adjacent cell centers
    lmix_f[1:-1] = l_mix  # assign interior mixing lengths

    # ground face: the conductance carries the whole unresolved layer from z0 to z[0]
    # Km_f[0] is the diffusivity that reprocudes tau_0 in this layer
    # since Km_f[0] (U[0]-0)/dz0 = g_tau U[0]**2 <=> Km_f[0] = g_tau U[0] dz0
    dz0 = z_midpoint[0] - z0
    tau_f[0] = g_tau*U[0]**2
    Km_f[0] = g_tau*U[0]*dz0
    U_f[0] = 0.0  # U(z0) = 0, here we approzimate that U(z=0) = U(z0) = 0
    lmix_f[0] = VON_KARMAN*z0

    # top face: Assign last interior values to the top face
    tau_f[-1] = tau_f[-2]
    Km_f[-1] = Km_f[-2]
    lmix_f[-1] = lmix_f[-2]
    U_f[-1] = Utop  # For U use Dirichlet BC at top

    zo = _compute_zo(z_faces, U_f, d)  # roughness sublayer height

    # Normalize tau profile for ustar scaling
    tau_n = tau_f/(tau_f[-2] + EPS)
    return tau_n, U_f, Km_f, lmix_f, d, zo, g_tau


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


def closure_1_model_scalar(dz: float, Ks: np.ndarray, source: np.ndarray, ubc: float, lbc: float,
                           scalar: str, T: float = 20.0, P: float = 101300.0, lbc_dirchlet=False) -> np.ndarray:
    r""" 
    Solves stedy-state scalar profiles in 1-D grid using 1st order closure.
    OPT changed to use solve_banded 09/2026.

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
    # Take harmonic mean of the first two faces since
    # closure_1_model_U_fvm returns Km[0] as conductance between z0 and dz/2
    # instead of arithmetic mean we want conductors in series (= harmonic mean)
    Ks[1] = 2.0*Ks[0]*Ks[1]/(Ks[0] + Ks[1] + EPS)
    if scalar.upper() == 'CO2':  # [umol] -> [mol]
        ubc = 1e-6 * ubc
        source = 1e-6 * source
        lbc = 1e-6 * lbc

    if scalar.upper() == 'T':
        CF = CF * SPECIFIC_HEAT_AIR  # [J m-3 K-1], volumetric heat capacity of air

    # --- Set elements of tridiagonal matrix --
    ab = np.zeros((3, N))  # tridiagonal matrix for solve_banded
    rhs = np.zeros(N)

    # interior nodes
    ab[2, 0:N-2] = Ks[1:-2]
    ab[1, 1:-1] = - (Ks[1:-2] + Ks[2:-1])
    ab[0, 2:] = Ks[2:-1]
    rhs[1:-1] = - source[1:-1] / CF * dz**2

    # uppermost node, Dirchlet boundary
    ab[1, -1] = 1.0
    rhs[-1] = ubc
    # lowermost node
    if not lbc_dirchlet:  # flux-based
        ab[1, 0] = -Ks[1]
        ab[0, 1] = Ks[1]
        rhs[0] = -(lbc*dz / CF) - source[0] * dz**2 / (2*CF)

    else:  # fixed concentration/temperature
        ab[1, 0] = 1.0
        rhs[0] = lbc

    x = solve_banded((1, 1), ab, rhs)

    if scalar.upper() == 'CO2':  # [mol] -> [umol]
        x = 1e6*x

    return x


def mixing_length(z: np.ndarray, h: float, d: float, l_min: float = None) -> np.ndarray:
    """
    computes turbulend mixing length. The l_mix is assumed linear above the canopy, constant within and
    decreases linearly close the ground (below z< alpha*h/VON_KARMAN)

    references:
        juang, j.-y., katul, g.g., siqueira, M.B., Stoy, P.C., McCarthy, H.R., 2008.
        investigating a hierarchy of Eulerian closure models for scalar transfer inside
        forested canopies. boundary-layer Meteorology 128, 1–32.    
    args:
        z (array): [m], computation grid, constant increment
        h (float): [m], canopy height
        d (float): [m], displacement height
        l_min (float): [m], set to finite value at ground

    returns:
        (np.ndarray):
            lmix (array): [m], turbulent mixing length

    """
    dz = z[1] - z[0]

    if not l_min:
        l_min = dz / 2.0

    alpha = (h - d)*VON_KARMAN / (h + EPS)
    i_f = np.sign(z - h) + 1.0
    l_mix = alpha*h*(1 - i_f / 2) + (i_f / 2) * (VON_KARMAN*(z - d))

    sc = (alpha*h) / VON_KARMAN
    ix = np.where(z < sc)
    l_mix[ix] = VON_KARMAN*(z[ix] + dz / 2)
    l_mix = l_mix + l_min
#    l_mix[ix] = von_karman*z[ix]
#    l_mix = np.maximum(l_mix, l_min)

    return l_mix


def mixing_length_fvm(z, z0, hc, d, dz=None, l_min=None):

    dz = z[1] - z[0] if dz is None else dz

    if l_min is None:
        l_min = VON_KARMAN * z0

    l_ground = VON_KARMAN * z

    if hc < 3 * dz:   # canopy not resolved by the grid -> open-ground branch
        return np.maximum(l_ground, l_min)

    alpha = (hc - d) * VON_KARMAN / (hc + EPS)
    I_F = np.sign(z - hc) + 1.0
    l_mix = alpha * hc * (1 - I_F / 2) + (I_F / 2) * (VON_KARMAN * (z - d))

    sc = (alpha * hc) / VON_KARMAN
    ix = np.where(z < sc)
    l_mix[ix] = l_ground[ix]

    return np.maximum(l_mix, l_min)


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