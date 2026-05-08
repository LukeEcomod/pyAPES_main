# -*- coding: utf-8 -*-
"""
.. module: radiation
    :synopsis: pyAPES microclimate component
.. moduleauthor:: Samuli Launiainen, Kersti Leppä

#Shortwave and longwave radiation transfer in multi-layer canopies#

References:
    Launiainen, S., Katul, G.G., Lauren, A. and Kolari, P., 2015. Coupling boreal
    forest CO2, H2O and energy flows by a vertically structured forest canopy –
    Soil model with separate bryophyte layer. Ecological modelling, 312, pp.385-405.

    Zhao W. & Qualls R.J. (2005). A multiple-layer canopy scattering model
    to simulate shortwave radiation distribution within a homogenous plant canopy. 
    Water Resources Res. 41, W08409, 1-16.

    Spitters C.T.J. (1986): Separating the diffuse and direct component of global radiation and
    its implications for modeling canopy photosynthesis part II: Calculation of canopy photosynthesis. 
    Agric. For. Meteorol. 38, 231-242.

    Flerchinger et al. 2009. Simulation of within-canopy radiation exchange, NJAS 57, 5-15

    NOAA solar calculator: https://www.esrl.noaa.gov/gmd/grad/solcalc/
    NOAA solar calculator equations: https://www.esrl.noaa.gov/gmd/grad/solcalc/solareqns.PDF

"""

from builtins import range
import numpy as np
import pandas as pd
import logging
from matplotlib import pyplot as plt
from typing import List, Dict, Tuple
from scipy.linalg import solve_banded

from pyAPES.utils.utilities import tridiag
from pyAPES.utils.constants import DEG_TO_RAD, DEG_TO_KELVIN, STEFAN_BOLTZMANN, SPECIFIC_HEAT_AIR, EPS
logger = logging.getLogger(__name__)


class Radiation(object):
    """
    Short-wave (SW) & long-wave (LW) radiation transfer within horizontally homogeneous multi-layer canopy.
    """

    def __init__(self, p: Dict, Ebal: bool):
        """ 
        Args:
            p (dict):
                clump: [-], clumping index
                leaf_angle: [-], leaf-angle distribution parameter
                Par_alb: [-], shoot Par-albedo [-]
                Nir_alb: [-], shoot NIR-albedo [-]
                leaf_emi: [-], shoot emissivity
            Ebal (bool): True solves LW radiation
        Returns:
            (object):
                self

        """

        # parameters
        self.clump = p['clump']
        self.leaf_angle = p['leaf_angle']  # leaf-angle distribution [-]
        self.alb = {'PAR': p['Par_alb'],  # shoot Par-albedo [-]
                    'NIR': p['Nir_alb']}  # shoot Nir-albedo [-]
        self.leaf_emi = p['leaf_emi']

        # model functions to use: MOVE AS ARGUMENTS
        self.SWmodel = 'ZHAOQUALLS'
        self.LWmodel = 'ZHAOQUALLS'

        logger.info('Shortwave radiation model: %s', self.SWmodel)
        if Ebal:
            logger.info('Longwave radiation model: %s', self.LWmodel)

    def shortwave_profiles(self, forcing: Dict, parameters: Dict) -> Dict:
        """ 
        Computes distribution of within canopy shortwave radiation using specified model.

        Reference:
            Zhao W. & Qualls R.J. (2005). A multiple-layer canopy scattering model
            to simulate shortwave radiation distribution within a homogenous plant
            canopy. Water Resources Res. 41, W08409, 1-16.

        Note: The sunlit fraction at the ground should be computed as: f_sl[0] / clump

        Args:
            forcing (dict):
                zenith_angle: solar zenith angle [rad]
                radtype (dict): keys PAR | NIR 
                        direct [W m-2]
                        diffuse [W m-2]
            parameters (dict):
                'LAIz': layewise one-sided leaf-area index [m2m-2]
                'radiation_type': 'NIR' or 'PAR'

        Returns:
            (tuple):
                Q_sl (array): incident SW sunlit leaves are exposed to [W m-2]
                Q_sh (array): incident SW shaded leaves  are exposed to [W m-2]
                q_sl (array): absorbed SW by sunlit leaves [W m-2(leaf)]
                q_sh (array): absorbed SW by shaded leaves [W m-2(leaf)]
                q_soil (array): absorbed SW by soil surface [W m-2(ground)]
                f_sl (array): sunlit fraction of leaves [-]                
                SW_gr (float): incident SW at ground level [W m-2]

        """

        radtype = parameters['radiation_type'].upper()
        if radtype == 'PAR' or radtype == 'NIR':

            if self.SWmodel == 'ZHAOQUALLS':
                SWb, SWd, SWu, Q_sl, Q_sh, q_sl, q_sh, q_gr, f_sl, alb = canopy_sw_ZhaoQualls(
                    parameters['LAIz'],
                    self.clump, self.leaf_angle,
                    forcing['zenith_angle'],
                    forcing[radtype]['direct'],
                    forcing[radtype]['diffuse'],
                    self.alb[radtype],
                    parameters['ff_albedo'][radtype])

            results = {'sunlit': {'incident': Q_sl, 'absorbed': q_sl, 'fraction': f_sl},
                       'shaded': {'incident': Q_sh, 'absorbed': q_sh},
                       'ground': SWb[0] + SWd[0],
                       'up': SWu,
                       'down': SWb + SWd}

            # ADD OTHER MODELS??

            return results

        else:
            raise ValueError("Radiation type is not 'PAR' or 'NIR'")

    def longwave_profiles(self, forcing: Dict, parameters: Dict) -> Dict:
        """ 
        Computes distribution of within canopy longwave radiation using specified model.

        References:
            Flerchinger et al. 2009. Simulation of within-canopy radiation exchange,
            NJAS 57, 5-15.
            Zhao, W. and Qualls, R.J., 2006. Modeling of long‐wave and net radiation
            energy distribution within a homogeneous plant canopy via multiple scattering
            processes. Water resources research, 42(8).

        Args:
            forcing (dict):
                leaf_temperature (array): leaf temperature [degC]
                lw_in (float): downwelling longwave raditiona above uppermost gridpoint [W m-2(ground)]
                lw_up (float): upwelling longwave radiation below canopy (forest floor) [W m-2(ground)]
            parameters (dict):
                LAIz (array): layewise one-sided leaf-area index [m2 m-2]
                ff_emissivity (float):

        Returns:
            (dict): profiles:
                net_leaf_lw (array): leaf net longwave radiation [W m-2(leaf)]; 
                lw_dn (array): downwelling LW [W m-2]; 
                lw_up (array): upwelling LW [W m-2]; 
                radiative_conductance (array): radiative conductance [mol m-2 s-1]; 

        """

        if self.LWmodel == 'FLERCHINGER':
            lw_leaf, lw_dn, lw_up, gr = canopy_lw(
                LAIz=parameters['LAIz'],
                Clump=self.clump,
                x=self.leaf_angle,
                T=forcing['leaf_temperature'],
                LWdn0=forcing['lw_in'],
                LWup0=forcing['lw_up'],
                leaf_emi=self.leaf_emi
            )

        if self.LWmodel == 'ZHAOQUALLS':
            lw_leaf, lw_dn, lw_up, gr = canopy_lw_ZhaoQualls(
                LAIz=parameters['LAIz'],
                Clump=self.clump,
                x=self.leaf_angle,
                Tleaf=forcing['leaf_temperature'],
                LWdn0=forcing['lw_in'],
                LWup0=forcing['lw_up'],
                leaf_emi=self.leaf_emi,
                soil_emi=parameters['ff_emissivity']
            )

        # ADD OTHER MODELS??

        results = {
            'net_leaf': lw_leaf,
            'down': lw_dn,
            'up': lw_up,
            'radiative_conductance': gr
        }

        return results

# %%
# --- stand-alone functions start here: these can be called with arguments only


def solar_angles(lat: float, lon: float, jday: float, timezone: float = +2.0) -> Tuple:
    """
    Zenith, azimuth and declination angles for given location and time

    Reference: 
        Algorithm based on NOAA solar calculator: https://www.esrl.noaa.gov/gmd/grad/solcalc/
        Equations: https://www.esrl.noaa.gov/gmd/grad/solcalc/solareqns.PDF

    Args:
        lat (float): decimal latitude [deg]
        lon (float): decimal longitude [deg]
        jday (float|array): decimal day of year
        timezone (float): >0 when east from Greenwich UTC

    Returns:
        (tuple):
            zen (float|array): [rad], zenith angle
            azim (float|array): [rad], azimuth angle
            decl (float|array): [rad], declination angle
            sunrise (float_array): [minutes of day], time of sunrise 
            sunset (float|array): [minutes of day], time of sunset
            daylength (float|array): [minutes]

    """
    lat0 = lat * DEG_TO_RAD
    jday = np.array(jday, ndmin=1)

    # fract. year (rad)
    if np.max(jday) > 366:
        y = 2*np.pi / 366.0 * (jday - 1.0)
    else:
        y = 2*np.pi / 365.0 * (jday - 1.0)

    # declination angle (rad)
    decl = (6.918e-3 - 0.399912*np.cos(y) + 7.0257e-2*np.sin(y) - 6.758e-3*np.cos(2.*y)
            + 9.07e-4*np.sin(2.*y) - 2.697e-3*np.cos(3.*y) + 1.48e-3*np.sin(3.*y))

    # equation of time (min)
    et = 229.18*(7.5e-5 + 1.868e-3*np.cos(y) - 3.2077e-2*np.sin(y)
                 - 1.4615e-2*np.cos(2.*y) - 4.0849e-2*np.sin(2.*y))
    # print et / 60.
    # hour angle
    offset = et + 4.*lon - 60.*timezone
    fday = np.modf(jday)[0]  # decimal day
    ha = DEG_TO_RAD * ((1440.0*fday + offset) / 4. - 180.)  # rad

    # zenith angle (rad)
    aa = np.sin(lat0)*np.sin(decl) + np.cos(lat0)*np.cos(decl)*np.cos(ha)
    zen = np.arccos(aa)
    del aa

    # azimuth angle, clockwise from north in rad
    aa = -(np.sin(decl) - np.sin(lat0)*np.cos(zen)) / \
        (np.cos(lat0)*np.sin(zen))
    azim = np.arccos(aa)

    # sunrise, sunset, daylength
    # zenith angle at sunries/sunset after refraction correction
    zen0 = 90.833 * DEG_TO_RAD

    aa = np.cos(zen0) / (np.cos(lat0)*np.cos(decl)) - np.tan(lat0)*np.tan(decl)
    ha0 = np.arccos(aa) / DEG_TO_RAD

    sunrise = 720.0 - 4.*(lon + ha0) - et  # minutes
    sunset = 720.0 - 4.*(lon - ha0) - et  # minutes

    daylength = (sunset - sunrise)  # minutes

    sunrise = sunrise + timezone
    sunrise[sunrise < 0] = sunrise[sunrise < 0] + 1440.0

    sunset = sunset + timezone
    sunset[sunset > 1440] = sunset[sunset > 1440] - 1440.0

    return zen, azim, decl, sunrise, sunset, daylength


def kbeam(zen: float, x: float = 1.0) -> float:
    """
    Attenuation coefficient for direc beam Kb [-] for given solar zenith angle zen [rad]
    and leaf angle distribution x [-]

    Reference: 
        Campbell & Norman. 1998., Introduction to environmental biophysics

    Args:
        zen (float|array): [rad], solar zenith angle
        x (float): [-], leaf-angle distr. parameter
                x = 1 : spherical leaf-angle distr. (default)
                x = 0 : vertical leaf-angle distr.
                x -> inf. : horizontal leaf-angle distr
    Returns:
        Kb (float|aray): [-], beam attenuation coefficient

    """

    zen = np.array(zen)
    zen[zen > 90.0 * DEG_TO_RAD] = 90.0 * DEG_TO_RAD
    x = np.array(x)

    XN1 = (np.sqrt(x*x + np.tan(zen)**2))
    XD1 = (x + 1.774*(x + 1.182)**(-0.733))
    Kb = XN1 / XD1  # direct beam

    Kb = np.minimum(15, Kb)

    return Kb


def kdiffuse(LAI: float, x: float = 1.0) -> float:
    """
    Attenuation coefficient for isotropic diffuse ratioan Kd [-] obtained by integrating
    beam attenuation coefficient over hemisphere

    Reference:
        Campbell & Norman. 1998. Introduction to environmental biophysics, eq. 15.5
    Args:
        LAI (float): [m2 m-2], stand leaf (or plant) area index
        x (float): [-] leaf-angle distr. parameter
                x = 1 : spherical leaf-angle distr. (default)
                x = 0 : vertical leaf-angle distr.
                x = inf. : horizontal leaf-angle distr.
    Returns:
        (float):
        Kd (float): [-], diffuse attenuation coefficient

    """

    LAI = float(LAI)
    x = np.array(x)

    ang = np.linspace(0, np.pi / 2, 90)  # zenith angles at 1deg intervals
    dang = ang[1]-ang[0]

    # beam attenuation coefficient - call kbeam
    Kb = kbeam(ang, x)

    # integrate over hemisphere to get Kd, Campbell & Norman (1998, eq. 15.5)
    YY = np.exp(-Kb*LAI)*np.sin(ang)*np.cos(ang)

    Taud = 2.0*np.trapezoid(YY*dang)
    # extinction coefficient for diffuse radiation
    Kd = -np.log(Taud) / (LAI + EPS)

    return Kd


def canopy_sw_ZhaoQualls(LAIz: np.ndarray, Clump: float, x: float, Zen: float,
                         IbSky: float, IdSky: float, LeafAlbedo: float, SoilAlbedo: float, PlotFigs: bool = False) -> Tuple:
    """
    Computes short-wave (SW) radiation transfer inside horizontally homogeneous multi-layer 
    canopy using two-stream approach. Includes multiple reflections between foliage layers and soil surface.
    Incident radiation is given per ground area to be in line with A-gs measurements at leaf and shoot scale, i.e.
    Vcmax and Jmax reported in literature.

    version Apr 7th, 2026 / Samuli & Kersti

    Reference:
        Zhao W. & Qualls R.J. (2005). A multiple-layer canopy scattering model
        to simulate shortwave radiation distribution within a homogenous plant
        canopy. Water Resources Res. 41, W08409, 1-16.

    Note: 
        To get sunlit fraction below all vegetation: f_sl[0] / Clump.

        Within-clump shading is accounted for by conceptualizing three leaf pools:
          1. Truly sunlit (outer clump surface): fraction f_sl = Clump * exp(-Kb * Clump * LAI_cum) of physical LAI.
             These leaves receive full direct beam + diffuse.
          2. Within-clump self-shaded: fraction (1-Clump) * exp(-Kb * Clump * LAI_cum) of physical LAI.
             These leaves are in a sunlit clump but shielded inside it; receive diffuse only.
          3. Canopy-shaded: fraction 1 - exp(-Kb * Clump * LAI_cum) of physical LAI.
             Receive diffuse only.
        Pools 2 and 3 are merged into a single shaded pool with f_sh = 1 - Clump * exp(-Kb * Clump * LAI_cum).

    Args:
        LAIz (array): [m2 m-2 (ground)], layewise one-sided leaf-area index
        Clump (float): [-], element clumping index
        x (float): [-], param. of leaf angle distribution (1=spherical, 0=vertical, ->inf.=horizontal)
        Zen (float): [rad], solar zenith angle
        IbSky (float): [W m-2], incident direct (beam) radiation above canopy
        IdSky (float): [W m-2], downwelling diffuse radiation above canopy
        LeafAlbedo (float): [-], leaf albedo of desired waveband
        SoilAlbedo (float): [-], soil albedo of desired waveband
        PlotFigs (bool): plot figures

    Returns:
        (tuple):
            SWbo (array): [W m-2 (ground)], direct radiation; 
            SWdo (array): [W m-2 (ground)], downwelling diffuse radiation; 
            SWuo (array): [W m-2 (ground)], upwelling diffuse; 
            Q_sl (array): [W m-2 (ground)] incident SW for truly sunlit un-clumped (physical) leaf area; 
            Q_sh: (array): [W m-2 (ground)] incident SW for shaded un-clumped (physical) leaf area; 
            q_sl: (array): [W m-2 (leaf)], absorbed SW per unit truly sunlit un-clumped (physical) leaf area; 
            q_sh: (array): [W m-2 (leaf)], absorbed SW per unit shaded un-clumped (physical) leaf area; 
            q_soil (float): [W m-2 (ground)], absorbed SW by ground; 
            f_sl: (array): [-] truly sunlit fraction of physical leaf area = Clump * exp(-Kb*Clump*LAI_cum);
                           = outer clump surface fraction; within-clump self-shaded leaves are in shaded pool;
                           multiply by LAIz for truly sunlit physical leaf area per layer;
            alb (array): [-] ecosystem SW albedo; 

    """
    # --- check inputs and create local variables
    IbSky = max(IbSky, 0.0001)
    IdSky = max(IdSky, 0.0001)

    # original and computational grid
    LAI = Clump*sum(LAIz)  # effective LAI, corrected for clumping (m2 m-2)

    Lo = Clump*LAIz  # effective layerwise LAI (or PAI) in original grid

    # cumulative plant area index from canopy top
    Lcumo = np.cumsum(np.flipud(Lo), 0)
    Lcumo = np.flipud(Lcumo)  # node 0 is canopy bottom, N is top

    # --- create computation grid
    N = np.size(Lo)  # nr of original layers
    M = np.minimum(100, N)  # nr of comp. layers
    L = np.ones([M+2])*LAI / M  # effective leaf-area density (m2m-3)
    L[0] = 0.
    L[M + 1] = 0.
    Lcum = np.cumsum(np.flipud(L), 0)  # cumulative plant area from top

    # ---- optical parameters
    aL = np.ones([M+2])*(1 - LeafAlbedo)  # leaf absorptivity
    tL = np.ones([M+2])*0.4  # transmission as fraction of scattered radiation
    rL = np.ones([M+2])*0.6  # reflection as fraction of scattered radiation

    # soil surface, no transmission
    aL[0] = 1. - SoilAlbedo
    tL[0] = 0.
    rL[0] = 1.

    # upper boundary = atm. is transparent for SW
    aL[M+1] = 0.
    tL[M+1] = 1.
    rL[M+1] = 0.

    # black leaf extinction coefficients for direct beam and diffuse radiation
    Kb = kbeam(Zen, x)
    Kd = kdiffuse(LAI, x)

    # fraction of sunlit & shad ground area in a layer (-)
    f_sl = np.flipud(np.exp(-Kb*(Lcum)))

    # beam radiation at each layer
    Ib = f_sl*IbSky

    # propability for beam and diffuse penetration through layer without interception
    taub = np.zeros([M+2])
    taud = np.zeros([M+2])
    taub[0:M+2] = np.exp(-Kb*L)
    taud[0:M+2] = np.exp(-Kd*L)

    # soil surface is non-transparent
    taub[0] = 0.
    taud[0] = 0.

    # backward-scattering functions (eq. 22-23) for beam rb and diffuse rd
    rb = np.zeros([M+2])
    rd = np.zeros([M+2])
    rb = 0.5 + 0.3334*(rL - tL) / (rL + tL)*np.cos(Zen)
    rd = 2.0 / 3.0*rL/(rL + tL) + 1.0 / 3.0*tL / (rL + tL)

    rb[0] = 1.
    rd[0] = 1.
    rb[M+1] = 0.
    rd[M+1] = 0.

    # --- set up tridiagonal matrix A and solve SW without multiple scattering
    # from A*SW = C (Zhao & Qualls, 2006. eq. 39 & 42)

    A_diag = np.zeros(2*M+2)
    A_superdiag = np.zeros(2*M+2)
    A_subdiag = np.zeros(2*M+2)

    # middle rows
    A_diag[1:2*M:2] = - rd[:-2]*(taud[1:-1] + (1 - taud[1:-1]) *
                                 (1 - aL[1:-1])*(1 - rd[1:-1]))*(1 - aL[:-2])*(1 - taud[:-2])
    A_diag[2:2*M+1:2] = - rd[2:]*(taud[1:-1] + (1 - taud[1:-1])
                                  * (1 - aL[1:-1])*(1 - rd[1:-1]))*(1 - aL[2:])*(1 - taud[2:])

    # solve banded needs superdiag with zero in beginning and subdiag with zero at end
    A_superdiag[2:2*M+1:2] = (1 - rd[:-2]*rd[1:-1]*(1 - aL[:-2])
                              * (1 - taud[:-2])*(1 - aL[1:-1])*(1 - taud[1:-1]))
    A_superdiag[3:2*M+2:2] = - \
        (taud[1:-1] + (1 - taud[1:-1])*(1 - aL[1:-1])*(1 - rd[1:-1]))

    A_subdiag[0:2*M-1:2] = - \
        (taud[1:-1] + (1 - taud[1:-1])*(1 - aL[1:-1])*(1 - rd[1:-1]))
    A_subdiag[1:2*M:2] = (1 - rd[1:-1]*rd[2:]*(1 - aL[1:-1])
                          * (1 - taud[1:-1])*(1 - aL[2:])*(1 - taud[2:]))

    # tridiag needs superdiag with zero in end and subdiag with zero at beginning
    # A_superdiag[1:2*M:2] = (1 - rd[:-2]*rd[1:-1]*(1 - aL[:-2])*(1 - taud[:-2])*(1 - aL[1:-1])*(1 - taud[1:-1]))
    # A_superdiag[2:2*M+1:2] = -(taud[1:-1] + (1 - taud[1:-1])*(1 - aL[1:-1])*(1 - rd[1:-1]))

    # A_subdiag[1:2*M:2] = - (taud[1:-1] + (1 - taud[1:-1])*(1 - aL[1:-1])*(1 - rd[1:-1]))
    # A_subdiag[2:2*M+1:2] = (1 - rd[1:-1]*rd[2:]*(1 - aL[1:-1])*(1 - taud[1:-1])*(1 - aL[2:])*(1 - taud[2:]))

    # lower and upeermost nodes
    A_diag[0] = 1.
    A_diag[2*M+1] = 1.

    # --- RHS vector C
    C = np.zeros([2*M+2])

    C[1:2*M:2] = (1 - rd[:-2]*rd[1:-1]*(1 - aL[:-2])*(1 - taud[:-2])*(1 - aL[1:-1])
                  * (1 - taud[1:-1]))*rb[1:-1]*(1 - taub[1:-1])*(1 - aL[1:-1])*Ib[1:-1]
    C[2:2*M+1:2] = (1 - rd[1:-1]*rd[2:]*(1 - aL[1:-1])*(1 - taud[1:-1])*(1 - aL[2:])
                    * (1 - taud[2:]))*(1 - taub[1:-1])*(1 - aL[1:-1])*(1 - rb[1:-1])*Ib[1:-1]

    # lower and uppermost row
    C[0] = SoilAlbedo*Ib[0]
    C[2*M+1] = IdSky

    # ---- solve A*SW = C
    # SW = tridiag(A_subdiag, A_diag, A_superdiag, C)
    SW = solve_banded((1, 1), np.vstack(
        (A_superdiag, A_diag, A_subdiag)), C)  # faster

    # upward and downward hemispherical radiation (Wm-2 ground)
    SWu0 = SW[0:2*M+2:2]
    SWd0 = SW[1:2*M+2:2]

    # ---- Compute multiple scattering, Zhao & Qualls, 2005. eq. 24 & 25.
    # downwelling diffuse after multiple scattering, eq. 24
    SWd = np.zeros([M+1])
    for k in range(M-1, -1, -1):  # downwards from layer k+1 to layer k
        X = SWd0[k+1] / (1 - rd[k]*rd[k+1]*(1-aL[k]) *
                         (1 - taud[k])*(1 - aL[k+1])*(1 - taud[k+1]))
        Y = SWu0[k]*rd[k+1]*(1 - aL[k+1])*(1 - taud[k+1]) / (1 - rd[k] *
                                                             rd[k+1]*(1 - aL[k])*(1 - taud[k])*(1 - aL[k+1])*(1 - taud[k+1]))
        SWd[k+1] = X + Y
    SWd[0] = SWd[1]

    # upwelling diffuse after multiple scattering, eq. 25
    SWu = np.zeros([M+1])
    for k in range(0, M, 1):  # upwards from layer k to layer k+1
        X = SWu0[k] / (1 - rd[k]*rd[k+1]*(1 - aL[k]) *
                       (1 - taud[k])*(1 - aL[k+1])*(1 - taud[k+1]))
        Y = SWd0[k+1]*rd[k]*(1 - aL[k])*(1 - taud[k]) / (1 - rd[k]*rd[k+1]
                                                         * (1 - aL[k])*(1 - taud[k])*(1 - aL[k+1])*(1 - taud[k+1]))
        SWu[k] = X + Y
    SWu[M] = SWu[M-1]

    # match dimensions of all vectors
    Ib = Ib[1:M+2]
    f_sl = f_sl[1:M+2]
    Lcum = np.flipud(Lcum[0:M+1])
    aL = aL[0:M+1]

    # --- NOW return values back to the original grid
    # Between-clump sunlit fraction of ground area (= exp(-Kb * Clump * cumsum(LAIz))).
    # Used only to derive SWbo; NOT returned as f_sl.
    f_slo = np.exp(-Kb*(Lcumo))
    # Beam radiation at each level [W m-2 ground]
    SWbo = f_slo*IbSky

    # Truly sunlit fraction of physical leaf area = Clump * f_slo.
    # Physical interpretation: only the outer clump surface (fraction Clump of physical LAI)
    # is exposed to direct beam; the inner (1-Clump) fraction is always within-clump self-shaded.
    f_slo_true = Clump * f_slo

    # interpolate diffuse fluxes
    X = np.flipud(Lcumo)
    xi = np.flipud(Lcum)
    SWdo = np.flipud(np.interp(X, xi, np.flipud(SWd)))
    SWuo = np.flipud(np.interp(X, xi, np.flipud(SWu)))

    # stand albedo
    alb = SWuo[-1] / (IbSky + IdSky + EPS)
    # soil absorption (Wm-2 (ground))
    q_soil = (1 - SoilAlbedo)*(SWdo[0] + SWbo[0])

    # --- absorbed components per layer (per unit un-clumped / physical leaf area) ---
    # Exact flux-divergence formulation.
    # Arrays SWbo/SWdo/SWuo[k] are at the BOTTOM of layer k (index 0 = canopy bottom).
    # Top-of-layer k = bottom-of-layer k+1, except for the topmost layer where
    # canopy-top upwelling is approximated as SWuo[-1] (exact for all other layers).
    aLo = np.ones(len(Lo))*(1 - LeafAlbedo)

    SWdo_top = np.append(SWdo[1:], IdSky)
    SWuo_top = np.append(SWuo[1:], SWuo[-1])  # approx for top layer only
    SWbo_top = np.append(SWbo[1:], IbSky)

    # Total absorbed per layer [W m-2 ground] = divergence of net downward flux
    total_abs = (SWbo_top + SWdo_top - SWuo_top) - (SWbo + SWdo - SWuo)

    # Beam absorbed per layer [W m-2 ground]
    abs_beam = aLo * (SWbo_top - SWbo)

    # Absorbed per unit physical leaf area [W m-2 physical leaf].
    # Dividing abs_beam by (f_slo_true * LAIz) = (Clump * f_slo * LAIz) gives the full
    # direct beam per truly sunlit leaf: aDiro ≈ aL * Kb * IbSky  (sun-angle-projected beam).
    # The (1-Clump)*f_slo within-clump self-shaded leaves and the (1-f_slo) canopy-shaded
    # leaves are all merged into the shaded pool and receive only diffuse (aDiffo).
    aDiffo = (total_abs - abs_beam) / np.maximum(LAIz,
                                                 EPS)        # diffuse; per unit LAIz
    # beam; per unit truly sunlit LAIz
    aDiro = abs_beam / np.maximum(f_slo_true * LAIz, EPS)

    # --- Sunlit / shaded split, energy-conserving ---
    # f_slo_true = Clump * exp(-Kb * Clump * LAI_cum) is the truly sunlit fraction.
    # Shaded pool = within-clump self-shaded [(1-Clump)*f_slo] + canopy-shaded [1-f_slo].
    #
    # Conservation check:
    #   q_sl * f_slo_true * LAIz + q_sh * (1 - f_slo_true) * LAIz
    #   = (aDiffo + aDiro) * f_slo_true * LAIz + aDiffo * (1 - f_slo_true) * LAIz
    #   = aDiffo * LAIz + aDiro * f_slo_true * LAIz
    #   = (total_abs - abs_beam) + abs_beam = total_abs  [exact]
    # [W m-2 physical leaf], shaded (incl. within-clump self-shaded)
    q_sh = aDiffo
    q_sl = aDiffo + aDiro  # [W m-2 physical leaf], truly sunlit

    # Incident radiation is given per ground area.
    # Sunlit leaves receive direct (IbSky) + diffuse (SWdo + SWuo), shaded leaves receive diffuse only (SWdo + SWuo).
    # [W m-2 ground], incident on truly sunlit leaves
    Q_sl = IbSky + SWdo + SWuo
    Q_sh = SWdo + SWuo  # [W m-2 ground], incident on shaded leaves

    # --- for diagonstics ---
    if PlotFigs:
        depth = -Lcumo / Clump   # y-axis: 0 near top, -LAI/Clump at bottom

        # --- conservation quantities ---
        SWbo_int = np.append(SWbo, IbSky)
        SWdo_int = np.append(SWdo, IdSky)
        SWuo_int = np.append(SWuo, alb * (IbSky + IdSky))
        net_down = SWbo_int + SWdo_int - SWuo_int
        abs_flux = np.diff(net_down)
        abs_leaf = q_sl * f_slo_true * LAIz + q_sh * (1 - f_slo_true) * LAIz

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        fig.suptitle(
            f"canopy_sw_ZhaoQualls_unclumped  LAI={sum(LAIz):.1f} m2 m-2,  Zen={Zen/DEG_TO_RAD:.0f} deg,"
            f"  Clump={Clump},  aL={LeafAlbedo},  as={SoilAlbedo}",
            fontsize=11)

        # SW flux profiles
        ax = axes[0, 0]
        ax.plot(SWbo, depth, 'r-o', ms=4, label='Direct beam SWbo')
        ax.plot(SWdo, depth, 'b-o', ms=4, label='Diffuse down SWdo')
        ax.plot(SWuo, depth, 'g-o', ms=4, label='Diffuse up SWuo')
        ax.set_xlabel('Flux [W m-2 ground]')
        ax.set_ylabel('-Lcum / Clump')
        ax.set_title('SW flux profiles')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Three-pool fractions
        ax = axes[0, 1]
        ax.plot(f_slo_true, depth, 'r-o', ms=4,
                label=r'f$_{sl}$ truly sunlit = $\Omega \cdot e^{-K_b \Omega L}$')
        ax.plot((1 - Clump) * f_slo, depth, 'm--o', ms=4,
                label=r'within-clump self-shaded = $(1-\Omega) \cdot e^{-K_b \Omega L}$')
        ax.plot(1 - f_slo, depth, 'b-o', ms=4,
                label=r'canopy-shaded = $1 - e^{-K_b \Omega L}$')
        ax.plot(1 - f_slo_true, depth, 'k--o', ms=4,
                label=r'total shaded = $1 - \Omega \cdot e^{-K_b \Omega L}$')
        ax.set_xlabel('Fraction [-]')
        ax.set_ylabel('-Lcum / Clump')
        ax.set_title('Three-pool leaf fractions')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1)

        # Incident per physical leaf area
        ax = axes[0, 2]
        ax.plot(Q_sl, depth, 'r-o', ms=4, label='Q_sl truly sunlit')
        ax.plot(Q_sh, depth, 'b-o', ms=4, label='Q_sh shaded')
        ax.set_xlabel('Incident [W m-2 physical leaf]')
        ax.set_ylabel('-Lcum / Clump')
        ax.set_title('Incident radiation per physical leaf area')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Absorbed per physical leaf area
        ax = axes[1, 0]
        ax.plot(q_sl, depth, 'r-o', ms=4, label='q_sl truly sunlit')
        ax.plot(q_sh, depth, 'b-o', ms=4, label='q_sh shaded')
        ax.set_xlabel('Absorbed [W m-2 physical leaf]')
        ax.set_ylabel('-Lcum / Clump')
        ax.set_title('Absorbed radiation per physical leaf area')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Per-layer conservation comparison
        ax = axes[1, 1]
        ax.plot(abs_leaf, depth, 'k-o', ms=4,
                label=r'Leaf-based: $q_{sl} f_{sl} LAI_z + q_{sh}(1-f_{sl})LAI_z$')
        ax.plot(abs_flux, depth, 'r--s', ms=4,
                label='Flux divergence (net_down diff)')
        ax.set_xlabel('Layer absorbed [W m-2 ground]')
        ax.set_ylabel('-Lcum / Clump')
        ax.set_title('Per-layer absorption: leaf vs flux')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        # Residual
        ax = axes[1, 2]
        ax.plot(abs_leaf - abs_flux, depth, 'k-o', ms=4)
        ax.axvline(0, color='r', linestyle='--', linewidth=1)
        ax.set_xlabel('Residual: leaf - flux divergence [W m-2 ground]')
        ax.set_ylabel('-Lcum / Clump')
        ax.set_title(
            'Layer conservation residual\n(non-zero only for top layer: SWuo_top approx.)')
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    return SWbo, SWdo, SWuo, Q_sl, Q_sh, q_sl, q_sh, q_soil, f_slo_true, alb


def canopy_sw_Spitters(LAIz: np.ndarray, Clump: float, x: float, Zen: float,
                       IbSky: float, IdSky: float, LeafAlbedo: float, SoilAlbedo: float,
                       PlotFigs: bool = False) -> Tuple:
    """
    Computes profiles of incident and absorbed SW within horizontally homogeneous plant canopies
    using the analytic model of Spitters (1986) without explicit treatment of upward and downward scattering

    Reference:
        Spitters C.T.J. (1986): Separating the diffuse and direct component of global radiation
        and its implications for modeling canopy photosynthesis part II: Calculation of canopy photosynthesis.
        Agric. For. Meteorol. 38, 231-242.
        Attenuation coefficients and canopy reflectance based on Campbell & Norman (1998): An introduction to environmental
        biophysics, Springer.

    Note:
        At least for conifers NIR LeafAlbedo has to be decreased from leaf-scale values to correctly model canopy albedo of clumped canopies.
        Adjustment from ~0.7 to 0.55 seems to be sufficient. This corresponds to a=a_needle*[4*STAR / (1- a_needle*(1-4*STAR))],
        where a_needle is needle albedo and STAR silhouetteto total area ratio of a conifer shoot. STAR ~0.09-0.21 (mean 0.14) for
        Scots pine. Still then overestimates NIR absorption in upper canopy layers, compared to canopy_sw_ZhaoQualls with explicit multiple scattering.
        Assumes isotropic scattering and does not explicitly compute upward reflected SW.

    Args:
        LAIz (array): [m2 m-2 (ground)], layewise one-sided leaf-area index
        Clump (float): [-], element clumping index
        x (float): [-], param. of leaf angle distribution (1=spherical, 0=vertical, ->inf.=horizontal)
        Zen (float): [rad], solar zenith angle
        IbSky (float): [W m-2], incident direct (beam) radiation above canopy
        IdSky (float): [W m-2], downwelling diffuse radiation above canopy
        LeafAlbedo (float): [-], leaf albedo of desired waveband
        SoilAlbedo (float): [-], soil albedo of desired waveband
        PlotFigs (bool): plot figures

    Returns:
        (tuple):
            SWb (array): [W m-2 (ground)], direct radiation; 
            SWd (array): [W m-2 (ground)], downwelling diffuse radiation; 
            Q_sl (array): [W m-2 (leaf)] incident SW radiation normal to sunlit leaves; 
            Q_sh: (array): [W m-2 (leaf)] incident SW radiation normal to shaded leaves; 
            q_sl: (array): [W m-2 (leaf)], absorbed SW by sunlit leaves; 
            q_sh: (array): [W m-2 (leaf)], absorbed SW by shaded leaves; 
            q_soil (float): [W m-2 (ground)], absorbed SW by ground; 
            f_sl: (array): [-] sunlit fraction of leaves; 
            alb (array): [-] ecosystem SW albedo; 

    """
    # --- check inputs and create local variables
    IbSky = max(IbSky, 0.0001)
    IdSky = max(IdSky, 0.0001)

    L = Clump*LAIz  # effective layerwise LAI (or PAI) in original grid
    # cumulative plant area from the sky, index 0 = ground
    Lcum = np.flipud(np.cumsum(np.flipud(L), 0.0))
    LAI = max(Lcum)

    # attenuation coefficients
    Kb = kbeam(Zen, x)
    Kd = kdiffuse(x, LAI)

    # sunlit fraction as a function of L
    f_sl = np.exp(-Kb*Lcum)

    # canopy hemispherical reflection coefficient Campbell & Norman (1998)
    rcpy1 = (1.0 - (1.0 - LeafAlbedo)**0.5) / (1.0 + (1.0 - LeafAlbedo)**0.5)

    # in case canopy is deep, soil reflections have small impact and canopy reflection coefficients becomes
    rb1 = 2.0*Kb / (Kb + 1.0)*rcpy1  # beam
    rd1 = 2.0*Kd / (Kd + 1.0)*rcpy1  # diffuse

    # but in sparser canopies soil reflectance has to be taken into account and this yields
    AA = ((rb1 - SoilAlbedo) / (rb1*SoilAlbedo - 1.0)) * \
        np.exp(-2.0*(1.0 - LeafAlbedo)**0.5*Kb*LAI)
    rb1 = (rb1 + AA) / (1.0 + rb1*AA)  # beam
    del AA

    AA = ((rd1 - SoilAlbedo) / (rd1*SoilAlbedo - 1.0)) * \
        np.exp(-2.0*(1.0 - LeafAlbedo)**0.5*Kd*LAI)
    rd1 = (rd1 + AA) / (1.0 + rd1*AA)  # diffuse
    del AA

    # canopy albedo
    alb = (rb1*IbSky + rd1*IdSky) / (IbSky + IdSky)

    # Incident SW as a function of Lcum
    qd1 = (1.0 - rd1)*IdSky*np.exp(-(1.0 - LeafAlbedo)
                                   ** 0.5*Kd*Lcum)  # attenuated diffuse
    qb1 = IbSky*np.exp(-Kb*Lcum)  # beam
    qbt1 = (1.0 - rb1)*IbSky*np.exp(-(1.0 - LeafAlbedo)
                                    ** 0.5*Kb*Lcum)  # total beam
    qsc1 = qbt1 - (1.0 - rb1)*qb1  # scattered part of beam
    # print Lcum, f_sl, qd1, qb1, qsc1

    # incident fluxes at each layer per unit ground area
    SWd = qd1 + qsc1  # total diffuse
    SWb = qb1  # total direct beam

    # incident to leaf surfaces (per m2 (leaf))
    Q_sh = Kd*SWd
    Q_sl = Q_sh + Kb*IbSky

    # absorbed components: A = -dq/dL (Wm-2 (leaf))
    # diffuse, total beam, direct beam
    Ad1 = (1.0 - rd1)*IdSky*(1.0 - LeafAlbedo)**0.5 * \
        Kd*np.exp(-(1.0 - LeafAlbedo)**0.5*Kd*Lcum)
    Abt1 = (1.0 - rb1)*IbSky*(1.0 - LeafAlbedo)**0.5 * \
        Kb*np.exp(-(1.0 - LeafAlbedo)**0.5*Kb*Lcum)
    Ab1 = (1.0 - rb1)*(1.0 - LeafAlbedo)*IbSky*Kb * \
        np.exp(-(1.0 - LeafAlbedo)**0.5*Kb*Lcum)

    # absorbed at sunlit & shaded leaves (Wm-2(leaf))
    q_sh = Ad1 + (Abt1 - Ab1)  # total absorbed diffuse
    # sunlit leaves recieve additional direct radiation, Spitters eq. 14
    q_sl = q_sh + (1.0 - LeafAlbedo)*Kb*IbSky

    # absorbed by soil surface (Wm-2(ground))
    q_soil = (1.0 - SoilAlbedo)*(SWb[-1] + SWd[-1])

    # sunlit fraction in clumped foliage; clumping means elements shade each other
    f_sl = Clump*f_sl

    if PlotFigs:
        plt.figure(999)
        plt.subplot(221)
        plt.title("Source: radiation.canopy_sw_Spitters")

        plt.plot(f_sl, -Lcum/Clump, 'r-', (1 - f_sl), -Lcum/Clump, 'b-')
        plt.ylabel("-Lcum eff.")
        plt.xlabel("sunlit & shaded fractions (-)")
        plt.legend(('f_{sl}, total = %.2f' % np.sum(
            f_sl*LAIz), 'f_{sh}, total = %.2f' % np.sum((1 - f_sl)*LAIz)), loc='best')

        # add input parameter values to fig
        plt.text(0.05, 0.75, r'$LAI$ = %1.1f m2 m-2' % (LAI))
        plt.text(0.05, 0.65, r'$ZEN$ = %1.3f rad' % (Zen))
        plt.text(0.05, 0.55, r'$\alpha_l$ = %0.2f' % (LeafAlbedo))
        plt.text(0.05, 0.45, r'$\alpha_s$ = %0.2f' % (SoilAlbedo))

        plt.subplot(222)
        plt.plot(Q_sl, -Lcum/Clump, 'ro-', Q_sh, -Lcum/Clump, 'bo-')
        plt.ylabel("-Lcum eff.")
        plt.xlabel("Incident radiation (Wm-2 (leaf))")
        plt.legend(('sunlit', 'shaded'), loc='best')

        plt.subplot(223)
        plt.plot(SWd, -Lcum/Clump, 'bo', SWb, -Lcum/Clump, 'ro')
        plt.legend(('SWd', 'SWb'), loc='best')
        plt.ylabel("-Lcum eff.")
        plt.xlabel("Incident SW (Wm-2 )")

        plt.subplot(224)
        plt.plot(q_sl, -Lcum/Clump, 'ro-', q_sh, -Lcum/Clump, 'bo-')
        plt.ylabel("-Lcum eff.")
        plt.xlabel("Absorbed radiation (Wm-2 (leaf))")
        plt.legend(('sunlit', 'shaded'), loc='best')

    return SWb, SWd, Q_sl, Q_sh, q_sl, q_sh, q_soil, f_sl, alb


def compute_clouds_rad(doy: float, Zen: float, Rg: float, H2O: float, Tair: float) -> Tuple:
    """
    Estimates atmospheric transmissivity, cloud cover fraction and fraction of 
    diffuse to total SW radiation from surface observations at a given location and time.

    References:
        Cloudiness estimate is based on Song et al. 2009 JGR 114, 2009, Appendix A & C
        Clear-sky emissivity as in Niemelä et al. 2001 Atm. Res 58: 1-18.
        eq. 18 and cloud correction as Maykut & Church 1973.
        Reasonable fit against Hyytiälä data (tested 20.6.13)

    Note: values for Rg < 100 W/m2 linearly interpolated

    Args:
        doy (float|array): julian day
        Zen (float|array): [rad], sun zenith angle
        Rg (float|array): [W m-2], incoming total global radiation (direct + diffuse) above canopy
        H2O (float|array): [Pa], partial pressure of water vapor in the air

    Returns:
        f_cloud (float|array): [-], cloud cover fraction; 
        f_diff (float|array): [-], fraction of diffuse to total radiation; 
        emi_sky (float|array): [-], atmospheric emissivity; 

    """

    # solar constant at top of atm.
    So = 1367.0
    # clear sky Global radiation at surface
    Qclear = np.maximum(0.0,
                        (So * (1.0 + 0.033 * np.cos(2.0 * np.pi * (np.minimum(doy, 365) - 10) / 365)) * np.cos(Zen)))
    tau_atm = Rg / (Qclear + EPS)

    # cloud cover fraction
    f_cloud = np.maximum(0, 1.0 - tau_atm / 0.7)

    # calculate fraction of diffuse to total Global radiation: Song et al. 2009 JGR eq. A17.
    f_diff = np.ones(f_cloud.shape)
    f_diff[tau_atm > 0.7] = 0.2

    ind = np.where((tau_atm >= 0.3) & (tau_atm <= 0.7))
    f_diff[ind] = 1.0 - 2.0 * (tau_atm[ind] - 0.3)

    # clear-sky atmospheric emissivity
    ea = H2O / 100  # near-surface vapor pressure (hPa)
#    emi0 = np.where(ea >= 2.0, 0.72 + 0.009 * (ea - 2.0), 0.72 -0.076 * (ea - 2.0))
    emi0 = 1.24 * (ea/(Tair + 273.15))**(1./7.)  # Song et al 2009

    # all-sky emissivity (cloud-corrections)
#    emi_sky = (1.0 + 0.22 * f_cloud**2.75) * emi0  # Maykut & Church (1973)
    # Song et al 2009 / (Unsworth & Monteith, 1975)
    emi_sky = (1 - 0.84 * f_cloud) * emi0 + 0.84 * f_cloud

#    other emissivity formulas tested
#    emi_sky=(1 + 0.2*f_cloud)*emi0;
#    emi_sky=(1 + 0.3*f_cloud.^2.75)*emi0; % Maykut & Church (1973)
#    emi_sky=(1 + (1./emi0 -1)*0.87.*f_cloud^3.49).*emi0; % Niemelä et al. 2001 eq. 15 assuming Ts = Ta and surface emissivity = 1
#    emi_sky=(1-0.84*f_cloud)*emi0 + 0.84*f_cloud; % atmospheric emissivity (Unsworth & Monteith, 1975)

#    f_cloud[Rg < 100] = np.nan
#    f_diff[Rg < 100] = np.nan
#    emi_sky[Rg < 100] = np.nan

    f_cloud[Qclear < 10] = np.nan
    f_diff[Qclear < 10] = np.nan
    emi_sky[Qclear < 10] = np.nan

    df = pd.DataFrame(
        {'f_cloud': f_cloud, 'f_diff': f_diff, 'emi_sky': emi_sky})
    df = df.interpolate()
    df = df.bfill()
    df = df.ffill()

    return df['f_cloud'].values, df['f_diff'].values, df['emi_sky'].values


def canopy_lw(LAIz: np.ndarray, Clump: float, x: float, T: np.ndarray, LWdn0: float, LWup0: float,
              leaf_emi: float = 1.0, PlotFigs: bool = False) -> Tuple:
    """
    Estimates long-wave (LW) radiation budget and net isothermal LW radiation within horizontally 
    homogeneous canopy. Assumes canopy elements as black bodies (es=1.0) at local air temperature
    T(z), i.e. neglects scattering.

    Reference:
       Adapted from Flerchinger et al. 2009. Simulation of within-canopy radiation exchange, NJAS 57, 5-15.

    Args:
       LAIz (array): [m2 m2(ground)], layer 1-sided leaf-area index
       CLUMP (float): [-], clumping factor
       T (array): [degC], leaf temperature
       LWdn0 (float): [W m-2 (ground)], downward LW above canopy. LWdn0=eatm*b*Tatm^4
       LWup0 (float): [W m-2 (ground)], upward LW at ground. LWup0=esurf*b*Tsurf^4

    Returns:
        (tuple):
            LWleaf (array):  [W m-2 (leaf)], leaf net isothermal LW balance;
            LWdn (array): [W m-2 (ground), downward LW profile in the canopy;
            LWup (array): [W m-2 (ground), upward LW profile in the canopy;
            gr (array): [mol m-2 (leaf) s-1], leaf radiative conductance.

    """

    N = len(LAIz)  # Node 0 = ground
    LWdn = np.zeros(N)
    LWup = np.zeros(N)
    LWleaf = np.zeros(N)
    LWnet = np.zeros(N)

    LayerLAI = Clump*LAIz  # plant-area m2m-2 in a layer addjusted for clumping
    cantop = max(np.where(LayerLAI > 0)[0])  # node at canopy top

    # layerwise attenuation coeffcient
    Kd = kdiffuse(sum(Clump*LAIz), x)
    tau = np.exp(-Kd*LayerLAI)

    # LW down
    LWdn[cantop+1:N] = LWdn0  # downwelling LW entering layer i=cantop
    for k in range(cantop, -1, -1):
        LWdn[k] = tau[k]*LWdn[k+1] + \
            (1 - tau[k])*(leaf_emi*STEFAN_BOLTZMANN*(T[k] + DEG_TO_KELVIN)**4)
    del k

    # LW up
    LWup[0] = LWup0  # upwelling LW entering layer i=0
    for k in range(1, cantop+2):
        LWup[k] = tau[k-1]*LWup[k-1] + \
            (1 - tau[k-1])*(leaf_emi*STEFAN_BOLTZMANN *
                            (T[k-1] + DEG_TO_KELVIN)**4)
    del k
    LWup[cantop+2:N] = LWup[cantop+1]

    # absorbed isothermal net radiation by the leaf (Wm-2(leaf))
    # Kd is mean projection of leaves
    LWleaf[0:cantop+1] = (1 - tau[0:cantop+1])*(
        LWdn[1:cantop+2] + LWup[0:cantop+1] - 2*STEFAN_BOLTZMANN*leaf_emi*(T[0:cantop+1] + DEG_TO_KELVIN)**4)/(
        LAIz[0:cantop+1] + EPS)
    LWnet[0:cantop+1] = (LWdn[1:cantop+2] - LWdn[0:cantop+1] +
                         LWup[0:cantop+1] - LWup[1:cantop+2])/(LAIz[0:cantop+1]+EPS)

    gr = 2 * 4 * leaf_emi * STEFAN_BOLTZMANN * \
        (1 - tau) * (T + DEG_TO_KELVIN) ** 3 / (LAIz + EPS) / SPECIFIC_HEAT_AIR

    if PlotFigs:
        # cumulative plant area index from canopy top
        Lcum = np.cumsum(np.flipud(LAIz))
        Lcum = np.flipud(Lcum)
        plt.figure(99)
        plt.subplot(221)
        plt.title("radiation.canopy_lw", fontsize=8)

        plt.plot(LWdn, -Lcum, 'bo', label='LWdn')
        plt.plot(LWup, -Lcum, 'ro', label='LWup')
        plt.ylabel("-Lcum eff.")
        plt.xlabel("LW (Wm-2 )")
        plt.legend()

        plt.subplot(222)
        plt.plot(LWnet, -Lcum, 'go', label='LWnet')
        plt.plot(LWleaf, -Lcum, 'ro', label='LWleaf')
        plt.ylabel("-Lcum eff.")
        plt.xlabel("LW (Wm-2 )")
        plt.title('LWup0=%.1f, LWdn0=%.1f' % (LWup0, LWdn0))
        plt.legend()

        plt.subplot(223)
        plt.plot(LWdn, list(range(len(Lcum))), 'bo', label='LWdn')
        plt.plot(LWup, list(range(len(Lcum))), 'ro', label='LWup')
        plt.ylabel("N")
        plt.xlabel("LW (Wm-2 )")
        plt.legend()

        plt.subplot(224)
        plt.plot(LWleaf, list(range(len(Lcum))), 'ro', label='LWleaf')
        plt.ylabel("N")
        plt.xlabel("LW (Wm-2 )")
        plt.title('LWup0=%.1f, LWdn0=%.1f' % (LWup0, LWdn0))
        plt.legend()

    return LWleaf, LWdn, LWup, gr


def canopy_lw_ZhaoQualls(LAIz: np.ndarray, Clump: float, x: float, Tleaf: np.ndarray,
                         LWdn0: float, LWup0: float, leaf_emi: float = 0.98, soil_emi: float = 0.98,
                         PlotFigs: bool = False) -> Tuple:
    """
    Long-wave (LW) radiation transfer within horizontally homogeneous plant canopies, accounting for 
    multiple scattering among the canopy layers and soil surface.

    Reference:
        Zhao, W. and Qualls, R.J., 2006. Modeling of long‐wave and net radiation energy distribution 
        within a homogeneous plant canopy via multiple scattering processes. Water resources research, 42(8).

        Flerchinger et al. 2009. NJAS 57, 5-15

    Args:
        LAIz (array): [m2 m2(ground)], layer 1-sided leaf-area index. LAIz[-1] MUST be 0, i.e. domain expand above the canopy!
        CLUMP (float): [-], clumping factor
        x (float): [-] leaf-agle distribution parameter (1=spherical, 0=vertical, ->inf = horizontal)
        Tleaf (array): [degC], leaf temperature
        LWdn0 (float): [W m-2 (ground)], downward LW above canopy. LWdn0=eatm*b*Tatm^4
        LWup0 (float): [W m-2 (ground)], upward LW at ground. LWup0=esurf*b*Tsurf^4
        leaf_emi (float): [-], leaf emissivity 
        soil_emi (float): [-], soil emissivity
        PlotFigs (bool): plots profiles

    Returns:
        (tuple):
            LWleaf (array):  [W m-2 (leaf)], leaf net isothermal LW balance;
            LWdn (array): [W m-2 (ground), downward LW profile in the canopy;
            LWup (array): [W m-2 (ground), upward LW profile in the canopy;
            gr (array): [mol m-2 (leaf) s-1], leaf radiative conductance;

    """
    # original and computational grid
    LAI = Clump*sum(LAIz)  # effective LAI, corrected for clumping (m2 m-2)
    Lo = Clump*LAIz  # effective layerwise LAI (or PAI) in original grid

    # cumulative plant area index from canopy top
    Lcumo = np.cumsum(np.flipud(Lo), 0)
    Lcumo = np.flipud(Lcumo)  # node 0 is canopy bottom, N is top

    # --- create computation grid
    N = np.size(Lo)  # nr of original layers
    M = np.maximum(10, N)  # nr of comp. layers

    L = np.ones([M+2])*LAI / M  # effective leaf-area density (m2m-3)
    L[0] = 0.
    L[M + 1] = 0.
    Lcum = np.cumsum(np.flipud(L), 0)  # cumulative plant area from top

    # interpolate T to comp. grid. T[0] at soil surface
    # for some reason x needs to be increasing..?
    T = np.flipud(np.interp(Lcum, np.flipud(Lcumo), np.flipud(Tleaf)))

    # ---- optical parameters
    # back-scattering fraction, approximation, ZQ06 eq. (6-7)
    if x == 1:  # spherical leaf distrib.
        rd = 2./3.
    elif x == 0:  # vertical leafs
        rd = 0.5
    elif x > 100:  # horizontal leafs
        rd = 1.
    else:
        print("radiation.canopy_lw_ZhaoQualls: check leaf angle distr.")

    rd = np.ones([M+2])*rd
    rd[0] = 1.
    rd[M+1] = 0.

    aL = np.ones([M+2])*leaf_emi  # leaf emissivity
    aL[0] = soil_emi
    # aL[M+1] = 0.

    # extinction coefficients for diffuse radiation
    Kd = kdiffuse(LAI, x)

    # propability of contact with canopy elements in each layer
    taud = np.exp(-Kd*L)  # diffuse
    taud[0] = 0.
    taud[M+1] = 1.

    # --- set up tridiagonal matrix A and solve LW without multiple scattering.
    # ZQ06 eq's. (16 - 25)
    # initialize arrays: A=subdiag, B=diag, C=superdiag, D=rhs
    A = np.zeros(2*M+2)
    B = np.zeros(2*M+2)
    C = np.zeros(2*M+2)
    D = np.zeros(2*M+2)

    # diagonal
    B[0] = 1.0
    B[1:2*M+1:2] = - rd[0:M]*(taud[1:M+1] + (1 - taud[1:M+1])*(1 - aL[1:M+1])*(1 - rd[1:M+1]))*(
                    1 - aL[0:M])*(1 - taud[0:M])
    B[2:2*M+1:2] = - rd[2:M+2]*(taud[1:M+1] + (1 - taud[1:M+1])*(1 - aL[1:M+1])*(1 - rd[1:M+1]))*(
                    1 - aL[2:M+2])*(1 - taud[2:M+2])
    B[2*M+1] = 1.0

    # # for tridiag
    # # subdiagonal
    # A[1:2*M+1:2] = - (taud[1:M+1] + (1 - taud[1:M+1])*(1 - aL[1:M+1])*(1 - rd[1:M+1]))
    # A[2:2*M+1:2] = 1 - rd[1:M+1]*rd[2:M+2]*(1 - aL[1:M+1])*(1 - taud[1:M+1])*(1 - aL[2:M+2])*(1 - taud[2:M+2])
    # # superdiagonal
    # C[1:2*M+1:2] = 1 - rd[0:M]*rd[1:M+1]*(1 - aL[0:M])*(1 - taud[0:M])*(1 - aL[1:M+1])*(1 - taud[1:M+1])
    # C[2:2*M+1:2] = - (taud[1:M+1] + (1 - taud[1:M+1])*(1 - aL[1:M+1])*(1 - rd[1:M+1]))

    # for solve_banded
    # subdiagonal
    A[0:2*M:2] = - (taud[1:M+1] + (1 - taud[1:M+1])*(1 - aL[1:M+1])*(1 - rd[1:M+1]))
    A[1:2*M:2] = 1 - rd[1:M+1]*rd[2:M+2]*(1 - aL[1:M+1])*(1 - taud[1:M+1])*(1 - aL[2:M+2])*(1 - taud[2:M+2])
    # superdiagonal
    C[2:2*M+2:2] = 1 - rd[0:M]*rd[1:M+1]*(1 - aL[0:M])*(1 - taud[0:M])*(1 - aL[1:M+1])*(1 - taud[1:M+1])
    C[3:2*M+2:2] = - (taud[1:M+1] + (1 - taud[1:M+1])*(1 - aL[1:M+1])*(1 - rd[1:M+1]))

    # rhs
    LWsource = aL*STEFAN_BOLTZMANN*(T + DEG_TO_KELVIN)**4
    # lowermost row 0
    D[0] = LWup0
    # rows 1,3,5,...,M-3, M-1
    D[1:2*M+1:2] = (1 - rd[0:M]*rd[1:M+1]*(1 - aL[0:M])*(1 - taud[0:M])*(1 - aL[1:M+1])*(1 - taud[1:M+1]))*(
        1 - taud[1:M+1]) * LWsource[1:M+1]
    # rows 2,4,6,..., M-2, M
    D[2:2*M+1:2] = (1 - rd[1:M+1]*rd[2:M+2]*(1 - aL[1:M+1])*(1 - taud[1:M+1])*(1 - aL[2:M+2])*(1 - taud[2:M+2]))*(
        1 - taud[1:M+1])*LWsource[1:M+1]
    # uppermost row M+1
    D[2*M+1] = LWdn0

    # ---- solve a*LW = D
    if soil_emi < 1.0 and leaf_emi < 1.0:
        # LW = tridiag(A,B,C,D)
        LW = solve_banded((1, 1), np.vstack((C, B, A)), D)
    else:
        matrix = np.zeros([2*M+2, 2*M+2])
        row, col = np.diag_indices(matrix.shape[0])
        matrix[row, col] = B
        matrix[row[1:], col[:-1]] = A[1:]
        matrix[row[:-1], col[1:]] = C[:-1]
        LW = np.linalg.solve(matrix, D)
        del matrix, row, col

    # upward and downward hemispherical radiation (Wm-2 ground)
    LWu0 = LW[0:2*M+2:2]
    LWd0 = LW[1:2*M+2:2]
    del A, B, C, D, LW

    # ---- Compute multiple scattering, Zhao & Qualls, 2006. eq. (8 & 9)
    # downwelling diffuse after multiple scattering
    LWd = np.zeros(M+1)
    X = LWd0 / (1 - rd[0:M+1]*rd[1:M+2]*(1-aL[0:M+1]) *
                (1 - taud[0:M+1])*(1 - aL[1:M+2])*(1 - taud[1:M+2]))
    Y = LWu0*rd[1:M+2]*(1 - aL[1:M+2])*(1 - taud[1:M+2]) / (1 - rd[0:M+1]*rd[1:M+2]
                                                            * (1-aL[0:M+1])*(1 - taud[0:M+1])*(1 - aL[1:M+2])*(1 - taud[1:M+2]))
    LWd = X + Y

    # upwelling diffuse after multiple scattering
    LWu = np.zeros(M+1)
    X = LWu0 / (1 - rd[0:M+1]*rd[1:M+2]*(1-aL[0:M+1]) *
                (1 - taud[0:M+1])*(1 - aL[1:M+2])*(1 - taud[1:M+2]))
    Y = LWd0*rd[0:M+1]*(1 - aL[0:M+1])*(1 - taud[0:M+1]) / (1 - rd[0:M+1]*rd[1:M+2]
                                                            * (1-aL[0:M+1])*(1 - taud[0:M+1])*(1 - aL[1:M+2])*(1 - taud[1:M+2]))
    LWu = X + Y

    # --- NOW return values back to the original grid
    Lcum = Lcum[0:M+1]
    LWd = np.flipud(LWd)
    LWu = np.flipud(LWu)
    # plt.figure(99); plt.plot(LWd, -Lcum, 'r', LWu, -Lcum, 'g')

    X = Lcumo  # node 0 is canopy bottom
    xi = Lcum
    LWdn = np.interp(X, xi, LWd)
    LWup = np.interp(X, xi, LWu)
#    del X, xi

    # ---------------------------------------------------------------------
    # check that interpolation is ok
    # plt.figure(100)
    # plt.plot(LWd,-xi,'r.-',LWdn,-X,'ro', LWu,-xi,'b-',LWup,-X,'bs')
    # plt.title('r = dn, b = up' ); plt.ylabel('LAI eff (m2m-2)')
    # ---------------------------------------------------------------------

    # absorbed net LW per unit un-clumped leaf area (Wm-2(leaf)),
    # Flerchinger et al. 2009. NJAS 57, 5-15
    taud = np.exp(-Kd*Lo)
    LWleaf = np.zeros(len(taud))
    ic = np.where(LAIz > 0)[0]
    if len(ic) == 0:
        cantop = 0
    else:
        cantop = max(ic)
    LWleaf[0:cantop+1] = (1 - taud[0:cantop+1])*leaf_emi*(
        LWdn[1:cantop+2] + LWup[0:cantop+1] - 2*STEFAN_BOLTZMANN*(Tleaf[0:cantop+1] + DEG_TO_KELVIN)**4)/(
        LAIz[0:cantop+1] + EPS)

    gr = 2 * 4 * leaf_emi * STEFAN_BOLTZMANN * \
        (1 - taud) * (Tleaf + DEG_TO_KELVIN) ** 3 / \
        (LAIz + EPS) / SPECIFIC_HEAT_AIR

    if PlotFigs:
        plt.figure(99)
        plt.subplot(221)
        plt.title("radiation.canopy_lw_ZhaoQualls", fontsize=8)

        plt.plot(LWdn, -X/Clump, 'bo', label='LWdn')
        plt.plot(LWup, -X/Clump, 'ro', label='LWup')
        plt.ylabel("-Lcum eff.")
        plt.xlabel("LW (Wm-2 )")
        plt.legend()

        plt.subplot(222)
        plt.plot((-LWd[1:] + LWd[:-1] - LWu[:-1] + LWu[1:]) /
                 (L[1:-1]/Clump + EPS), -xi[1:]/Clump, 'go', label='LWnet')
        plt.plot(LWleaf, -X/Clump, 'ro', label='LWleaf')
        plt.ylabel("-Lcum eff.")
        plt.xlabel("LW (Wm-2 )")
        plt.title('LWup0=%.1f, LWdn0=%.1f' % (LWup0, LWdn0))
        plt.legend()

        plt.subplot(223)

        plt.plot(LWdn, list(range(len(X))), 'bo', label='LWdn')
        plt.plot(LWup, list(range(len(X))), 'ro', label='LWup')
        plt.ylabel("N")
        plt.xlabel("LW (Wm-2 )")
        plt.legend()

        plt.subplot(224)
        plt.plot(LWleaf, list(range(len(X))), 'ro', label='LWleaf')
        plt.ylabel("N")
        plt.xlabel("LW (Wm-2 )")
        plt.title('LWup0=%.1f, LWdn0=%.1f' % (LWup0, LWdn0))
        plt.legend()

    return LWleaf, LWdn, LWup, gr


def test_radiation_functions(LAIz: np.ndarray, Clump: float, Zen: float, x: float = 1.0, method="canopy_sw_ZhaoQualls",
                             leaf_emi: float = 0.98, soil_emi: float = 0.98, leaf_alb: float = 0.12, soil_alb: float = 0.1):
    """
    Test script for SW and LW radiation functions.
    Args:
        LAIz (array): [m2 m2(ground)], layer 1-sided leaf-area index. LAIz[-1] MUST be 0, i.e. domain expand above the canopy!
        CLUMP (float): [-], clumping factor
        Zen (float): [rad], solar zenith angle
        x (float): [-] leaf-agle distribution parameter (1=spherical, 0=vertical, ->inf = horizontal)
        method (str): "canopy_sw_ZhaoQualls", "canopy_sw_Spitters", "canopy_lw_ZhaoQualls", "canopy_lw"
        leaf_emi (float): [-], leaf_emissivity
        soil_emi (float): [-], soil_emissivity
        leaf_alb (float): [-], leafalbedo
        soil_alb (float): [-], soil albedo

    Returns:
        None, plots figures

    """

    # define setup for testing models

    IbSky = 100.0
    IdSky = 100.0

    N = len(LAIz)

    # for LW calculations
    # Tair is 15degC at ground and 17 at upper boundary
    T = np.linspace(15, 17, N)
    Tatm = 17
    Tsurf = 16
    T = T * LAIz / (LAIz + EPS)
    LWdn0 = 0.85*STEFAN_BOLTZMANN*(Tatm + DEG_TO_KELVIN)**4
    LWup0 = 0.98*STEFAN_BOLTZMANN*(Tsurf + DEG_TO_KELVIN)**4

    if method == "canopy_sw_ZhaoQualls":
        print("------TestRun of radiation.canopy_sw_ZhaoQualls with given LAI and CLUMP -----------")
        SWb, SWd, SWu, Q_sl, Q_sh, q_sl, q_sh, q_soil, f_sl, alb = canopy_sw_ZhaoQualls(
            LAIz, Clump, x, Zen, IbSky, IdSky, leaf_alb, soil_alb, PlotFigs="True")
        print(SWu[-1]/(SWb[-1]+SWd[-1]), alb)
#        print SWb,SWd,SWu,Q_sl,Q_sh,q_sl,q_sh,q_soil,f_sl,alb

    if method == "canopy_sw_Spitters":
        print("------TestRun of radiation.canopy_sw_Spitters with given LAI and predefined lad profile-----------")
        SWb, SWd, Q_sl, Q_sh, q_sl, q_sh, q_soil, f_sl, alb = canopy_sw_Spitters(
            LAIz, Clump, x, Zen, IbSky, IdSky, leaf_alb, soil_alb, PlotFigs="True")
        # print SWb, SWd, Q_sl, Q_sh, q_sl, q_sh, q_soil, f_sl, alb

    if method == "canopy_lw":
        print("------TestRun of radiation.canopy_lw------------")
        LWnet, LWdn, LWup, gr = canopy_lw(
            LAIz, Clump, x, T, LWdn0, LWup0, leaf_emi=leaf_emi, PlotFigs=True)
        print(sum(LWnet*LAIz), LWdn[-1]-LWup[-1] - (LWdn[0] - LWup[0]))

    if method == "canopy_lw_ZhaoQualls":
        print("------TestRun of radiation.canopy_lw_ZhaoQualls with given LAI and CLUMP -----------")
        LWnet, LWdn, LWup, gr = canopy_lw_ZhaoQualls(
            LAIz, Clump, x, T, LWdn0, LWup0, leaf_emi=leaf_emi, soil_emi=soil_emi, PlotFigs=True)
        print(LWdn[-1], LWdn[-1]-LWup[-1])

# EOF
