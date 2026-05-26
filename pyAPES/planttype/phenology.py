# -*- coding: utf-8 -*-
"""
.. module: phenology
    :synopsis: pyAPES-model planttype -component
.. moduleauthor:: Samuli Launiainen & Kersti Leppä

Describes seasonal cycle of photosynthetic capacity and leaf-area development in a PlantType.
For seasonal scaling of Vcmax25, Jmax25, Rd25 different functions for conifers and deciduous
"""

import numpy as np
from typing import Dict, List, Tuple
from pyAPES.utils.constants import DEG_TO_RAD

class Photo_cycle_conifer(object):
    r"""
    Seasonal cycle of photosynthetic machinery in conifers.

    References:
        Kolari et al. 2007 Tellus.
    """
    def __init__(self, p: Dict):
        r""" Initializes photo cycle model for conifers.

        Args:
            p (dict):
                'Xo': initial delayed temperature [degC]
                'fmin': minimum photocapacity [-]
                'Tbase': base temperature [degC]
                'tau': time constant [days]
                'smax': threshold for full acclimation [degC]
            X (float): delayed temperature [degC] at onset of simulations
        Returns:
            self (object)
        """
        self.tau = p['tau']  # time constant (days)
        self.Tbase = p['Tbase']  # base temperature (degC)
        self.Smax = p['smax']  # threshold for full acclimation (degC)
        self.fmin = p['fmin']  # minimum photocapacity (-)

        # state variables
        self.X = p['Xo']  # initial delayed temperature (degC)
        self.f = 1.0  # relative photocapacity

    def run(self, doy: float, T: float, out: bool=False):
        r"""
        Calculate modifier once per pday

        Args:
            T (float): mean daily air temperature [degC]
            out (bool): if true returns phenology modifier [0...1]

        NOTE: Call once per day
        """
        if doy == 0:
            self.X = 0.0
            self.f = self.fmin
        else:

            self.X = self.X + 1.0 / self.tau * (T - self.X)  # degC

            S = np.maximum(self.X - self.Tbase, 0.0)
            self.f = np.maximum(self.fmin,
                            np.minimum(S / (self.Smax - self.Tbase), 1.0))

        if out:
            return self.f
        
class Photo_cycle_decid(object):
    """
    DDsum + daylength three-phase seasonal acclimation of photosynthetic capacity.
    Modifier scalar for Vcmax25, Jmax25, Rd25.

    Reference: Following three-stage Vcmax-pattern inspired by
    Wilson, K.B., Baldocchi, D.D. and Hanson, P.J. (2001): Leaf age affects the seasonal
    pattern of photosynthetic capacity and net ecosystem exchange of carbon in a deciduous 
    forest. Plant, Cell & Environment, 24: 571-583. https://doi.org/10.1046/j.0016-8025.2001.00706.x

    """

    def __init__(self, p: Dict, loc: Dict, DDsum0: float=0.0):
        """
        Args:

        Parameters (p dict)
        -------------------
        Tbase : base temperature for DDsum accumulation [degC]
        ddo   : DDsum at bud burst [degC d]
        ddmat : DDsum at full Vcmax maturity [degC d]
        sdl   : daylength threshold for senescence onset [h]
        sdur  : senescence duration [days]
        f_sso : relative Vcmax at senescence onset [-]
        fmin  : winter minimum [-]

        loc dict must contain 'lat' and 'lon' [decimal degrees].
        """
        self.Tbase = p['Tbase']
        self.ddo   = p['ddo']
        self.ddmat = p['ddmat']
        self.sdl   = p['sdl']
        self.sdur  = p['sdur']
        self.f_sso = p['f_sso']
        self.fmin  = p['fmin']

        # precompute sso: last post-solstice DOY where daylength >= sdl
        doys     = np.arange(1, 366)
        dl_arr   = daylength(lat=loc['lat'], lon=loc['lon'], doy=doys)
        post_sol = doys > 172
        mask     = (dl_arr >= self.sdl) & post_sol
        self.sso = int(doys[mask][-1]) if mask.any() else 258

        self.DDsum   = DDsum0
        self.mat_doy = None   # first DOY ddmat was exceeded
        self.f       = self.fmin

    def run(self, doy: float, T: float, out: bool=False):
        """
        Call once per day.

        Parameters
        ----------
        doy (int): day of year (1–365)
        T (float): mean daily air temperature [degC]
        out (bool): True returns modifier value

        Returns:
            f (float):  relative Vcmax modifier [-]
        """
        fmin  = self.fmin
        f_sso = self.f_sso
        sso   = self.sso

        # reset DDsum at start of each year
        if doy == 1:
            self.DDsum   = 0.0
            self.mat_doy = None

        self.DDsum += np.maximum(0.0, T - self.Tbase)

        # phase 1: exponential spring recovery (fmin → 1.0)
        if self.DDsum <= self.ddo:
            f = fmin
        elif self.DDsum < self.ddmat:
            t = (self.DDsum - self.ddo) / (self.ddmat - self.ddo)
            f = fmin * (1.0 / fmin) ** t
        else:
            f = 1.0

        # phase 2: exponential summer decline (1.0 → f_sso)
        if self.DDsum >= self.ddmat and doy <= sso:
            if self.mat_doy is None:
                self.mat_doy = doy
            span = max(1, sso - self.mat_doy)
            t = min(1.0, max(0.0, (doy - self.mat_doy) / span))
            f = f_sso ** t

        # phase 3: exponential senescence (f_sso → fmin)
        if doy > sso:
            t = min(1.0, (doy - sso) / self.sdur)
            f = max(fmin, f_sso * (fmin / f_sso) ** t)

        self.f = f

        if out:
            return self.f
    
class LAI_cycle(object):
    r"""
    Describes seasonal cycle of leaf-area index (LAI)

    Reference:
        Launiainen et al. 2015 Ecol. Mod
    """
    def __init__(self, p: Dict, loc: Dict, DDsum0: float=0.0):
        r""" Initializes LAI cycle model.

        Args:
            'laip' (dict): parameters for seasonal LAI-dynamics
                'lai_min': minimum LAI, fraction of annual maximum [-]
                'lai_ini': initial LAI fraction, if None lai_ini = Lai_min * LAImax
                'DDsum0': degreedays at initial time [days]
                'Tbase': base temperature [degC]
                'ddo': degreedays at bud burst [days]
                'ddur': duration of recovery period [days]
                'sdl':  daylength for senescence start [h]
                'sdur': duration of decreasing period [days]
            'loc' (dict): 
                'lat': latitude [decimal degrees]
                'lon': longitude [decimal degrees]
            'DDsum0' (float): degree-day sum at onset of simulations [degC]
            
        Returns:
            self (object)
        """
        self.LAImin = p['lai_min']  # minimum LAI, fraction of annual maximum
        self.ddo = p['ddo']
        self.ddmat = p['ddmat']

        # senescence starts at first doy when daylength < sdl
        doy = np.arange(1, 366)
        dl = daylength(lat=loc['lat'], lon=loc['lon'], doy=doy)

        ix = np.max(np.where(dl > p['sdl']))
        self.sso = doy[ix]  # this is onset date for senescence

        self.sdur = p['sdur']
        if p['lai_ini']==None:
            self.f = p['lai_min']  # current relative LAI [...1]
        else:
            self.f = p['lai_ini']

        # degree-day model
        self.Tbase = p['Tbase']  # [degC]
        self.DDsum = DDsum0 # [degC]

    def run(self, doy: int, T: float, out: bool=False):
        r"""
        Computes relative LAI based on seasonal cycle.

        Args:
            T (float): mean daily air temperature [degC]
            out (bool): if true returns LAI relative to annual maximum

        NOTE: Call once per day
        """
        # update DDsum
        if doy == 1:  # reset in the beginning of the year
            self.DDsum = 0.0
        else:
            self.DDsum += np.maximum(0.0, T - self.Tbase)
      
        # spring growth phase
        if self.DDsum <= self.ddo:
            f = self.LAImin
        elif self.DDsum > self.ddo:
            f = np.minimum(1.0, self.LAImin + (1.0 - self.LAImin) *
                 (self.DDsum - self.ddo) / (self.ddmat - self.ddo))

        # autumn senescence phase
        if doy > self.sso:
            f = 1.0 - (1.0 - self.LAImin) * np.minimum(1.0,
                    (doy - self.sso) / self.sdur)

        # update LAI
        self.f = f
        if out:
            return f

def daylength(lat: float, lon: float, doy: float) -> float:
    """
    Computes daylength from a given location and day of year.
    
    Args:
        lat (float|array): [decimal degrees]
        lon (float|array): [decimal degrees]
        doy (float|array): day of year
    Returns:
        dl (float|array): daylength [hours]
    """

    lat = lat * DEG_TO_RAD
    lon = lon * DEG_TO_RAD

    # ---> compute declination angle
    xx = 278.97 + 0.9856 * doy + 1.9165 * np.sin((356.6 + 0.9856 * doy) * DEG_TO_RAD)
    decl = np.arcsin(0.39785 * np.sin(xx * DEG_TO_RAD))

    # --- compute day length, the period when sun is above horizon
    # i.e. neglects civil twilight conditions
    cosZEN = 0.0
    dl = 2.0 * np.arccos(cosZEN - np.sin(lat)*np.sin(decl) /
                         (np.cos(lat)*np.cos(decl))) / DEG_TO_RAD / 15.0  # hours

    return dl

# EOF