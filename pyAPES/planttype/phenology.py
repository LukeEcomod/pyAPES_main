# -*- coding: utf-8 -*-
"""
.. module: phenology
    :synopsis: pyAPES-model planttype -component
.. moduleauthor:: Samuli Launiainen & Kersti Leppä

Describes seasonal cycle of photosynthetic capacity and leaf-area development in a PlantType.

"""

import numpy as np
from typing import Dict, List, Tuple
from pyAPES.utils.constants import DEG_TO_RAD

class Photo_cycle(object):
    r"""
    Seasonal cycle of photosynthetic machinery.

    References:
        Kolari et al. 2007 Tellus.
    """
    def __init__(self, p: Dict):
        r""" Initializes photo cycle model.

        Args:
            p (dict):
                'Xo': initial delayed temperature [degC]
                'fmin': minimum photocapacity [-]
                'Tbase': base temperature [degC]
                'tau': time constant [days]
                'smax': threshold for full acclimation [degC]
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

    def run(self, T: float, out: bool=False):
        r"""
        Computes & updates stage of temperature acclimation and relative photosynthetic capacity.

        Args:
            T (float): mean daily air temperature [degC]
            out (bool): if true returns phenology modifier [0...1]

        NOTE: Call once per day
        """
        self.X = self.X + 1.0 / self.tau * (T - self.X)  # degC

        S = np.maximum(self.X - self.Tbase, 0.0)
        self.f = np.maximum(self.fmin,
                            np.minimum(S / (self.Smax - self.Tbase), 1.0))

        if out:
            return self.f

class LAI_cycle(object):
    r"""
    Describes seasonal cycle of leaf-area index (LAI)

    Reference:
        Launiainen et al. 2015 Ecol. Mod
    """
    def __init__(self, p: Dict, loc: Dict):
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
        self.DDsum = p['DDsum0']  # [degC]

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
            self.DDsum = 0.
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