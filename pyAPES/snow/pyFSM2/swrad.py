"""
.. module: snow
    :synopsis: APES-model component
.. moduleauthor:: Jari-Pekka Nousu

*Surface and vegetation net shortwave radiation (based on FSM2)*

"""

import numpy as np
from typing import Dict, List, Tuple
from pyAPES.utils.constants import T_MELT

class SWrad(object):
    def __init__(self, 
                 properties: Dict) -> object:
        """
        SWrad based on FSM2.

        Args:
            properties (dict):
                'physics_options' (dict):
                    'ALBEDO' (int): Snow albedo scheme (0,1,2)
                    'SNFRAC' (int): Snow cover fraction scheme (0,1,2)
                'params' (dict):
                    'asmx' (float): Maximum albedo for fresh snow
                    'asmn' (float): Minimum albedo for melting snow
                    'hfsn' (float): Snow cover fraction depth scale (m)
                    'Salb' (float): Snowfall to refresh albedo (kg/m^2)
                    'Talb' (float): Snow albedo decay temperature threshold (C)
                    'tcld' (float): Cold snow albedo decay time scale (s)
                    'tmlt' (float): Melting snow albedo decay time scale (s)
                'intial_conditions' (dict):
                    'fsnow' (float): Initial snow cover fraction [-]
        Returns:
            self (object)
        """

        self.asmx = properties['params']['asmx']
        self.asmn = properties['params']['asmn']
        self.hfsn = properties['params']['hfsn']
        self.Salb = properties['params']['Salb']
        self.Talb = properties['params']['Talb']
        self.tcld = properties['params']['tcld']
        self.tmlt = properties['params']['tmlt']
        self.ALBEDO = properties['physics_options']['ALBEDO']
        self.SNFRAC = properties['physics_options']['SNFRAC']
        self.fsnow = properties['initial_conditions']['fsnow']
        self.albs = 0.1  # initial albedo

        # temporary storage of iteration results
        self.iteration_state = None

    def update(self):
        """ 
        Updates swrad state.
        """
        self.albs = self.iteration_state['albs']
        self.fsnow = self.iteration_state['fsnow']

    def run(self, dt: float, forcing: Dict) -> Tuple:
        """
            Calculates one timestep

            Args:
                dt (float): timestep [s]
                forcing (dict):
                    'Sdif' (float):  # Diffuse shortwave radiation (W/m^2)
                    'Sdir' (float):  # Direct shortwave radiation (W/m^2)
                    'Sf' (float):    # Snowfall rate (kg/m^2/s)
                    'Tsrf' (float):  # Surface temperature (K)
                    'Dsnw' (float):  # Snow depth (m)
                    'alb0' (float):  # Surface albedo [-]
            
            Returns:
                (tuple):
                    fluxes (dict):
                        'SWout' (float): Shortwave radiation reflected by surface (W/m^2)
                        'SWsrf' (float): Shortwave radiation absorbed by surface (W/m^2)
                        'SWsub' (float): Shortwave radiation absorbed by snowpack (W/m^2)
                    states (dict):
                        'snow_albedo' (float): Snow albedo
                        'srf_albedo' (float): Surface albedo
                        'fsnow' (float): Snow cover fraction
        """
        # read forcings
        Sdif = forcing['Sdif']
        Sdir = forcing['Sdir']
        Sf = forcing['Sf']
        Tsrf = forcing['Tsrf']
        Dsnw = forcing['Dsnw']
        alb0 = forcing['alb0']

        if self.ALBEDO == 1:
            # Diagnostic snow albedo
            albs = self.asmn + (self.asmx - self.asmn)*(Tsrf - T_MELT) / self.Talb
        
        if self.ALBEDO == 2:
            # Prognostic snow albedo
            tdec = self.tcld
            if (Tsrf >= T_MELT):
                tdec = self.tmlt
            alim = (self.asmn/tdec + self.asmx*Sf/self.Salb)/(1/tdec + Sf/self.Salb)
            albs = alim + (self.albs - alim)*np.exp(-(1/tdec + Sf/self.Salb)*dt)
        albs = np.maximum(np.minimum(albs,self.asmx),self.asmn)

        # Partial snowcover on ground
        hs = sum(Dsnw[:])
        if self.SNFRAC == 0:
            fsnow = 1.0
        if self.SNFRAC == 1:
            fsnow = np.minimum(hs/self.hfsn, 1.)
        if self.SNFRAC == 2:
            fsnow = hs / (hs + self.hfsn)


        # Surface and vegetation net shortwave radiation
        asrf = (1 - self.fsnow)*alb0 + fsnow * albs
        SWsrf = (1 - asrf)*(Sdif + Sdir)
        SWout = asrf*(Sdif + Sdir)
        SWsub = Sdif + Sdir

        # store iteration state
        self.iteration_state =  {'albs': albs,
                                 'asrf': asrf,
                                 'fsnow': fsnow}
        
            
        fluxes = {'SWout': SWout,
                  'SWsrf': SWsrf,
                  'SWsub': SWsrf,
                  'SWsub': SWsrf,
                 }

        states = {'snow_albedo': albs,
                  'srf_albedo': asrf,
                  'fsnow': fsnow
                 }
        
        return fluxes, states