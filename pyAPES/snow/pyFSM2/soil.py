"""
.. module: snow
    :synopsis: APES-model component
.. moduleauthor:: Jari-Pekka Nousu

*Update soil temperatures (based on FSM2)*

"""

import numpy as np
from typing import Dict, List, Tuple
from pyAPES.utils.utilities import tridiag_fsm

EPS = np.finfo(float).eps  # machine epsilon

class SoilModel:
    def __init__(self, 
                 properties: Dict) -> object:        
        """
        Initialize the SoilModel.

        Args:
            'properties' (dict):
                'layers' (dict):
                    'Dzsoil' (np.ndarray): Soil layer thicknesses (m).
                    'Nsoil' (int): Number of soil layers.
                'soilprops' (dict):
                    'fcly' (float): Fraction of clay
                    'fsnd' (float): Fraction of sand
                'initial_conditions' (dict):
                    'Tsoil' (np.ndarray): Soil layer temperatures (k)
                    'Vsmc' (np.ndarray): Soil layer moisture [-]
        """
        self.Dzsoil = properties['layers']['Dzsoil']
        self.Nsoil = properties['layers']['Nsoil']
        self.fcly = properties['soilprops']['fcly']
        self.fsnd = properties['soilprops']['fsnd']
        self.Tsoil = properties['initial_conditions']['Tsoil']
        self.Vsmc = properties['initial_conditions']['Vsmc']

        Vsat = 0.505 - 0.037*self.fcly - 0.142*self.fsnd
        fsat = 0.5
        Tprf = 285.

        for k in range(self.Nsoil):
            self.Vsmc[k]= fsat*Vsat
            self.Tsoil[k] = Tprf

        self.a =  np.zeros(self.Nsoil) # Below-diagonal matrix elements
        self.b = np.zeros(self.Nsoil) # Diagonal matrix elements
        self.c = np.zeros(self.Nsoil) # Above-diagonal matrix elements
        self.dTs = np.zeros(self.Nsoil) # Temperature increments (k)
        self.gs = np.zeros(self.Nsoil) # Thermal conductivity between layers (W/m^2/k)
        self.rhs = np.zeros(self.Nsoil) # Matrix equation rhs

        # temporary storage of iteration results
        self.iteration_state = None  

    def update(self):
        """ 
        Updates soil state.
        """
        self.Tsoil = self.iteration_state['Tsoil']
        self.Vsmc = self.iteration_state['Vsmc']

    def run(self, dt: float, forcing: Dict) -> Tuple[Dict, Dict]:
            """
            Calculates one timestep.

            Args:
                dt (float): timestep (s)
                forcing (dict):
                    'Gsoil' (float): Soil heat flux at the soil surface (W/m²).
                    'csoil' (np.ndarray): Heat capacity of each soil layer (J/m²/K).
                    'ksoil' (np.ndarray): Thermal conductivity of soil layers (W/m/K).
            Returns:
                tuple:
                    'fluxes' (dict):
                    'states' (dict):
                        'Tsoil' (np.ndarray): Soil layer temperatures (K)
            """

            Gsoil = forcing['Gsoil']
            csoil = forcing['csoil']
            ksoil = forcing['ksoil']

            self.gs[0] = 2 / (self.Dzsoil[0]/ksoil[0] + self.Dzsoil[1]/ksoil[1])
            self.a[0] = 0
            self.b[0] = csoil[0] + self.gs[0]*dt
            self.c[0] = - self.gs[0]*dt
            self.rhs[0] = (Gsoil - self.gs[0]*(self.Tsoil[0] - self.Tsoil[1]))*dt

            for k in range(1, self.Nsoil-1):
                self.gs[k] = 2 / (self.Dzsoil[k]/ksoil[k] + self.Dzsoil[k+1]/ksoil[k+1])
                self.a[k] = self.c[k-1]
                self.b[k] = csoil[k] + (self.gs[k-1] + self.gs[k])*dt
                self.c[k] = - self.gs[k]*dt
                self.rhs[k] = self.gs[k-1]*(self.Tsoil[k-1] - self.Tsoil[k])*dt + self.gs[k]*(self.Tsoil[k+1] - self.Tsoil[k])*dt 

            k = self.Nsoil-1
            self.gs[k] = ksoil[k]/self.Dzsoil[k]
            self.a[k] = self.c[k-1]
            self.b[k] = csoil[k] + (self.gs[k-1] + self.gs[k])*dt
            self.c[k] = 0
            self.rhs[k] = self.gs[k-1]*(self.Tsoil[k-1] - self.Tsoil[k])*dt

            #self.dTs = tridiag(a=self.a, b=self.b, C=self.c, D=self.rhs)
            self.dTs = tridiag_fsm(Nvec=self.Nsoil, Nmax=self.Nsoil, a=self.a, b=self.b, c=self.c, r=self.rhs)

            for k in range(self.Nsoil):
                self.Tsoil[k] = self.Tsoil[k] + self.dTs[k]

            # store iteration state
            self.iteration_state =  {'Tsoil': self.Tsoil,
                                     'Vsmc': self.Vsmc}

            fluxes = {}
            states = {'Tsoil': self.Tsoil,
                      'Vsmc': self.Vsmc}
            
            return fluxes, states




