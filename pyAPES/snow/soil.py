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
            properties (dict): Dictionary containing soil layer parameters.
                It must include a key 'layers' with the following entries:
                    - 'Dzsoil' (np.ndarray): Soil layer thicknesses (m).
                    - 'Nsoil' (int): Number of soil layers.
        """
        self.Dzsoil = properties['layers']['Dzsoil'] # Soil layer thicknesses (m)
        self.Nsoil = properties['layers']['Nsoil'] # Number of soil layers

        self.Tsoil = properties['initial_conditions']['Tsoil']
        self.Vsmc = properties['initial_conditions']['Vsmc']

        # no in nor out
        self.a =  np.zeros(self.Nsoil) # Below-diagonal matrix elements
        self.b = np.zeros(self.Nsoil) # Diagonal matrix elements
        self.c = np.zeros(self.Nsoil) # Above-diagonal matrix elements
        self.dTs = np.zeros(self.Nsoil) # Temperature increments (k)
        self.gs = np.zeros(self.Nsoil) # Thermal conductivity between layers (W/m^2/k)
        self.rhs = np.zeros(self.Nsoil) # Matrix equation rhs

    def run(self, dt: float, forcing: Dict) -> Tuple[Dict, Dict]:
        """
        Update soil temperatures for a given timestep.

        This method updates the soil temperature profile by setting up and solving a
        tridiagonal system representing heat conduction between soil layers. The system
        is constructed using the soil heat capacity, thermal conductivity, and the current
        temperature profile. The solution yields the temperature increments for each layer,
        which are then added to update the soil temperatures.

        Args:
            dt (float): Timestep in seconds.
            forcing (dict): Dictionary containing the following keys:
                - 'Gsoil' (float): Soil heat flux at the soil surface (W/m²).
                - 'csoil' (array_like): Heat capacity of each soil layer (J/m²/K).
                - 'ksoil' (array_like): Thermal conductivity of soil layers (W/m/K).
        Returns:
            tuple: A tuple containing two dictionaries:
                - fluxes (dict): An empty dictionary (no fluxes are explicitly produced).
                - states (dict): A dictionary containing:
                    - 'Tsoil' (array_like): Updated soil temperature profile (K).
        """

        # Extract required variables from the forcing dictionary
        Gsoil = forcing['Gsoil']
        csoil = forcing['csoil']
        ksoil = forcing['ksoil']
        #Tsoil = forcing['Tsoil']

        for k in range(self.Nsoil-1):
            self.gs[k] = 2 / self.Dzsoil[k]/ksoil[k] + self.Dzsoil[k+1]/ksoil[k+1]

        self.a[0] = 0
        self.b[0] = csoil[0] + self.gs[0]*dt
        self.c[0] = - self.gs[0]*dt
        self.rhs[0] = (Gsoil - self.gs[0]*(self.Tsoil[0] - self.Tsoil[1]))*dt

        for k in range(1, self.Nsoil-1):
            self.a[k] = self.c[k-1]
            self.b[k] = csoil[k] + (self.gs[k-1] + self.gs[k])*dt
            self.c[k] = - self.gs[k]*dt
            self.rhs[k] = self.gs[k-1]*(self.Tsoil[k-1] - self.Tsoil[k])*dt + self.gs[k]*(self.Tsoil[k-1] - self.Tsoil[k])*dt 

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

        # Return the updated soil temperature profile; no fluxes are produced.
        fluxes = {}
        states = {'Tsoil': self.Tsoil,
                  'Vsmc': self.Vsmc}

        return fluxes, states




