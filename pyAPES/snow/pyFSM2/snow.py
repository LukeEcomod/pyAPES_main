"""
.. module: snow
    :synopsis: APES-model component
.. moduleauthor:: Jari-Pekka Nousu

*Snow thermodynamics and hydrology (based on FSM2)*

"""

import numpy as np
from typing import Dict, List, Tuple
from pyAPES.utils.utilities import tridiag_fsm
from pyAPES.utils.constants import GRAVITY, SPECIFIC_HEAT_ICE, SPECIFIC_HEAT_H2O, \
    LATENT_HEAT_FUSION, WATER_VISCOCITY, ICE_DENSITY, WATER_DENSITY, T_MELT


EPS = np.finfo(float).eps  # machine epsilon


class SnowModel(object):
    def __init__(self,
                 properties: Dict) -> object:
        """
        Snowpack module based on FSM2.

        Args:
            properties (dict)
                'physics_options' (dict):
                    'DENSTY' (int): 0,1,2
                    'HYDRL': (int): 0,1,2
                'params' (dict):
                    'eta0' (float): # Reference snow viscosity (Pa s)
                    'rcld' (float): # Maximum density for cold snow (kg/m^3)
                    'rfix' (float): # Fixed snow density (kg/m^3)
                    'rgr0' (float): # Fresh snow grain radius (m)
                    'rhof' (float): # Fresh snow density (kg/m^3)
                    'rhow' (float): # Wind-packed snow density (kg/m^3)
                    'rmlt' (float): # Maximum density for melting snow (kg/m^3)
                    'snda' (float): # Thermal metamorphism parameter (1/s)
                    'trho' (float): # Snow compaction timescale (s)
                    'Wirr' (float): # Irreducible liquid water content of snow
                'layers' (dict):
                    'Nsmax' (int): # Maximum number of snow layers
                    'Dzsnow' (list): # Minimum snow layer thicknesses (m)
                'initial_conditions' (dict):
                    Nsnow (int): # Number of snow layers
                    Dsnw (np.ndarray): # Snow layer thicknesses (m)
                    Rgrn (np.ndarray): # Snow layer grain radius (m)
                    Tsnow (np.ndarray): # Snow layer temperatures (K)
                    Sice (np.ndarray): # Liquid content of snow layers (kg/m^2)
                    Sice (np.ndarray): # Ice content of snow layers (kg/m^2)
                    Wflx (np.ndarray): # Water flux into snow layer (kg/m^2/s)

        Returns:
            self (object)
        """

        self.Dzsnow = properties['layers']['Dzsnow']
        self.Nsmax = properties['layers']['Nsmax']
        self.eta0 = properties['params']['eta0']
        self.rcld = properties['params']['rcld']
        self.rfix = properties['params']['rfix']
        self.rgr0 = properties['params']['rgr0']
        self.rhof = properties['params']['rhof']
        self.rhow = properties['params']['rhow']
        self.rmlt = properties['params']['rmlt']
        self.snda = properties['params']['snda']
        self.trho = properties['params']['trho']
        self.Wirr = properties['params']['Wirr']
        self.HYDRL = properties['physics_options']['HYDRL']
        self.DENSTY = properties['physics_options']['DENSTY']
        self.Nsnow = properties['initial_conditions']['Nsnow']
        self.Dsnw = properties['initial_conditions']['Dsnw'] 
        self.Rgrn = properties['initial_conditions']['Rgrn'] 
        self.Sice = properties['initial_conditions']['Sice']
        self.Sliq = properties['initial_conditions']['Sliq'] 
        self.Tsnow = properties['initial_conditions']['Tsnow']
        self.Wflx = properties['initial_conditions']['Wflx']

        # Initializing other variables
        self.a = np.zeros(self.Nsmax)  # Below-diagonal matrix elements
        self.b = np.zeros(self.Nsmax)  # Diagonal matrix elements
        self.c = np.zeros(self.Nsmax)  # Above-diagonal matrix elements
        self.csnow = np.zeros(self.Nsmax) # Areal heat capacity of snow (J/K/m^2)
        self.dTs = np.zeros(self.Nsmax)  # Temperature increments (k)
        self.D = np.zeros(self.Nsmax)  # Layer thickness before adjustment (m)
        self.E = np.zeros(self.Nsmax) # Energy contents before adjustment (J/m^2)
        self.Gs = np.zeros(self.Nsmax) # Thermal conductivity between layers (W/m^2/k)
        self.phi = np.zeros(self.Nsmax)  # Porosity of snow layers
        self.rhs = np.zeros(self.Nsmax)  # Matrix equation rhs
        self.U = np.zeros(self.Nsmax)  # Layer internal energy contents (J/m^2)
        self.dtheta = np.zeros(self.Nsmax)  # Change in liquid water content
        self.ksat = np.zeros(self.Nsmax) # Saturated hydraulic conductivity (m/s)
        self.thetar = np.zeros(self.Nsmax)  # Irreducible water content
        self.thetaw = np.zeros(self.Nsmax)  # Volumetric liquid water content
        self.theta0 = np.zeros(self.Nsmax)  # Liquid water content at start of timestep
        self.Qw = np.zeros(self.Nsmax+1)    # Water flux at snow layer boundaruess (m/s)

        # temporary storage of iteration results
        self.iteration_state = None

    def update(self):
        """ 
        Updates snow state.
        """
        self.Sliq = self.iteration_state['Sliq']
        self.Sice = self.iteration_state['Sice']
        self.Tsnow = self.iteration_state['Tsnow']
        self.Nsnow = self.iteration_state['Nsnow']
        self.Dsnw = self.iteration_state['Dsnw']
        self.Rgrn = self.iteration_state['Rgrn']

    def run(self, dt: float, forcing: Dict) -> Tuple:
        """
        Calculates one timestep.

        Args:
            dt (float): timestep [s]
            forcing (dict):
                'drip' (float):       # Melt water drip from vegetation (kg/m^2)
                'Esrf' (float):       # Moisture flux from the surface (kg/m^2/s)
                'Gsrf' (float):       # Heat flux into snow/ground surface (W/m^2)
                'ksnow' (float):      # Thermal conductivity of snow layers (W/m/K)
                'Melt' (float):       # Surface melt rate (kg/m^2/s)
                'Rf' (float):         # Rainfall rate (kg/m2/s)
                'Sf' (float):         # Snowfall rate (kg/m2/s)
                'Ta' (float):         # Air temperature (K)
                'trans' (float):      # Wind-blown snow transport rate (kg/m^2/s)
                'Tsrf' (float):       # Snow/ground surface temperature (K)
                'unload' (float):     # Snow mass unloaded from vegetation (kg/m^2)
                'Tsoil' (float):      # Soil layer temperature (K)
                'ksoil' (float):      # Thermal conductivity of soil layer (W/m/K)
                'Dzsoil' (float):     # Soil layer thickness (m)

        Returns
            (tuple):
            fluxes (dict):
                'Gsoil' (float):        # Heat flux into soil (W/m^2)
                'Roff' (float):         # Runoff from snow (kg/m^2/s)
                'wbal' (float):         # Water balance error (kg/m^2/s)
            states (dict):
                'swe' (float):          # Total snow mass on ground (kg/m^2)
                'hs' (float):           # Snow depth (m)
                'Sice' (np.ndarray):    # Snow ice content (kg/m^2)
                'Sliq' (np.ndarray):    # Snow liquid content (kg/m^2)
                'Nsnow' (int):          # Number of snow layers
                'Dsnw' (np.ndarray):    # Snow layer thicknesses (m)
                'Tsnow' (np.ndarray):   # Snow layer temperatures (K)
                'rhos' (np.ndarray):    # Snow layer densities (kg/m^3)
        """

        # read forcings
        drip = forcing['drip']
        Esrf = forcing['Esrf']
        Gsrf = forcing['Gsrf']
        ksnow = forcing['ksnow']
        Melt = forcing['Melt']
        Rf = forcing['Rf']
        Sf = forcing['Sf']
        Ta = forcing['Ta']
        trans = forcing['trans']
        Tsrf = forcing['Tsrf']
        unload = forcing['unload']
        Tsoil = forcing['Tsoil']
        ksoil = forcing['ksoil']
        Dzsoil = forcing['Dzsoil']

        # old states
        Nsnow = int(self.Nsnow)
        Sliq = self.Sliq.copy()
        Sice = self.Sice.copy()
        Tsnow = self.Tsnow.copy()
        Dsnw = self.Dsnw.copy()
        Rgrn = self.Rgrn.copy()

        # No snow
        Gsoil = Gsrf.copy()
        Roff = Rf + drip / dt
        Wflx = np.zeros(self.Nsmax)

        # Existing snowpack
        if (Nsnow > 0):
            # Heat conduction
            for k in range(Nsnow):
                # Areal heat capacity
                self.csnow[k] = Sice[k]*SPECIFIC_HEAT_ICE + \
                    Sliq[k]*SPECIFIC_HEAT_H2O

            if (Nsnow == 1):
                self.Gs[0] = 2 / (Dsnw[0]/ksnow[0] +
                                  Dzsoil/ksoil)
                self.dTs[0] = (Gsrf + self.Gs[0]*(Tsoil - Tsnow[0])
                               ) * dt / (self.csnow[0] + self.Gs[0] * dt)

            else:
                self.Gs[0] = 2 / (Dsnw[0]/ksnow[0] +
                                  Dsnw[1]/ksnow[1])
                self.a[0] = 0
                self.b[0] = self.csnow[0] + self.Gs[0]*dt
                self.c[0] = -self.Gs[0]*dt
                self.rhs[0] = (Gsrf - self.Gs[0] *
                               (Tsnow[0] - Tsnow[1]))*dt

                for k in range(1, Nsnow-1):
                    self.a[k] = self.c[k-1]
                    self.b[k] = self.csnow[k] + (self.Gs[k-1] + self.Gs[k])*dt
                    self.c[k] = - self.Gs[k]*dt
                    self.rhs[k] = self.Gs[k-1]*(Tsnow[k-1] - Tsnow[k])*dt + self.Gs[k]*(
                        Tsnow[k+1] - Tsnow[k])*dt

                k = Nsnow - 1
                self.Gs[k] = 2 / (Dsnw[k]/ksnow[k] +
                                  Dzsoil/ksoil)
                self.a[k] = self.c[k-1]
                self.b[k] = self.csnow[k] + (self.Gs[k-1] + self.Gs[k])*dt
                self.c[k] = 0
                self.rhs[k] = self.Gs[k-1] * (Tsnow[k-1] - Tsnow[k]) * \
                    dt + self.Gs[k]*(Tsoil - Tsnow[k]) * dt
                self.dTs = tridiag_fsm(
                    Nvec=Nsnow, Nmax=self.Nsmax, a=self.a, b=self.b, c=self.c, r=self.rhs)

            for k in range(Nsnow):
                Tsnow[k] = Tsnow[k] + self.dTs[k]

            k = Nsnow - 1
            Gsoil = self.Gs[k] * (Tsnow[k] - Tsoil)

            # Convert melting ice to liquid water
            dSice = Melt * dt
            for k in range(Nsnow):
                coldcont = self.csnow[k]*(T_MELT - Tsnow[k])
                if (coldcont < 0.):
                    dSice = dSice - coldcont / LATENT_HEAT_FUSION
                    Tsnow[k] = T_MELT
                if (dSice > EPS):
                    if (dSice > Sice[k]):  # Layer melts completely
                        dSice = dSice - Sice[k]
                        Dsnw[k] = 0.
                        Sliq[k] = Sliq[k] + Sice[k]
                        Sice[k] = 0.
                    else:                       # Layer melts partially
                        Dsnw[k] = (1 - dSice/Sice[k])*Dsnw[k]
                        Sice[k] = Sice[k] - dSice
                        Sliq[k] = Sliq[k] + dSice
                        dSice = 0.
                        break

            # Remove snow by sublimation
            dSice = Esrf * dt
            if (dSice > EPS):
                for k in range(Nsnow):
                    if (dSice > Sice[k]):  # Layer sublimates completely
                        dSice = dSice - Sice[k]
                        Dsnw[k] = 0.
                        Sice[k] = 0.
                    else:                  # Layer sublimates partially
                        Dsnw[k] = (1 - dSice/Sice[k])*Dsnw[k]
                        Sice[k] = Sice[k] - dSice
                        dSice = 0.
                        break

            # Remove wind-transported snow
            dSice = trans * dt
            if (dSice > EPS):
                for k in range(Nsnow):
                    if (dSice > Sice[k]):  # Layer completely removed
                        dSice = dSice - Sice[k]
                        Dsnw[k] = 0.
                        Sice[k] = 0.
                    else:                       # Layer partially removed
                        Dsnw[k] = (1 - dSice/Sice[k])*Dsnw[k]
                        Sice[k] = Sice[k] - dSice
                        dSice = 0.
                        break

            if self.DENSTY == 0:
                # Fixed snow density
                for k in range(Nsnow):
                    if (Dsnw[k] > EPS):
                        Dsnw[k] = (
                            Sice[k] + Sliq[k]) / self.rfix
            if self.DENSTY == 1:
                # Snow compaction with age
                for k in range(Nsnow):
                    if Dsnw[k] > EPS:
                        rhos = (Sice[k] + Sliq[k]) / Dsnw[k]
                        if Tsnow[k] >= T_MELT:
                            if rhos < self.rmlt:
                                rhos = self.rmlt + \
                                    (rhos - self.rmlt) * \
                                    np.exp(-dt / self.trho)
                        else:
                            if rhos < self.rcld:
                                rhos = self.rcld + \
                                    (rhos - self.rcld) * \
                                    np.exp(-dt / self.trho)
                        Dsnw[k] = (Sice[k] + Sliq[k]) / rhos

            if self.DENSTY == 2:
                # Snow compaction by overburden
                mass = 0
                for k in range(Nsnow):
                    mass = mass + 0.5*(Sice[k] + Sliq[k])
                    if (Dsnw[k] > EPS):
                        rhos = (Sice[k] + Sliq[k]) / Dsnw[k]
                        rhos = rhos + (rhos*GRAVITY*mass*dt/(self.eta0*np.exp(-(Tsnow[k] - T_MELT)/12.4 + rhos/55.6)) +
                                       dt * rhos*self.snda*np.exp((Tsnow[k] - T_MELT)/23.8 - max(rhos - 150, 0.)/21.7))
                        Dsnw[k] = (Sice[k] + Sliq[k]) / rhos
                    mass = mass + 0.5*(Sice[k] + Sliq[k])

            # Snow grain growth
            for k in range(Nsnow):
                ggr = 2e-13
                if (Tsnow[k] < T_MELT):
                    if (Rgrn[k] < 1.50e-4):
                        ggr = 2e-14
                    else:
                        ggr = 7.3e-8*np.exp(-4600/Tsnow[k])
                Rgrn[k] = Rgrn[k] + dt * ggr / Rgrn[k]
        # End if for existing snowpack

        # Add snowfall and frost to layer 1 with fresh snow density and grain size
        Esnow = 0.
        if (Esrf < 0.) and (Tsrf < T_MELT):
            Esnow = Esrf
        dSice = (Sf - Esnow)*dt
        Dsnw[0] = Dsnw[0] + dSice / self.rhof
        if (Sice[0] + dSice > EPS):
            Rgrn[0] = (Sice[0]*Rgrn[0] + dSice *
                            self.rgr0) / (Sice[0] + dSice)
        Sice[0] = Sice[0] + dSice

        # Add canopy unloading to layer 1 with bulk snow density and grain size
        rhos = self.rhof
        swe = sum(Sice[:]) + sum(Sliq[:])
        hs = sum(Dsnw[:])
        if (hs > EPS):
            rhos = swe / hs
        Dsnw[0] = Dsnw[0] + unload / rhos
        if (Sice[0] + unload > EPS):
            Rgrn[0] = (Sice[0]*Rgrn[0] + unload *
                            self.rgr0) / (Sice[0] + unload)
        Sice[0] = Sice[0] + unload

        # Add wind-blown snow to layer 1 with wind-packed density and fresh grain size
        dSice = - trans*dt
        if (dSice > EPS):
            Dsnw[0] = Dsnw[0] + dSice / self.rhow
            Rgrn[0] = (Sice[0]*Rgrn[0] + dSice *
                            self.rgr0) / (Sice[0] + dSice)
            Sice[0] = Sice[0] + dSice

        # New snowpack
        if (Nsnow == 0) and (Sice[0] > EPS):
            Nsnow = 1
            Rgrn[0] = self.rgr0
            Tsnow[0] = min(Ta, T_MELT)

        # Store state of old layers
        D = Dsnw[:].copy()
        R = Rgrn[:].copy()
        S = Sice[:].copy()
        W = Sliq[:].copy()

        if Nsnow > 0:
            for k in range(Nsnow):
                self.csnow[k] = Sice[k]*SPECIFIC_HEAT_ICE + \
                    Sliq[k]*SPECIFIC_HEAT_H2O
                self.E[k] = self.csnow[k]*(Tsnow[k] - T_MELT)
        Nold = Nsnow
        hs = sum(Dsnw[:])

        # Initialise new layers
        Dsnw[:] = 0.0
        Rgrn[:] = self.rgr0
        Sice[:] = 0.0
        Sliq[:] = 0.0
        Tsnow[:] = T_MELT
        self.U[:] = 0.0
        Nsnow = 0

        if (hs > EPS):  # Existing or new snowpack
            # Re-assign and count snow layers
            dnew = hs
            Dsnw[0] = dnew
            k = 0 # means one layer
            if (Dsnw[0] > self.Dzsnow[0]):
                for k in range(self.Nsmax):
                    Dsnw[k] = self.Dzsnow[k]
                    dnew = dnew - self.Dzsnow[k]
                    if (dnew <= self.Dzsnow[k]) or (k == self.Nsmax - 1):
                        Dsnw[k] = Dsnw[k] + dnew
                        break
            Nsnow = k + 1

            # Fill new layers from the top downwards
            knew = 0
            dnew = Dsnw[0]

            for kold in range(Nold):
                while True:
                    # Ensure we don't try to access out-of-bounds indices
                    if knew > Nsnow-1:
                        break  # Exit the loop if we've reached the maximum number of new layers
                    if D[kold] < dnew:
                        # All snow from old layer partially fills new layer
                        Rgrn[knew] += S[kold] * R[kold]
                        Sice[knew] += S[kold]
                        Sliq[knew] += W[kold]
                        self.U[knew] += self.E[kold]
                        dnew -= D[kold]
                        break  # Exit the inner while loop
                    else:
                        # Some snow from old layer fills new layer
                        wt = dnew / D[kold]
                        Rgrn[knew] += wt * S[kold] * R[kold]
                        Sice[knew] += wt * S[kold]
                        Sliq[knew] += wt * W[kold]
                        self.U[knew] += wt * self.E[kold]
                        D[kold] *= (1 - wt)
                        self.E[kold] *= (1 - wt)
                        S[kold] *= (1 - wt)
                        W[kold] *= (1 - wt)
                        knew += 1
                        if knew > Nsnow-1:
                            break  # Exit the inner while loop
                        dnew = Dsnw[knew]  # Move to the next new layer

        # Diagnose snow layer temperatures
        for k in range(Nsnow):
            self.csnow[k] = Sice[k]*SPECIFIC_HEAT_ICE + \
                Sliq[k]*SPECIFIC_HEAT_H2O
            Tsnow[k] = T_MELT + self.U[k] / self.csnow[k]
            Rgrn[k] = Rgrn[k] / Sice[k]

        # Drain, retain or freeze snow in layers
        if self.HYDRL == 0:
            # Free-draining snow, no retention or freezing
            Wflx[0] = Roff
            for k in range(Nsnow):
                Roff = Roff + Sliq[k] / dt
                Sliq[k] = 0
                if (k < Nsnow - 1):
                    Wflx[k+1] = Roff

        if self.HYDRL == 1:
            # Bucket storage
            if (np.max(Sliq) > EPS) or (Rf > EPS):
                for k in range(Nsnow):
                    self.phi[k] = 1 - Sice[k]/(ICE_DENSITY * Dsnw[k])
                    SliqMax = WATER_DENSITY * \
                        Dsnw[k] * self.phi[k] * self.Wirr
                    Sliq[k] = Sliq[k] + Roff * dt
                    Wflx[k] = Roff
                    Roff = 0.

                    if (Sliq[k] > SliqMax):       # Liquid capacity exceeded
                        # so drainage to next layer
                        Roff = (Sliq[k] - SliqMax)/dt
                        Sliq[k] = SliqMax

                    self.csnow[k] = Sice[k]*SPECIFIC_HEAT_ICE + \
                        Sliq[k]*SPECIFIC_HEAT_H2O
                    coldcont = self.csnow[k]*(T_MELT - Tsnow[k])

                    if (coldcont > EPS):            # Liquid can freeze
                        dSice = np.minimum(
                            Sliq[k], coldcont / LATENT_HEAT_FUSION)
                        Sliq[k] = Sliq[k] - dSice
                        Sice[k] = Sice[k] + dSice
                        Tsnow[k] = Tsnow[k] + \
                            LATENT_HEAT_FUSION*dSice/self.csnow[k]

        if self.HYDRL == 2:  # NOTE THIS NEEDS TESTING!
            # Gravitational drainage
            if (np.max(Sliq) > EPS) | (Rf > EPS):
                self.Qw[:] = 0.
                self.Qw[0] = Rf/WATER_DENSITY
                Roff = 0.
                for k in range(Nsnow):
                    self.ksat[k] = 0.31*(WATER_DENSITY*GRAVITY/WATER_VISCOCITY) * Rgrn[k]**2 * \
                        np.exp(-7.8 * Sice[k] /
                               (WATER_DENSITY * Dsnw[k]))
                    self.phi[k] = 1 - Sice[k]/(ICE_DENSITY*Dsnw[k])
                    self.thetar[k] = self.Wirr*self.phi[k]
                    self.thetaw[k] = Sliq[k]/(WATER_DENSITY*Dsnw[k])
                    if (self.thetaw[k] > self.phi[k]):
                        Roff = Roff + WATER_DENSITY * \
                            Dsnw[k]*(self.thetaw[k] - self.phi[k])/dt
                        self.thetaw[k] = self.phi[k]
                dth = 0.1*dt
                for i in range(10):  # subdivide timestep NOTE CHECK THIS LATER!
                    self.theta0[:] = self.thetaw[:]
                    for j in range(10):  # Newton-Raphson iteration
                        self.a[:] = 0.
                        self.b[:] = 1/dth
                        if (self.thetaw[0] > self.thetar[0]):
                            self.b[0] = 1/dth + 3*self.ksat[0]*(self.thetaw[0] - self.thetar[0])**2/(
                                self.phi[0] - self.thetar[0])**3 / Dsnw[0]
                            self.Qw[1] = self.ksat[0]*(
                                (self.thetaw[0] - self.thetar[0])/(self.phi[0] - self.thetar[0]))**3
                        self.rhs[0] = (self.thetaw[0] - self.theta0[0]) / \
                            dth + (self.Qw[1] - self.Qw[0])/Dsnw[0]
                        for k in range(1, Nsnow):
                            if (self.thetaw[k-1] > self.thetar[k-1]):
                                self.a[k] = - 3*self.ksat[k-1]*(self.thetaw[k-1] - self.thetar[k-1])**2/(
                                    self.phi[k-1] - self.thetar[k-1])**3 / Dsnw[k-1]
                            if (self.thetaw[k] > self.thetar[k]):
                                self.b[k] = 1/dth + 3*self.ksat[k]*(self.thetaw[k] - self.thetar[k])**2/(
                                    self.phi[k] - self.thetar[k])**3 / Dsnw[k]
                                self.Qw[k+1] = self.ksat[k]*(
                                    (self.thetaw[k] - self.thetar[k])/(self.phi[k] - self.thetar[k]))**3
                            self.rhs[k] = (self.thetaw[k] - self.theta0[k]) / \
                                dth + (self.Qw[k+1] - self.Qw[k]
                                       ) / Dsnw[k]
                        self.dtheta[0] = - self.rhs[0]/self.b[0]
                        for k in range(1, Nsnow):
                            self.dtheta[k] = - \
                                (self.a[k]*self.dtheta[k-1] +
                                 self.rhs[k])/self.b[k]
                        for k in range(Nsnow):
                            self.thetaw[k] = self.thetaw[k] + self.dtheta[k]
                            if (self.thetaw[k] > self.phi[k]):
                                self.Qw[k+1] = self.Qw[k+1] + \
                                    (self.thetaw[k] - self.phi[k]
                                     ) * Dsnw[k]/dth
                                self.thetaw[k] = self.phi[k]
                    Wflx[:] = Wflx[:] + \
                        WATER_DENSITY*self.Qw[0:self.Nsmax]/10
                    Roff = Roff + WATER_DENSITY*self.Qw[Nsnow]/10
                Sliq[:] = WATER_DENSITY * Dsnw[:]*self.thetaw[:]
                for k in range(Nsnow):
                    self.csnow[k] = Sice[k]*SPECIFIC_HEAT_ICE + \
                        Sliq[k]*SPECIFIC_HEAT_H2O
                    coldcont = self.csnow[k]*(T_MELT - Tsnow[k])
                    if (coldcont > EPS):  # Liquid can freeze
                        dSice = min(Sliq[k], coldcont/LATENT_HEAT_FUSION)
                        Sliq[k] = Sliq[k] - dSice
                        Sice[k] = Sice[k] + dSice
                        Tsnow[k] = Tsnow[k] + \
                            LATENT_HEAT_FUSION*dSice/self.csnow[k]

        swe = sum(Sice[:]) + sum(Sliq[:])

        # Treating snow layered outputs so that np.nan if there is no such layer
        Tsnow_out = np.array([np.nan, np.nan, np.nan])
        Tsnow_out[:Nsnow] = Tsnow[:Nsnow]

        # Swe0 - Swe1 = Sf + Rf - Roff - Evap
        wbal = sum(self.Sliq[:] + self.Sice[:]) - sum(Sliq[:] + Sice[:]) - Sf - Rf + Roff + Esrf
        
        # store iteration state
        self.iteration_state =  {'Nsnow': int(Nsnow),
                                 'Sliq': Sliq.copy(),
                                 'Sice': Sice.copy(),
                                 'Tsnow': Tsnow.copy(),
                                 'Dsnw': Dsnw.copy(),
                                 'Rgrn': Rgrn.copy()
                                 }

        fluxes = {'Gsoil': Gsoil,
                  'Roff': Roff,
                  'wbal': wbal,
                  }

        states = {'swe': swe,
                  'hs': hs,
                  'Sice': Sice,
                  'Sliq': Sliq,
                  'Nsnow': Nsnow,
                  'Dsnw': Dsnw,
                  'Tsnow': Tsnow_out,
                  'rhos': rhos
                  }

        # End if existing or new snowpack

        return fluxes, states
