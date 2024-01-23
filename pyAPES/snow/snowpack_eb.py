"""
.. module: snowpack
    :synopsis: APES-model component
.. moduleauthor:: Jari-Pekka Nousu

*Energy balance, layered snowpack, based on FSM2*

"""

import numpy as np
from pyFSM2_MODULES import Constants, Layers, Parameters

class SnowModel:
    def __init__(self):

        constants = Constants()
        layers = Layers()
        params = Parameters(SETPAR=2, DENSITY=0)
        
        # from Constants
        self.g = constants.g # Acceleration due to gravity (m/s^2)
        self.hcap_ice = constants.hcap_ice # Specific heat capacity of ice (J/K/kg)
        self.hcap_wat = constants.hcap_wat # Specific heat capacity of water (J/K/kg)
        self.Lf = constants.Lf # Latent heat of fusion (J/kg)
        self.Ls = constants.Ls # Latent heat of sublimation (J/kg)
        self.mu_wat = constants.mu_wat # Dynamic viscosity of water (kg/m/s)
        self.rho_ice = constants.rho_ice # Density of ice (kg/m^3)
        self.rho_wat = constants.rho_wat # Density of water (kg/m^3)
        self.Tm = constants.Tm # Melting point (K)

        # from Layers
        self.Dzsnow = layers.Dzsnow # Minimum snow layer thicknesses (m)
        self.Dzsoil = layers.Dzsoil # Soil layer thicknesses (m)
        self.Nsmax = layers.Nsmax # Maximum number of snow layers
        self.Nsoil = layers.Nsoil # Number of soil layers

        # from Parameters
        self.eta0 = params.eta0 # Reference snow viscosity (Pa s)
        self.rcld = params.rcld # Maximum density for cold snow (kg/m^3)
        self.rfix = params.rfix # Fixed snow density (kg/m^3)
        self.rgr0 = params.rgr0 # Fresh snow grain radius (m)
        self.rhof = params.rhof # Fresh snow density (kg/m^3)
        self.rhow = params.rhow # Wind-packed snow density (kg/m^3)
        self.rmlt = params.rmlt # Maximum density for melting snow (kg/m^3)
        self.snda = params.snda # Thermal metamorphism parameter (1/s)
        self.trho = params.trho # Snow compaction timescale (s)
        self.Wirr = params.Wirr # Irreducible liquid water content of snow

        self.HYDROL = 2 # NOTE THIS NEEDS TO COME FROM THE NAMELIST!
        self.CONDUCT = 1
        self.DENSITY = 1
        self.dt = 3600
        
        # Model state variables (in/out)
        self.Nsnow = np.zeros(1) # Number of snow layers
        self.Dsnw = np.zeros(self.Nsmax) # Snow layer thicknesses (m)
        self.Rgrn = np.zeros(self.Nsmax) # Snow layer grain radius (m)
        self.Sice = np.zeros(self.Nsmax) # Ice content of snow layers (kg/m^2)
        self.Sliq = np.zeros(self.Nsmax) # Liquid content of snow layers (kg/m^2)
        self.Tsnow = np.zeros(self.Nsmax) # Snow layer temperatures (K)
        self.Tsoil = np.zeros(self.Nsmax) # Soil layer temperatures (K)
        self.Gsoil = np.zeros(1) # Heat flux into soil (W/m^2)
        self.Roff = np.zeros(1) # Runoff from snow (kg/m^2/s)
        self.hs = np.zeros(1) # Snow depth (m)
        self.swe = np.zeros(1) # Total snow mass on ground (kg/m^2)
        self.Wflx = np.zeros(self.Nsmax) # Water flux into snow layer (kg/m^2/s)
        self.i,self.j = np.zeros(1),np.zeros(1) # Hydrology iteration counters
        self.k = np.zeros(1) # Snow layer counter
        self.knew =  np.zeros(1) # New snow layer pointer
        self.kold = np.zeros(1) # Old snow layer pointer
        self.Nold = np.zeros(1) # Previous number of snow layers
        self.coldcont = np.zeros(1) # Layer cold content (J/m^2)
        self.dnew = np.zeros(1) # New snow layer thickness (m)
        self.dSice = np.zeros(1) # Change in layer ice content (kg/m^2)
        self.Esnow = np.zeros(1) # Snow sublimation rate (kg/m^2/s)
        self.ggr = np.zeros(1) # Grain area growth rate (m^2/s)
        self.mass = np.zeros(1) # Mass of overlying snow (kg/m^2)
        self.rhos = np.zeros(1) # Density of snow layer (kg/m^3)
        self.SliqMax = np.zeros(1) # Maximum liquid content for layer (kg/m^2)
        self.wt = np.zeros(1) # Layer weighting
        self.a =  np.zeros(self.Nsmax) # Below-diagonal matrix elements
        self.b = np.zeros(self.Nsmax) # Diagonal matrix elements
        self.c = np.zeros(self.Nsmax) # Above-diagonal matrix elements
        self.csnow = np.zeros(self.Nsmax) # Areal heat capacity of snow (J/K/m^2)
        self.dTs = np.zeros(self.Nsmax) # Temperature increments (k)
        self.D = np.zeros(self.Nsmax) # Layer thickness before adjustment (m)
        self.E = np.zeros(self.Nsmax) # Energy contents before adjustment (J/m^2)
        self.Gs = np.zeros(self.Nsmax) # Thermal conductivity between layers (W/m^2/k)
        self.phi = np.zeros(self.Nsmax) # Porosity of snow layers
        self.rhs = np.zeros(self.Nsmax) # Matrix equation rhs
        self.R = np.zeros(self.Nsmax) # Snow grain radii before adjustment (kg/m^2)
        self.S = np.zeros(self.Nsmax) # Ice contents before adjustment (kg/m^2)
        self.U = np.zeros(self.Nsmax) # Layer internal energy contents (J/m^2)
        self.W = np.zeros(self.Nsmax) # Liquid contents before adjustment (kg/m^2)   
        self.dth = np.zeros(1) # Hydrology timestep (s)
        self.dtheta = np.zeros(self.Nsmax) # Change in liquid water content
        self.ksat = np.zeros(self.Nsmax) # Saturated hydraulic conductivity (m/s)
        self.thetar = np.zeros(self.Nsmax) # Irreducible water content
        self.thetaw = np.zeros(self.Nsmax) # Volumetric liquid water content
        self.theta0 = np.zeros(self.Nsmax) # Liquid water content at start of timestep
        self.Qw = np.zeros(self.Nsmax+1) # Water flux at snow layer boundaruess (m/s)
    
    def run_timestep(self,drip,Esrf,Gsrf,ksoil,Rf,Sf,Ta,trans,Tsrf,unload,Tsoil):
        '''
        '''
 
        # No snow
        self.Gsoil = Gsrf
        self.Roff = Rf + drip/self.dt
        self.Wflx[:] = 0

        # Existing snowpack
        if (self.Nsnow > 0):
            # NOTE here add computation of ksnow with snow_thermal function, then remove from args!
            ksnow = snow_thermal(self.kfix, self.rhof, self.rho_wat, self.Dsnw, self.Nsnow, self.Sice, self.Sliq, self.CONDUCT, self.DENSITY)
            # Heat conduction
            for k in range(self.Nsnow):
                self.csnow[k] = self.Sice[k]*self.hcap_ice + self.Sliq[k]*self.hcap_wat
            if (self.Nsnow == 1):
                Gs[0] = 2 / (self.Dsnw[0]/ksnow[0] + self.Dzsoil[0]/ksoil[0])
                dTs[0] = (Gsrf + self.Gs[0]*(Tsoil[0] - Tsnow[0]))*self.dt / (self.csnow[0] + self.Gs[0]*self.dt)
            else:
                for k in range(self.Nsnow-1):
                      Gs[k] = 2 / (self.Dsnw[k]/ksnow[k] + self.Dsnw[k+1]/ksnow[k+1])
                a[0] = 0
                b[0] = self.csnow[0] + self.Gs[0]*self.dt
                c[0] = - Gs[0]*self.dt
                rhs[0] = (Gsrf - Gs[0]*(Tsnow[0] - Tsnow[1]))*self.dt
                for k in range(1,self.Nsnow-1):
                    a[k] = c[k-1]
                    b[k] = self.csnow[k] + (Gs[k-1] + Gs[k])*self.dt
                    c[k] = - Gs[k]*self.dt
                    rhs[k] = Gs[k-1]*(Tsnow[k-1] - Tsnow[k])*self.dt + Gs[k]*(Tsnow[k+1] - Tsnow[k])*self.dt 

            k = self.Nsnow
            Gs[k] = 2 / (Dsnw[k]/ksnow[k] + Dzsoil[0]/ksoil[0])
            a[k] = c[k-1]
            b[k] = self.csnow[k] + (Gs[k-1] + Gs[k])*self.dt
            c[k] = 0
            rhs[k] = Gs[k-1]*(Tsnow[k-1] - Tsnow[k])*self.dt + Gs[k]*(Tsoil[0] - Tsnow[k])*self.dt
            dTs = self.tridiag(self.Nsnow,Nsmax,a,b,c,rhs)
            
            for k in range(self.Nsnow):
                Tsnow[k] = Tsnow[k] + dTs[k]
            k = self.Nsnow
            self.Gsoil = Gs[k]*(Tsnow[k] - Tsoil[0])
        
            # Convert melting ice to liquid water
            dSice = Melt*self.dt
            for k in range(self.Nsnow):
                coldcont = self.csnow[k]*(Tm - Tsnow[k])
                if (coldcont < 0):
                    dSice = dSice - coldcont/Lf
                    Tsnow[k] = self.Tm
                if (dSice > 0):
                    if (dSice > self.Sice[k]):  # Layer melts completely
                        dSice = dSice - self.Sice[k]
                        self.Dsnw[k] = 0
                        self.Sliq[k] = self.Sliq[k] + self.Sice[k]
                        self.Sice[k] = 0
                    else:                       # Layer melts partially
                        self.Dsnw[k] = (1 - dSice/self.Sice[k])*Dsnw[k]
                        self.Sice[k] = self.Sice[k] - dSice
                        self.Sliq[k] = self.Sliq[k] + dSice
                        dSice = 0
                    
            # Remove snow by sublimation 
            dSice = Esrf*self.dt
            if (dSice > 0):
                for k in range(self.Nsnow):
                    if (dSice > self.Sice[k]):  # Layer sublimates completely
                        dSice = dSice - self.Sice[k]
                        self.Dsnw[k] = 0
                        self.Sice[k] = 0
                    else:                       # Layer sublimates partially
                        self.Dsnw[k] = (1 - dSice/self.Sice[k])*Dsnw[k]
                        self.Sice[k] = self.Sice[k] - dSice
                        dSice = 0

            # Remove wind-trasported snow 
            dSice = trans*self.dt
            if (dSice > 0):
                for k in range(self.Nsnow):
                    if (dSice > self.Sice[k]):  # Layer completely removed
                        dSice = dSice - self.Sice[k]
                        self.Dsnw[k] = 0
                        self.Sice[k] = 0
                    else:                       # Layer partially removed
                        self.Dsnw[k] = (1 - dSice/self.Sice[k])*Dsnw[k]
                        self.Sice[k] = self.Sice[k] - dSice
                        dSice = 0

            if DENSITY == 0:
                # Fixed snow density
                for k in range(self.Nsnow):
                    if (Dsnw[k] > np.finfo(float).eps):
                        self.Dsnw[k] = (self.Sice[k] + self.Sliq[k]) / rfix
            if DENSITY == 1:
                # Snow compaction with age
                for k in range(self.Nsnow):
                    if Dsnw[k] > np.finfo(float).eps: # epsillon different in FSM
                        rhos = (self.Sice[k] + self.Sliq[k]) / Dsnw[k]
                        if Tsnow[k] >= self.Tm:
                            if rhos < rmlt:
                                rhos = rmlt + (rhos - rmlt) * np.exp(-self.dt / trho)
                        else:
                            if rhos < rcld:
                                rhos = rcld + (rhos - rcld) * np.exp(-self.dt / trho)
                        self.Dsnw[k] = (self.Sice[k] + self.Sliq[k]) / rhos
                        
            if DENSITY == 2:
                # Snow compaction by overburden
                mass = 0
                for k in range(self.Nsnow):
                    mass = mass + 0.5*(self.Sice[k] + self.Sliq[k]) 
                    if (Dsnw[k] > np.finfo(float).eps):
                        rhos = (self.Sice[k] + self.Sliq[k]) / Dsnw[k]
                        rhos = rhos + (rhos*g*mass*self.dt/(eta0*exp(-(Tsnow[k] - self.Tm)/12.4 + rhos/55.6)) + self.dt*rhos*snda*exp((Tsnow[k] - self.Tm)/23.8 - max(rhos - 150, 0.)/21.7))
                        self.Dsnw[k] = (self.Sice[k] + self.Sliq[k]) / rhos
                    mass = mass + 0.5*(self.Sice[k] + self.Sliq[k])

            # Snow grain growth
            for k in range(self.Nsnow):
                ggr = 2e-13
                if (Tsnow[k] < self.Tm):
                    if (Rgrn[k] < 1.50e-4):
                        ggr = 2e-14
                    else:
                        ggr = 7.3e-8*exp(-4600/Tsnow[k])
                Rgrn[k] = Rgrn[k] + dt*ggr/Rgrn[k]

        # End if for existing snowpack

        # Add snowfall and frost to layer 1 with fresh snow density and grain size
        Esnow = 0
        if (Esrf < 0 & Tsrf < self.Tm):
            Esnow = Esrf
        dSice = (Sf - Esnow)*self.dt
        self.Dsnw[0] = self.Dsnw[0] + dSice / self.rhof
        if (self.Sice[0] + dSice > np.finfo(float).eps):
            self.Rgrn[0] = (self.Sice[0]*self.Rgrn[0] + dSice*self.rgr0) / (self.Sice[0] + dSice)
        self.Sice[0] = self.Sice[0] + dSice
    
        # Add canopy unloading to layer 1 with bulk snow density and grain size
        rhos = self.rhof
        swe = sum(self.Sice[:]) + sum(self.Sliq[:])
        hs = sum(self.Dsnw[:])
        if (hs > np.finfo(float).eps):
            rhos = swe / hs
        self.Dsnw[0] = self.Dsnw[0] + unload / rhos
        if (self.Sice[0] + unload > np.finfo(float).eps):
            self.Rgrn[0] = (self.Sice[0]*self.Rgrn[0] + unload*self.rgr0) / (self.Sice[0] + unload)
        self.Sice[0] = self.Sice[0] + unload

        # Add wind-blown snow to layer 1 with wind-packed density and fresh grain size
        dSice = - trans*self.dt
        if (dSice > 0):
            self.Dsnw[0] = self.Dsnw[0] + dSice / rhow
            self.Rgrn[0] = (self.Sice[0]*self.Rgrn[0] + dSice*self.rgr0) / (self.Sice[0] + dSice)
            self.Sice[0] = self.Sice[0] + dSice

        # New snowpack
        if (self.Nsnow == 0) & (self.Sice[0] > 0):
            self.Nsnow = 1
            self.Rgrn[0] = self.rgr0
            self.Tsnow[0] = min(Ta, self.Tm)

        # Store state of old layers
        self.D[:] = self.Dsnw[:]
        self.R[:] = self.Rgrn[:]
        self.S[:] = self.Sice[:]
        self.W[:] = self.Sliq[:]
        if self.Nsnow > 0:
            for k in range(self.Nsnow):
                self.csnow[k] = self.Sice[k]*self.hcap_ice + self.Sliq[k]*self.hcap_wat
                self.E[k] = self.csnow[k]*(self.Tsnow[k] - self.Tm)
        Nold = self.Nsnow
        hs = sum(self.Dsnw[:])

        # Initialise new layers
        self.Dsnw[:] = 0
        self.Rgrn[:] = 0
        self.Sice[:] = 0
        self.Sliq[:] = 0
        self.Tsnow[:] = self.Tm
        self.U[:] = 0
        self.Nsnow = 0

        if (hs > 0):  # Existing or new snowpack
            # Re-assign and count snow layers
            dnew = hs
            self.Dsnw[0] = dnew
            k = 1
            if (self.Dsnw[0] > self.Dzsnow[0]):
                for k in range(self.Nsmax):
                    self.Dsnw[k] = self.Dzsnow[k]
                    dnew = dnew - self.Dzsnow[k]
                    if (dnew <= self.Dzsnow[k]) | (k == self.Nsmax):
                        self.Dsnw[k] = Dsnw[k] + dnew
                        break
            self.Nsnow = k

            # Fill new layers from the top downwards
            knew = 1
            dnew = self.Dsnw[0]
            for kold in range(Nold):
                if (self.D[kold] < dnew):
                    # All snow from old layer partially fills new layer
                    self.Rgrn[knew] = self.Rgrn[knew] + self.S[kold]*self.R[kold]
                    self.Sice[knew] = self.Sice[knew] + self.S[kold]
                    self.Sliq[knew] = self.Sliq[knew] + self.W[kold]
                    self.U[knew] = self.U[knew] + self.E[kold]
                    dnew = dnew - self.D[kold]
                    break
                else:
                    # Some snow from old layer fills new layer
                    wt = dnew / self.D[kold]
                    self.Rgrn[knew] = self.Rgrn[knew] + wt*self.S[kold]*self.R[kold]
                    self.Sice[knew] = self.Sice[knew] + wt*self.S[kold]
                    self.Sliq[knew] = self.Sliq[knew] + wt*self.W[kold]
                    self.U[knew] = self.U[knew] + wt*self.E[kold]
                    self.D[kold] = (1 - wt)*self.D[kold]
                    self.E[kold] = (1 - wt)*self.E[kold]
                    self.S[kold] = (1 - wt)*self.S[kold]
                    self.W[kold] = (1 - wt)*self.W[kold]
                    knew = knew + 1
                    if (knew > self.Nsnow):
                        break
                    dnew = self.Dsnw[knew]

        # Diagnose snow layer temperatures
        for k in range(self.Nsnow):
            self.csnow[k] = self.Sice[k]*self.hcap_ice + self.Sliq[k]*self.hcap_wat
            self.Tsnow[k] = self.Tm + self.U[k] / self.csnow[k]
            self.Rgrn[k] = self.Rgrn[k] / self.Sice[k]

        # Drain, retain or freeze snow in layers
        if self.HYDROL == 0:
            # Free-draining snow, no retention or freezing 
            self.Wflx[0] = self.Roff
            for k in range(self.Nsnow):
                self.Roff = self.Roff + self.Sliq[k] / self.dt
                self.Sliq[k] = 0
                if (k < self.Nsnow):
                    self.Wflx[k+1] = self.Roff

        if self.HYDROL == 1:
            # Bucket storage 
            if (np.max(self.Sliq)) > 0 | (Rf > 0):
                for k in range(self.Nsnow):
                    self.phi[k] = 1 - self.Sice[k]/(rho_ice*Dsnw[k])
                    self.SliqMax = rho_wat*Dsnw[k]*self.phi[k]*self.Wirr
                    self.Sliq[k] = self.Sliq[k] + Roff*self.dt
                    Wflx[k] = Roff
                    Roff = 0
                if (self.Sliq[k] > self.SliqMax):       # Liquid capacity exceeded
                    Roff = (self.Sliq[k] - self.SliqMax)/self.dt   # so drainage to next layer
                    self.Sliq[k] = self.SliqMax
                self.csnow[k] = self.Sice[k]*self.hcap_ice + self.Sliq[k]*self.hcap_wat
                coldcont = self.csnow[k]*(self.Tm - Tsnow[k])
                if (coldcont > 0):            # Liquid can freeze
                    dSice = min(self.Sliq[k], coldcont/Lf)
                    self.Sliq[k] = self.Sliq[k] - dSice
                    self.Sice[k] = self.Sice[k] + dSice
                    Tsnow[k] = Tsnow[k] + Lf*dSice/self.csnow[k]


        if self.HYDROL == 2: # NOTE THIS NEEDS TESTING!
            # Gravitational drainage 
            if (np.max(self.Sliq) > 0 | Rf > 0):
                Qw[:] = 0
                Qw[0] = Rf/rho_wat
                Roff = 0
                for k in range(self.Nsnow):
                    ksat[k] = 0.31*(self.rho_wat*self.g/self.mu_wat)*self.Rgrn[k]**2*exp(-7.8*self.Sice[k]/(rho_wat*Dsnw[k]))
                    self.phi[k] = 1 - self.Sice[k]/(self.rho_ice*self.Dsnw[k])
                    thetar[k] = self.Wirr*self.phi[k]
                    thetaw[k] = self.Sliq[k]/(rho_wat*Dsnw[k])
                    if (thetaw[k]>self.phi[k]):
                        Roff = Roff + rho_wat*Dsnw[k]*(thetaw[k] - self.phi[k])/self.dt
                        thetaw[k] = self.phi[k]
                dth = 0.1*self.dt
                for i in range(10): # subdivide timestep NOTE CHECK THIS LATER!
                    theta0[:] = thetaw[:]
                    for j in range(10): # Newton-Raphson iteration
                        a[:] = 0
                        b[:] = 1/dth 
                        if (thetaw[0] > thetar[0]):
                            b[0] = 1/dth + 3*ksat[0]*(thetaw[0] - thetar[0])**2/(self.phi[0] - thetar[0])**3/Dsnw[0]
                            Qw[1] = ksat[0]*((thetaw[0] - thetar[0])/(self.phi[0] - thetar[0]))**3
                        rhs[0] = (thetaw[0] - theta0[0])/dth + (Qw[1] - Qw[0])/Dsnw[0]
                        for k in range(1, self.Nsnow):
                            if (thetaw[k-1] > thetar[k-1]):
                                a[k] = - 3*ksat[k-1]*(thetaw[k-1] - thetar[k-1])**2/(self.phi[k-1] - thetar[k-1])**3/Dsnw[k-1]
                            if (thetaw[k] > thetar[k]):
                                b[k] = 1/dth + 3*ksat[k]*(thetaw[k] - thetar[k])**2/(self.phi[k] - thetar[k])**3/Dsnw[k]
                                Qw[k+1] = ksat[k]*((thetaw[k] - thetar[k])/(self.phi[k] - thetar[k]))**3
                            rhs[k] = (thetaw[k] - theta0[k])/dth + (Qw[k+1] - Qw[k])/Dsnw[k]
                        dtheta[0] = - rhs[0]/b[0]
                        for k in range(1, self.Nsnow):
                            dtheta[k] = - (a[k]*dtheta[k-1] + rhs[k])/b[k]
                        for k in range(self.Nsnow):
                            thetaw[k] = thetaw[k] + dtheta[k]
                            if (thetaw[k] > self.phi[k]):
                                Qw[k+1] = Qw[k+1] + (thetaw[k] - self.phi[k])*Dsnw[k]/dth
                                thetaw[k] = self.phi[k]
                    Wflx[:] = Wflx[:] + rho_wat*Qw[0:Nsmax]/10
                    Roff = Roff + rho_wat*Qw[self.Nsnow+1]/10
                self.Sliq[:] = rho_wat*Dsnw[:]*thetaw[:]
                for k in range(self.Nsnow):
                    self.csnow[k] = self.Sice[k]*self.hcap_ice + self.Sliq[k]*self.hcap_wat
                    coldcont = self.csnow[k]*(Tm - Tsnow[k])
                    if (coldcont > 0): # Liquid can freeze
                        dSice = min(self.Sliq[k], coldcont/Lf)
                        self.Sliq[k] = self.Sliq[k] - dSice
                        self.Sice[k] = self.Sice[k] + dSice
                        Tsnow[k] = Tsnow[k] + Lf*dSice/self.csnow[k]

        self.swe = sum(self.Sice[:]) + sum(self.Sliq[:])
        # End if existing or new snowpack

        # NOTE SHOULD SAVE RESULTS BETTER!
        return self.Gsoil, self.Roff, self.hs, self.swe, self.Wflx

    def tridiag(Nvec,Nmax,a,b,c,r,x):
        '''
        Input
        Nvec: Vector length
        Nmax: Maximum vector length
        a: Below-diagonal matrix elements
        b: Diagonal matrix elements
        c: Above-diagonal matrix elements
        r: Matrix equation rhs
        
        Output
        x: Solution vector
        '''

        beta = b[0]
        x[0] = r[0] / beta

        for n in range(1, Nvec):
            g[n] = c[n-1] / beta
            beta = b[n] - a[n] * g[n]
            x[n] = (r[n] - a[n] * x[n - 1]) / beta

        for n in range(Nvec - 1, 0, -1):
            x[n] = x[n] - g[n + 1] * x[n + 1]

        return x

    def snow_thermal(kfix, rhof, rho_wat, Dsnw, Nsnow, Sice, Sliq, CONDUCT, DENSITY):
        '''
        Thermal conductivity of snow
        '''
        
        ksnow = kfix
        if CONDUCT == 1:
            for k in range(Nsnow):
                rhos = rhof
            if DENSITY == 1:
                if (Dsnw[k] > np.finfo(float).eps):
                    rhos = (Sice[k] + Sliq[k]) / Dsnw[k]
                ksnow[k] = 2.224*(rhos/rho_wat)**1.885
                
        return ksnow
