"""
.. module: snowpack
    :synopsis: APES-model component
.. moduleauthor:: Jari-Pekka Nousu

*Energy balance, layered snowpack, based on FSM*

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
        self.eta0 = parameters.eta0 # Reference snow viscosity (Pa s)
        self.rcld = parameters.rcld # Maximum density for cold snow (kg/m^3)
        self.rfix = parameters.rfix # Fixed snow density (kg/m^3)
        self.rgr0 = parameters.rgr0 # Fresh snow grain radius (m)
        self.rhof = parameters.rhof # Fresh snow density (kg/m^3)
        self.rhow = parameters.rhow # Wind-packed snow density (kg/m^3)
        self.rmlt = parameters.rmlt # Maximum density for melting snow (kg/m^3)
        self.snda = parameters.snda # Thermal metamorphism parameter (1/s)
        self.trho = parameters.trho # Snow compaction timescale (s)
        self.Wirr = parameters.Wirr # Irreducible liquid water content of snow
        
        # Model state variables (in/out)
        self.Nsnow = np.zeros(Nsmax) # Number of snow layers
        self.Dsnw = np.zeros(Nsmax) # Snow layer thicknesses (m)
        self.Rgrn = np.zeros(Nsmax) # Snow layer grain radius (m)
        self.Sice = np.zeros(Nsmax) # Ice content of snow layers (kg/m^2)
        self.Sliq = np.zeros(Nsmax) # Liquid content of snow layers (kg/m^2)
        self.Tsnow = np.zeros(Nsmax) # Snow layer temperatures (K)
        self.Tsoil = np.zeros(Nsmax) # Soil layer temperatures (K)
        self.Gsoil = np.zeros(1) # Heat flux into soil (W/m^2)
        self.Roff = np.zeros(1) # Runoff from snow (kg/m^2/s)
        self.hs = np.zeros(1) # Snow depth (m)
        self.swe = np.zeros(1) # Total snow mass on ground (kg/m^2)
        self.Wflx = np.zeros(Nsmax) # Water flux into snow layer (kg/m^2/s)
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
        self.a =  np.zeros(Nsmax) # Below-diagonal matrix elements
        self.b = np.zeros(Nsmax) # Diagonal matrix elements
        self.c = np.zeros(Nsmax) # Above-diagonal matrix elements
        self.csnow = np.zeros(Nsmax) # Areal heat capacity of snow (J/K/m^2)
        self.dTs = np.zeros(Nsmax) # Temperature increments (k)
        self.D = np.zeros(Nsmax) # Layer thickness before adjustment (m)
        self.E = np.zeros(Nsmax) # Energy contents before adjustment (J/m^2)
        self.Gs = np.zeros(Nsmax) # Thermal conductivity between layers (W/m^2/k)
        self.phi = np.zeros(Nsmax) # Porosity of snow layers
        self.rhs = np.zeros(Nsmax) # Matrix equation rhs
        self.R = np.zeros(Nsmax) # Snow grain radii before adjustment (kg/m^2)
        self.S = np.zeros(Nsmax) # Ice contents before adjustment (kg/m^2)
        self.U = np.zeros(Nsmax) # Layer internal energy contents (J/m^2)
        self.W = np.zeros(Nsmax) # Liquid contents before adjustment (kg/m^2)   
        self.dth = np.zeros(1) # Hydrology timestep (s)
        self.dtheta = np.zeros(Nsmax) # Change in liquid water content
        self.ksat = np.zeros(Nsmax) # Saturated hydraulic conductivity (m/s)
        self.thetar = np.zeros(Nsmax) # Irreducible water content
        self.thetaw = np.zeros(Nsmax) # Volumetric liquid water content
        self.theta0 = np.zeros(Nsmax) # Liquid water content at start of timestep
        self.Qw = np.zeros(Nsmax+1) # Water flux at snow layer boundaruess (m/s)
    
    def run_timestep(self,dt,drip,Esrf,Gsrf,ksnow,ksoil,Melt,Rf,Sf,Ta,trans,
                    Tsrf,unload,Nsnow,Dsnw,Rgrn,Sice,Sliq,Tsnow,Tsoil,
                    Gsoil,Roff,hs,swe,Wflx):
 
        # No snow
        Gsoil = Gsrf
        Roff = Rf + drip/dt
        Wflx[:] = 0

        # Existing snowpack
        if (self.Nsnow > 0):
            # NOTE here add computation of ksnow with snow_thermal function, then remove from args!
            # Heat conduction
            for k in range(self.Nsnow):
                self.csnow[k] = self.Sice[k]*self.hcap_ice + self.Sliq[k]*self.hcap_wat
            if (self.Nsnow == 1):
                Gs[0] = 2 / (Dsnw[0]/ksnow[0] + Dzsoil[0]/ksoil[0])
                dTs[0] = (Gsrf + Gs[0]*(Tsoil[0] - Tsnow[0]))*dt / (self.csnow[0] + Gs[0]*dt)
            else:
                for k in range(self.Nsnow-1):
                      Gs[k] = 2 / (Dsnw[k]/ksnow[k] + Dsnw[k+1]/ksnow[k+1])
                a[0] = 0
                b[0] = self.csnow[0] + Gs[0]*dt
                c[0] = - Gs[0]*dt
                rhs[0] = (Gsrf - Gs[0]*(Tsnow[0] - Tsnow[1]))*dt
                for k in range(1,self.Nsnow-1):
                    a[k] = c[k-1]
                    b[k] = self.csnow[k] + (Gs[k-1] + Gs[k])*dt
                    c[k] = - Gs[k]*dt
                    rhs[k] = Gs[k-1]*(Tsnow[k-1] - Tsnow[k])*dt + Gs[k]*(Tsnow[k+1] - Tsnow[k])*dt 

            k = self.Nsnow
            Gs[k] = 2 / (Dsnw[k]/ksnow[k] + Dzsoil[0]/ksoil[0])
            a[k] = c[k-1]
            b[k] = self.csnow[k] + (Gs[k-1] + Gs[k])*dt
            c[k] = 0
            rhs[k] = Gs[k-1]*(Tsnow[k-1] - Tsnow[k])*dt + Gs[k]*(Tsoil[0] - Tsnow[k])*dt
            dTs = self.tridiag(self.Nsnow,Nsmax,a,b,c,rhs)
            
            for k in range(self.Nsnow):
                Tsnow[k] = Tsnow[k] + dTs[k]
            k = self.Nsnow
            Gsoil = Gs[k]*(Tsnow[k] - Tsoil[0])
        
            # Convert melting ice to liquid water
            dSice = Melt*dt
            for k in range(self.Nsnow):
                coldcont = self.csnow[k]*(Tm - Tsnow[k])
                if (coldcont < 0):
                    dSice = dSice - coldcont/Lf
                    Tsnow[k] = Tm
                if (dSice > 0):
                    if (dSice > self.Sice[k]):  # Layer melts completely
                        dSice = dSice - self.Sice[k]
                        Dsnw[k] = 0
                        self.Sliq[k] = self.Sliq[k] + self.Sice[k]
                        self.Sice[k] = 0
                    else:                       # Layer melts partially
                        Dsnw[k] = (1 - dSice/self.Sice[k])*Dsnw[k]
                        self.Sice[k] = self.Sice[k] - dSice
                        self.Sliq[k] = self.Sliq[k] + dSice
                        dSice = 0
                    
            # Remove snow by sublimation 
            dSice = Esrf*dt
            if (dSice > 0):
                for k in range(self.Nsnow):
                    if (dSice > self.Sice[k]):  # Layer sublimates completely
                        dSice = dSice - self.Sice[k]
                        Dsnw[k] = 0
                        self.Sice[k] = 0
                    else:                       # Layer sublimates partially
                        Dsnw[k] = (1 - dSice/self.Sice[k])*Dsnw[k]
                        self.Sice[k] = self.Sice[k] - dSice
                        dSice = 0

            # Remove wind-trasported snow 
            dSice = trans*dt
            if (dSice > 0):
                for k in range(self.Nsnow):
                    if (dSice > self.Sice[k]):  # Layer completely removed
                        dSice = dSice - self.Sice[k]
                        Dsnw[k] = 0
                        self.Sice[k] = 0
                    else:                       # Layer partially removed
                        Dsnw[k] = (1 - dSice/self.Sice[k])*Dsnw[k]
                        self.Sice[k] = self.Sice[k] - dSice
                        dSice = 0

            if DENSITY == 0:
                # Fixed snow density
                for k in range(self.Nsnow):
                    if (Dsnw[k] > np.finfo(float).eps):
                        Dsnw[k] = (self.Sice[k] + self.Sliq[k]) / rfix
            if DENSITY == 1:
                # Snow compaction with age
                for k in range(self.Nsnow):
                    if Dsnw[k] > np.finfo(float).eps: # epsillon different in FSM
                        rhos = (self.Sice[k] + self.Sliq[k]) / Dsnw[k]
                        if Tsnow[k] >= Tm:
                            if rhos < rmlt:
                                rhos = rmlt + (rhos - rmlt) * np.exp(-dt / trho)
                        else:
                            if rhos < rcld:
                                rhos = rcld + (rhos - rcld) * np.exp(-dt / trho)
                        Dsnw[k] = (self.Sice[k] + self.Sliq[k]) / rhos
                        
            if DENSITY == 2:
                # Snow compaction by overburden
                mass = 0
                for k in range(self.Nsnow):
                    mass = mass + 0.5*(self.Sice[k] + self.Sliq[k]) 
                    if (Dsnw[k] > np.finfo(float).eps):
                        rhos = (self.Sice[k] + self.Sliq[k]) / Dsnw[k]
                        rhos = rhos + (rhos*g*mass*dt/(eta0*exp(-(Tsnow[k] - Tm)/12.4 + rhos/55.6)) + dt*rhos*snda*exp((Tsnow[k] - Tm)/23.8 - max(rhos - 150, 0.)/21.7))
                        Dsnw[k] = (self.Sice[k] + self.Sliq[k]) / rhos
                    mass = mass + 0.5*(self.Sice[k] + self.Sliq[k])

            # Snow grain growth
            for k in range(self.Nsnow):
                ggr = 2e-13
                if (Tsnow[k] < Tm):
                    if (Rgrn[k] < 1.50e-4):
                        ggr = 2e-14
                    else:
                        ggr = 7.3e-8*exp(-4600/Tsnow[k])
                Rgrn[k] = Rgrn[k] + dt*ggr/Rgrn[k]

        # End if for existing snowpack

        # Add snowfall and frost to layer 1 with fresh snow density and grain size
        Esnow = 0
        if (Esrf < 0 & Tsrf < Tm):
            Esnow = Esrf
        dSice = (Sf - Esnow)*dt
        Dsnw[0] = Dsnw[0] + dSice / rhof
        if (self.Sice[0] + dSice > np.finfo(float).eps):
            Rgrn[0] = (self.Sice[0]*Rgrn[0] + dSice*rgr0) / (self.Sice[0] + dSice)
        self.Sice[0] = self.Sice[0] + dSice
    
        # Add canopy unloading to layer 1 with bulk snow density and grain size
        rhos = rhof
        swe = sum(self.Sice[:]) + sum(self.Sliq[:])
        hs = sum(Dsnw[:])
        if (hs > np.finfo(float).eps):
            rhos = swe / hs
        Dsnw[0] = Dsnw[0] + unload / rhos
        if (self.Sice[0] + unload > np.finfo(float).eps):
            Rgrn[0] = (self.Sice[0]*Rgrn[0] + unload*rgr0) / (self.Sice[0] + unload)
        self.Sice[0] = self.Sice[0] + unload

        # Add wind-blown snow to layer 1 with wind-packed density and fresh grain size
        dSice = - trans*dt
        if (dSice > 0):
            Dsnw[0] = Dsnw[0] + dSice / rhow
            Rgrn[0] = (self.Sice[0]*Rgrn[0] + dSice*rgr0) / (self.Sice[0] + dSice)
            self.Sice[0] = self.Sice[0] + dSice

        # New snowpack
        if (self.Nsnow == 0 & self.Sice[0] > 0):
            self.Nsnow = 1
            Rgrn[0] = rgr0
            Tsnow[0] = min(Ta, Tm)


        # Store state of old layers
        D[:] = Dsnw[:]
        R[:] = Rgrn[:]
        S[:] = self.Sice[:]
        W[:] = self.Sliq[:]
        for k in range(self.Nsnow):
            self.csnow[k] = self.Sice[k]*self.hcap_ice + self.Sliq[k]*self.hcap_wat
            E[k] = self.csnow[k]*(Tsnow[k] - Tm)
        Nold = self.Nsnow
        hs = sum(Dsnw[:])

        # Initialise new layers
        Dsnw[:] = 0
        Rgrn[:] = 0
        self.Sice[:] = 0
        self.Sliq[:] = 0
        Tsnow[:] = Tm
        U[:] = 0
        self.Nsnow = 0

        if (hs > 0):  # Existing or new snowpack
            # Re-assign and count snow layers
            dnew = hs
            Dsnw[0] = dnew
            k = 1
            if (Dsnw[0] > Dzsnow[0]):
                for k in range(Nsmax):
                    Dsnw[k] = Dzsnow[k]
                    dnew = dnew - Dzsnow[k]
                    if (dnew <= Dzsnow[k] | k == Nsmax):
                        Dsnw[k] = Dsnw[k] + dnew
                        break
            self.Nsnow = k

            # Fill new layers from the top downwards
            knew = 1
            dnew = Dsnw[0]
            for kold in range(Nold):
                if (D[kold] < dnew):
                    # All snow from old layer partially fills new layer
                    Rgrn[knew] = Rgrn[knew] + S[kold]*R[kold]
                    self.Sice[knew] = self.Sice[knew] + S[kold]
                    self.Sliq[knew] = self.Sliq[knew] + W[kold]
                    U[knew] = U[knew] + E[kold]
                    dnew = dnew - D[kold]
                    break
                else:
                    # Some snow from old layer fills new layer
                    wt = dnew / D[kold]
                    Rgrn[knew] = Rgrn[knew] + wt*S[kold]*R[kold]
                    self.Sice[knew] = self.Sice[knew] + wt*S[kold]
                    self.Sliq[knew] = self.Sliq[knew] + wt*W[kold]
                    U[knew] = U[knew] + wt*E[kold]
                    D[kold] = (1 - wt)*D[kold]
                    E[kold] = (1 - wt)*E[kold]
                    S[kold] = (1 - wt)*S[kold]
                    W[kold] = (1 - wt)*W[kold]
                    knew = knew + 1
                    if (knew > self.Nsnow):
                        break
                    dnew = Dsnw[knew]

        # Diagnose snow layer temperatures
        for k in range(self.Nsnow):
            self.csnow[k] = self.Sice[k]*self.hcap_ice + self.Sliq[k]*self.hcap_wat
            Tsnow[k] = Tm + U[k] / self.csnow[k]
            Rgrn[k] = Rgrn[k] / self.Sice[k]

        # Drain, retain or freeze snow in layers
        if HYDROL == 0:
            # Free-draining snow, no retention or freezing 
            Wflx[0] = Roff
            for k in range(self.Nsnow):
                Roff = Roff + self.Sliq[k] / dt
                self.Sliq[k] = 0
                if (k < self.Nsnow):
                    Wflx[k+1] = Roff

        if HYDROL == 1:
            # Bucket storage 
            if (maxval(self.Sliq)>0 | Rf>0):
                for k in range(self.Nsnow):
                    phi[k] = 1 - self.Sice[k]/(rho_ice*Dsnw[k])
                    self.SliqMax = rho_wat*Dsnw[k]*phi[k]*Wirr
                    self.Sliq[k] = self.Sliq[k] + Roff*dt
                    Wflx[k] = Roff
                    Roff = 0
                if (self.Sliq[k] > self.SliqMax):       # Liquid capacity exceeded
                    Roff = (self.Sliq[k] - self.SliqMax)/dt   # so drainage to next layer
                    self.Sliq[k] = self.SliqMax
                self.csnow[k] = self.Sice[k]*self.hcap_ice + self.Sliq[k]*self.hcap_wat
                coldcont = self.csnow[k]*(Tm - Tsnow[k])
                if (coldcont > 0):            # Liquid can freeze
                    dSice = min(self.Sliq[k], coldcont/Lf)
                    self.Sliq[k] = self.Sliq[k] - dSice
                    self.Sice[k] = self.Sice[k] + dSice
                    Tsnow[k] = Tsnow[k] + Lf*dSice/self.csnow[k]


        if HYDROL == 2: # NOTE THIS NEEDS TESTING!
            # Gravitational drainage 
            if (np.max(self.Sliq) > 0 | Rf > 0):
                Qw[:] = 0
                Qw[0] = Rf/rho_wat
                Roff = 0
                for k in range(self.Nsnow):
                    ksat[k] = 0.31*(rho_wat*g/mu_wat)*Rgrn[k]**2*exp(-7.8*self.Sice[k]/(rho_wat*Dsnw[k]))
                    phi[k] = 1 - self.Sice[k]/(rho_ice*Dsnw[k])
                    thetar[k] = Wirr*phi[k]
                    thetaw[k] = self.Sliq[k]/(rho_wat*Dsnw[k])
                    if (thetaw[k]>phi[k]):
                        Roff = Roff + rho_wat*Dsnw[k]*(thetaw[k] - phi[k])/dt
                        thetaw(k) = phi(k)
                dth = 0.1*dt
                for i in range(10): # subdivide timestep NOTE CHECK THIS LATER!
                    theta0[:] = thetaw[:]
                    for j in range(10): # Newton-Raphson iteration
                        a[:] = 0
                        b[:] = 1/dth 
                        if (thetaw[0] > thetar[0]):
                            b[0] = 1/dth + 3*ksat[0]*(thetaw[0] - thetar[0])**2/(phi[0] - thetar[0])**3/Dsnw[0]
                            Qw[1] = ksat[0]*((thetaw[0] - thetar[0])/(phi[0] - thetar[0]))**3
                        rhs[0] = (thetaw[0] - theta0[0])/dth + (Qw[1] - Qw[0])/Dsnw[0]
                        for k in range(1, self.Nsnow):
                            if (thetaw[k-1] > thetar[k-1]):
                                a[k] = - 3*ksat[k-1]*(thetaw[k-1] - thetar[k-1])**2/(phi[k-1] - thetar[k-1])**3/Dsnw[k-1]
                            if (thetaw[k] > thetar[k]):
                                b[k] = 1/dth + 3*ksat[k]*(thetaw[k] - thetar[k])**2/(phi[k] - thetar[k])**3/Dsnw[k]
                                Qw(k+1) = ksat(k)*((thetaw(k) - thetar(k))/(phi(k) - thetar(k)))**3
                            rhs[k] = (thetaw[k] - theta0[k])/dth + (Qw[k+1] - Qw[k])/Dsnw[k]
                        dtheta[0] = - rhs[0]/b[0]
                        for k in range(1, self.Nsnow):
                            dtheta[k] = - (a[k]*dtheta[k-1] + rhs[k])/b[k]
                        for k in range(self.Nsnow):
                            thetaw[k] = thetaw[k] + dtheta[k]
                            if (thetaw[k] > phi[k]):
                                Qw[k+1] = Qw[k+1] + (thetaw[k] - phi[k])*Dsnw[k]/dth
                                thetaw[k] = phi[k]
                    Wflx[:] = Wflx[:] + rho_wat*Qw[0:Nsmax]/10
                    Roff = Roff + rho_wat*Qw[self.Nsnow+1]/10
                self.Sliq(:) = rho_wat*Dsnw(:)*thetaw(:)
                for k in range(self.Nsnow):
                    self.csnow[k] = self.Sice[k]*self.hcap_ice + self.Sliq[k]*self.hcap_wat
                    coldcont = self.csnow[k]*(Tm - Tsnow[k])
                    if (coldcont > 0): # Liquid can freeze
                        dSice = min(self.Sliq[k], coldcont/Lf)
                        self.Sliq[k] = self.Sliq[k] - dSice
                        self.Sice[k] = self.Sice[k] + dSice
                        Tsnow[k] = Tsnow[k] + Lf*dSice/self.csnow[k]

        swe = sum(self.Sice[:]) + sum(self.Sliq[:])
        # End if existing or new snowpack


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