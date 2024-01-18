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

        # Model state variables
        self.dt = []
        self.drip = []
        self.Esrf = []
        self.Gsrf = []
        self.Melt = []
        self.Rf = []
        self.Sf = []
        self.Ta = []
        self.trans = []
        self.Tsrf = []
        self.unload = []
        self.ksnow = []
        self.ksoil = []
        self.Nsnow = []
        self.Dsnw = []
        self.Rgrn = []
        self.Sice = []
        self.Sliq = []
        self.Tsnow = []
        self.Tsoil = []
        self.Gsoil = []
        self.Roff = []
        self.hs = []
        self.swe = []
        self.Wflx = []

    def run_timestep(self,dt,drip,Esrf,Gsrf,ksnow,ksoil,Melt,Rf,Sf,Ta,trans,
                    Tsrf,unload,Nsnow,Dsnw,Rgrn,Sice,Sliq,Tsnow,Tsoil,
                    Gsoil,Roff,hs,swe,Wflx):

        # No snow
        Gsoil = Gsrf
        Roff = Rf + drip/dt
        Wflx[:] = 0

        # Existing snowpack
        if (Nsnow > 0):
            # Heat conduction
            for k in range(Nsnow):
                csnow[k] = Sice[k]*hcap_ice + Sliq[k]*hcap_wat
            if (Nsnow == 1):
                Gs[1] = 2 / (Dsnw[1]/ksnow[1] + Dzsoil[1]/ksoil[1])
                dTs[1] = (Gsrf + Gs[1]*(Tsoil[1] - Tsnow[1]))*dt / (csnow[1] + Gs[1]*dt)
            else:
                for k in range(Nsnow-1):
                      Gs[k] = 2 / (Dsnw[k]/ksnow[k] + Dsnw[k+1]/ksnow[k+1])
                a[1] = 0
                b[1] = csnow[1] + Gs[1]*dt
                c[1] = - Gs[1]*dt
                rhs[1] = (Gsrf - Gs[1]*(Tsnow[1] - Tsnow[2]))*dt
                for k in range(1,Nsnow-1):
                    a[k] = c[k-1]
                    b[k] = csnow[k] + (Gs[k-1] + Gs[k])*dt
                    c[k] = - Gs[k]*dt
                    rhs[k] = Gs[k-1]*(Tsnow[k-1] - Tsnow[k])*dt + Gs[k]*(Tsnow[k+1] - Tsnow[k])*dt 

            k = Nsnow
            Gs[k] = 2 / (Dsnw[k]/ksnow[k] + Dzsoil[1]/ksoil[1])
            a[k] = c[k-1]
            b[k] = csnow[k] + (Gs[k-1] + Gs[k])*dt
            c[k] = 0
            rhs[k] = Gs[k-1]*(Tsnow[k-1] - Tsnow[k])*dt + Gs[k]*(Tsoil[1] - Tsnow[k])*dt
            self.tridiag(Nsnow,Nsmax,a,b,c,rhs,dTs)
            
            for k in range(Nsnow):
                Tsnow[k] = Tsnow[k] + dTs[k]
            k = Nsnow
            Gsoil = Gs[k]*(Tsnow[k] - Tsoil[1])
        

            # Convert melting ice to liquid water
            dSice = Melt*dt
            for k in range(Nsnow):
                coldcont = csnow[k]*(Tm - Tsnow[k])
                if (coldcont < 0):
                    dSice = dSice - coldcont/Lf
                    Tsnow[k] = Tm
                if (dSice > 0):
                    if (dSice > Sice[k]):  # Layer melts completely
                        dSice = dSice - Sice[k]
                        Dsnw[k] = 0
                        Sliq[k] = Sliq[k] + Sice[k]
                        Sice[k] = 0
                    else:                       # Layer melts partially
                        Dsnw[k] = (1 - dSice/Sice[k])*Dsnw[k]
                        Sice[k] = Sice[k] - dSice
                        Sliq[k] = Sliq[k] + dSice
                        dSice = 0
                    
            # Remove snow by sublimation 
            dSice = Esrf*dt
            if (dSice > 0):
                for k in range(Nsnow):
                    if (dSice > Sice[k]):  # Layer sublimates completely
                        dSice = dSice - Sice[k]
                        Dsnw[k] = 0
                        Sice[k] = 0
                    else:                       # Layer sublimates partially
                        Dsnw[k] = (1 - dSice/Sice[k])*Dsnw[k]
                        Sice[k] = Sice[k] - dSice
                        dSice = 0


            # Remove wind-trasported snow 
            dSice = trans*dt
            if (dSice > 0):
                for k in range(Nsnow):
                    if (dSice > Sice[k]):  # Layer completely removed
                        dSice = dSice - Sice[k]
                        Dsnw[k] = 0
                        Sice[k] = 0
                    else:                       # Layer partially removed
                        Dsnw[k] = (1 - dSice/Sice[k])*Dsnw[k]
                        Sice[k] = Sice[k] - dSice
                        dSice = 0


            if DENSTY == 0:
                # Fixed snow density
                for k in range(Nsnow):
                    if (Dsnw[k] > epsilon(Dsnw)):
                        Dsnw[k] = (Sice[k] + Sliq[k]) / rfix
            if DENSTY == 1:
                # Snow compaction with age
                for k in range(Nsnow):
                    if Dsnw[k] > np.finfo(float).eps: # epsillon different in FSM
                        rhos = (Sice[k] + Sliq[k]) / Dsnw[k]
                        if Tsnow[k] >= Tm:
                            if rhos < rmlt:
                                rhos = rmlt + (rhos - rmlt) * np.exp(-dt / trho)
                        else:
                            if rhos < rcld:
                                rhos = rcld + (rhos - rcld) * np.exp(-dt / trho)
                        Dsnw[k] = (Sice[k] + Sliq[k]) / rhos
                        
            if DENSTY == 2:
                # Snow compaction by overburden
                mass = 0
                for k in range(Nsnow):
                    mass = mass + 0.5*(Sice[k] + Sliq[k]) 
                    if (Dsnw[k] > np.finfo(float).eps):
                        rhos = (Sice[k] + Sliq[k]) / Dsnw[k]
                        rhos = rhos + (rhos*g*mass*dt/(eta0*exp(-(Tsnow[k] - Tm)/12.4 + rhos/55.6)) + dt*rhos*snda*exp((Tsnow[k] - Tm)/23.8 - max(rhos - 150, 0.)/21.7))
                        Dsnw[k] = (Sice[k] + Sliq[k]) / rhos
                    mass = mass + 0.5*(Sice[k] + Sliq[k])


            # Snow grain growth
            for k in range(Nsnow):
                ggr = 2e-13
                if (Tsnow[k] < Tm):
                    if (Rgrn[k] < 1.50e-4):
                        ggr = 2e-14
                    else:
                        ggr = 7.3e-8*exp(-4600/Tsnow[k])
                Rgrn[k] = Rgrn[k] + dt*ggr/Rgrn[k]

        # Existing snowpack

        # Add snowfall and frost to layer 1 with fresh snow density and grain size
        Esnow = 0
        if (Esrf < 0 & Tsrf < Tm):
            Esnow = Esrf
        dSice = (Sf - Esnow)*dt
        Dsnw[1] = Dsnw[1] + dSice / rhof
        if (Sice[1] + dSice > np.finfo(float).eps):
            Rgrn[1] = (Sice[1]*Rgrn[1] + dSice*rgr0) / (Sice[1] + dSice)
        Sice[1] = Sice[1] + dSice
    
        # Add canopy unloading to layer 1 with bulk snow density and grain size
        rhos = rhof
        swe = sum(Sice[:]) + sum(Sliq[:])
        hs = sum(Dsnw[:])
        if (hs > np.finfo(float).eps):
            rhos = swe / hs
        Dsnw[1] = Dsnw[1] + unload / rhos
        if (Sice[1] + unload > epsilon(Sice)):
            Rgrn[1] = (Sice[1]*Rgrn[1] + unload*rgr0) / (Sice[1] + unload)
        Sice[1] = Sice[1] + unload

        # Add wind-blown snow to layer 1 with wind-packed density and fresh grain size
        dSice = - trans*dt
        if (dSice > 0):
            Dsnw[1] = Dsnw[1] + dSice / rhow
            Rgrn[1] = (Sice[1]*Rgrn[1] + dSice*rgr0) / (Sice[1] + dSice)
            Sice[1] = Sice[1] + dSice

        # New snowpack
        if (Nsnow == 0 & Sice[1] > 0):
            Nsnow = 1
            Rgrn[1] = rgr0
            Tsnow[1] = min(Ta, Tm)


        # Store state of old layers
        D[:] = Dsnw[:]
        R[:] = Rgrn[:]
        S[:] = Sice[:]
        W[:] = Sliq[:]
        for k in range(Nsnow):
            csnow[k] = Sice[k]*hcap_ice + Sliq[k]*hcap_wat
            E[k] = csnow[k]*(Tsnow[k] - Tm)
        Nold = Nsnow
        hs = sum(Dsnw[:])

        # Initialise new layers
        Dsnw[:] = 0
        Rgrn[:] = 0
        Sice[:] = 0
        Sliq[:] = 0
        Tsnow[:] = Tm
        U[:] = 0
        Nsnow = 0

        if (hs > 0):  # Existing or new snowpack
            # Re-assign and count snow layers
            dnew = hs
            Dsnw[1] = dnew
            k = 1
            if (Dsnw[1] > Dzsnow[1]):
                for k in range(Nsmax):
                    Dsnw[k] = Dzsnow[k]
                    dnew = dnew - Dzsnow[k]
                    if (dnew <= Dzsnow[k] | k == Nsmax):
                        Dsnw[k] = Dsnw[k] + dnew
                        break
            Nsnow = k

            # Fill new layers from the top downwards
            knew = 1
            dnew = Dsnw[1]
            for kold in range(Nold): # NOTE HERE WAS 'DO'!
                if (D[kold] < dnew):
                    # All snow from old layer partially fills new layer
                    Rgrn[knew] = Rgrn[knew] + S[kold]*R[kold]
                    Sice[knew] = Sice[knew] + S[kold]
                    Sliq[knew] = Sliq[knew] + W[kold]
                    U[knew] = U[knew] + E[kold]
                    dnew = dnew - D[kold]
                    break
                else:
                    # Some snow from old layer fills new layer
                    wt = dnew / D[kold]
                    Rgrn[knew] = Rgrn[knew] + wt*S[kold]*R[kold]
                    Sice[knew] = Sice[knew] + wt*S[kold]
                    Sliq[knew] = Sliq[knew] + wt*W[kold]
                    U[knew] = U[knew] + wt*E[kold]
                    D[kold] = (1 - wt)*D[kold]
                    E[kold] = (1 - wt)*E[kold]
                    S[kold] = (1 - wt)*S[kold]
                    W[kold] = (1 - wt)*W[kold]
                    knew = knew + 1
                    if (knew > Nsnow):
                        break
                    dnew = Dsnw[knew]

        # Diagnose snow layer temperatures
        for k in range(Nsnow):
            csnow[k] = Sice[k]*hcap_ice + Sliq[k]*hcap_wat
            Tsnow[k] = Tm + U[k] / csnow[k]
            Rgrn[k] = Rgrn[k] / Sice[k]

        # Drain, retain or freeze snow in layers
        if HYDROL == 0:
            # Free-draining snow, no retention or freezing 
            Wflx[1] = Roff
            for k in range(Nsnow):
                Roff = Roff + Sliq[k] / dt
                Sliq[k] = 0
                if (k < Nsnow):
                    Wflx[k+1] = Roff

        if HYDROL == 1:
            # Bucket storage 
            if (maxval(Sliq)>0 | Rf>0):
                for k in range(Nsnow):
                    phi[k] = 1 - Sice[k]/(rho_ice*Dsnw[k])
                    SliqMax = rho_wat*Dsnw[k]*phi[k]*Wirr
                    Sliq[k] = Sliq[k] + Roff*dt
                    Wflx[k] = Roff
                    Roff = 0
                if (Sliq[k] > SliqMax):       # Liquid capacity exceeded
                    Roff = (Sliq[k] - SliqMax)/dt   # so drainage to next layer
                    Sliq[k] = SliqMax
                csnow[k] = Sice[k]*hcap_ice + Sliq[k]*hcap_wat
                coldcont = csnow[k]*(Tm - Tsnow[k])
                if (coldcont > 0):            # Liquid can freeze
                    dSice = min(Sliq[k], coldcont/Lf)
                    Sliq[k] = Sliq[k] - dSice
                    Sice[k] = Sice[k] + dSice
                    Tsnow[k] = Tsnow[k] + Lf*dSice/csnow[k]

        swe = sum(Sice[:]) + sum(Sliq[:])
        # Existing or new snowpack