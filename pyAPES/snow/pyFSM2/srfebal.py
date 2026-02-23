"""
.. module: energy balance
    :synopsis: APES-model component
.. moduleauthor:: Jari-Pekka Nousu

*Surface energy balance (based on FSM2)*

"""

import numpy as np
from typing import Dict, List, Tuple
from pyAPES.utils.utilities import tridiag, ludcmp
from pyAPES.utils.constants import SPECIFIC_HEAT_AIR, MOLAR_MASS_AIR, \
                                    LATENT_HEAT_VAPORISATION, LATENT_HEAT_FUSION, LATENT_HEAT_SUBLIMATION, \
                                    T_MELT, STEFAN_BOLTZMANN, VON_KARMAN, \
                                    GAS_CONSTANT, MOLAR_MASS_H2O, \
                                    SATURATION_VAPOR_PRESSURE_MELT, R_RATIO

EPS = np.finfo(float).eps  # machine epsilon

class EnergyBalance:
    def __init__(self,
                 properties: Dict):
        """
        Energy balance module based on FSM2.

        Args:
            properties (dict)
                'physics_options' (dict):
                    'DENSTY': (int): 0,1,2
                    'ZOFFST': (int): 0,1,2
                    'CANMOD': (int): 0,1,2
                    'EXCHNG': (int): 0,1,2
                'params' (dict):
                    'gsnf' (float): # Snow-free vegetation moisture conductance (m/s)
                    'hbas' (float): # Canopy base height (m)
                    'kext' (float): # Vegetation light extinction coefficient
                    'leaf' (float): # Leaf boundary resistance (s/m)^(1/2)
                    'wcan' (float): # Canopy wind decay coefficient
                    'z0sf' (float): # Snow-free surface roughness length (m)
                    'z0sn' (float): # Snow roughness length (m)
                    'kfix' (float): # Fixed thermal conductivity of snow (W/m/K)
                'layers' (dict):
                    'Nsmax' (int): # Maximum number of snow layers
                    'Ncnpy' (int): # Number of canopy layers
                    'fvg1' (float): # Fraction of vegetation in upper canopy layer
                    'zsub' (float): # Subcanopy wind speed diagnostic height (m)
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

        # From Layers
        self.Ncnpy = properties['layers']['Ncnpy'] # Number of canopy layers
        self.Nsmax = properties['layers']['Nsmax'] # Maximum number of snow layers
        self.fvg1 = properties['layers']['fvg1'] # Fraction of vegetation in upper canopy layer
        self.zsub = properties['layers']['zsub'] # Subcanopy wind speed diagnostic height (m)

        # From Parameters
        self.gsnf = properties['params']['gsnf'] # Snow-free vegetation moisture conductance (m/s)
        self.hbas = properties['params']['hbas'] # Canopy base height (m)
        self.kext = properties['params']['kext'] # Vegetation light extinction coefficient
        self.leaf = properties['params']['leaf'] # Leaf boundary resistance (s/m)^(1/2)
        self.wcan = properties['params']['wcan'] # Canopy wind decay coefficient
        self.z0sn = properties['params']['z0sn'] # Snow roughness length (m)
        self.kfix = properties['params']['kfix'] 
        self.rhof = properties['params']['rhof']  # Fresh snow density (kg/m^3)

        self.VAI = properties['params']['VAI']
        self.vegh = properties['params']['vegh']

        self.DENSTY = properties['physics_options']['DENSTY']
        self.ZOFFST = properties['physics_options']['ZOFFST']
        self.CANMOD = properties['physics_options']['CANMOD']
        self.EXCHNG = properties['physics_options']['EXCHNG']
        self.CONDCT = properties['physics_options']['CONDCT']

        self.Tsrf = properties['initial_conditions']['Tsrf']
        self.Eveg = np.zeros(self.Ncnpy) # Moisture flux from vegetation layers (kg/m^2/s)
        self.d = 0 #  Displacement height (m)
        self.fveg = 0 # Vegetation weighting
        self.Ssub = 0 # Mass of snow available for sublimation (kg/m^2)
        self.Hveg = np.zeros(self.Ncnpy) # Sensible heat flux from vegetation (W/m^2)
        self.zh = np.array(self.Ncnpy) # Vegetation layer heights (m)

        # temporary storage of iteration results
        self.iteration_state = None  


    def update(self):
        """ 
        Updates swrad state.
        """
        self.Tsrf = self.iteration_state['Tsrf']
    

    def run(self, dt: float, forcing: Dict) -> Tuple:
        """
        Calculates one timestep and updates surface temperature

        Args:
            dt (float): timestep [s]
            forcing' (dict):
                Ds1: 
                fcans:
                fsnow:
                gs1:
                ks1:
                LW:
                Ps:
                RH:
                SWsrf:
                Sveg:
                SWveg:
                Ta:
                tdif:
                Ts1:
                Ua:
                vegh:
                Sice: 
                Nsnow
                Sice
                Sliq
                Dsnw
        Returns
            (tuple):
            fluxes (dict):
                Esrf:
                Gsrf:
                H:
                LE:
                LWout:
                LWsub:
                Melt:
                subl:
                Usub:
                Eveg:
            states (dict):
                Tsrf:
        """
        Ds1 = forcing['Ds1']
        fsnow = forcing['fsnow']
        gs1 = forcing['gs1']
        ks1 = forcing['ks1']
        LW = forcing['LW']
        Ps = forcing['Ps']
        RH = forcing['RH']
        SWsrf = forcing['SWsrf']
        Ta = forcing['Ta']
        Ts1 = forcing['Ts1']
        Ua = forcing['Ua']
        Sice = forcing['Sice']
        Nsnow = forcing['Nsnow']
        Sice = forcing['Sice']
        Sliq = forcing['Sliq']
        Dsnw = forcing['Dsnw']
        zU = forcing['reference_height']
        zT = forcing['reference_height']
        Tsrf = forcing['Tsrf']
        z0sf = forcing['z0sf']

        n_geom = np.count_nonzero(Dsnw > np.finfo(float).eps)
        if n_geom != Nsnow:
            print("Layer mismatch!")
            print("Dsnw:", Dsnw)
            print("n_geom:", n_geom, "Nsnow:", Nsnow)
        
        if self.ZOFFST == 0:
            # Heights specified above ground
            self.zU1 = zU
            self.zT1 = zT

        if self.ZOFFST == 1:
            # Heights specified above canopy top
            self.zU1 = zU + self.vegh
            self.zT1 = zT + self.vegh

        if self.CANMOD == 1:
            self.zh[0] = self.hbas + 0.5 * (self.vegh - self.hbas)

        if self.CANMOD == 2:
            self.zh[0] = (1 - 0.5 * self.fvg1) * self.vegh
            self.zh[1] = 0.5 * (1 - self.fvg1) * self.vegh 

        # Convert relative to specific humidity
        Tc = Ta - T_MELT
        es = SATURATION_VAPOR_PRESSURE_MELT * np.exp(17.5043*Tc/(241.3 + Tc))
        Qa = (RH/100)*R_RATIO*es/Ps

        # Roughness lengths
        self.fveg = 1 - np.exp(-self.kext * self.VAI)
        self.d = 0.67 * self.fveg * self.vegh
        self.z0g = (self.z0sn**fsnow) * (z0sf**(1 - fsnow))
        self.z0h = 0.1 * self.z0g
        self.z0v = ((0.05 * self.vegh)**self.fveg) * (self.z0g**(1 - self.fveg))

        self.d = 0.67 * self.vegh
        self.z0v = 0.1 * self.vegh
        
        # Saturation humidity and air density
        Qsrf = self.qsat(Ps=Ps, T=Tsrf)
        Lsrf = np.array(LATENT_HEAT_SUBLIMATION)
        if (Tsrf > T_MELT):
            Lsrf = LATENT_HEAT_VAPORISATION
        Dsrf = Lsrf * Qsrf / (GAS_CONSTANT/MOLAR_MASS_H2O * Tsrf**2)
        rho = Ps / (GAS_CONSTANT/MOLAR_MASS_AIR * Ta)

        if (self.VAI == 0.0):  # open
            self.Eveg[:] = 0
            self.Hveg[:] = 0
            ustar = np.maximum(
                VON_KARMAN * Ua / np.log(self.zU1 / self.z0g),
                0.001)            # ustar should not be 0
            ga = VON_KARMAN * ustar / np.log(self.zT1 / self.z0h)

            for ne in range(40):  # Iterating for stability adjustments
                if self.EXCHNG == 0:
                    rL = 0.
                elif self.EXCHNG == 1:
                    if ne < 8:
                        B = ga * (Tsrf - Ta)
                        rL = -VON_KARMAN * B / (Ta * ustar**3)
                        rL = np.clip(rL, -2., 2.)

                # Update ustar and ga in every iteration
                ustar = np.maximum(
                    VON_KARMAN * Ua / (np.log(self.zU1 / self.z0g) - self.psim(self.zU1, rL) + self.psim(self.z0g, rL)),
                    0.001) # ustar should not be 0               
                ga = VON_KARMAN * ustar / (np.log(self.zT1 / self.z0h) - self.psih(self.zT1, rL) + self.psih(self.z0h, rL))
                           
                if not np.isfinite(ga):  # Ensure ga remains valid
                    raise ValueError(f"ga not finite: {ga}")
                if np.iscomplex(ga):
                    raise ValueError(f"ga became complex: {ga}")
                
                # Surface water availability
                if (Qa > Qsrf): # specific humidity > saturation specific humidity at surface temperature 
                    wsrf = 1.
                else:
                    wsrf = fsnow + (1 - fsnow) * gs1 / (gs1 + ga)

                # Explicit fluxes
                #ga = np.maximum(ga, 0.01)
                Esrf = rho * wsrf * ga * (Qsrf - Qa)
                self.Eveg[:] = 0.
                Gsrf = 2 * ks1 * (Tsrf - Ts1) / Ds1
                Hsrf = SPECIFIC_HEAT_AIR/MOLAR_MASS_AIR * rho * ga * (Tsrf - Ta)
                self.Hveg[:] = 0.
                Melt = 0.
                Rsrf = SWsrf + LW - STEFAN_BOLTZMANN * Tsrf**4

                # Surface energy balance increments without melt
                dTs = (Rsrf - Gsrf - Hsrf - Lsrf * Esrf) / \
                        (4 * STEFAN_BOLTZMANN * Tsrf**3 + 2 * ks1 / Ds1 + rho * (SPECIFIC_HEAT_AIR/MOLAR_MASS_AIR + Lsrf * Dsrf * wsrf) * ga)
                dEs = rho * wsrf * ga * Dsrf * dTs
                dGs = 2 * ks1 * dTs / Ds1 
                dHs = SPECIFIC_HEAT_AIR/MOLAR_MASS_AIR * rho * ga * dTs

                # Surface melting
                if (Tsrf + dTs > T_MELT) and (Sice[0] > 0):
                    Melt = np.sum(Sice) / dt
                    dTs = (Rsrf - Gsrf - Hsrf - Lsrf * Esrf - LATENT_HEAT_FUSION * Melt) \
                            / (4 * STEFAN_BOLTZMANN * Tsrf**3 + 2 * ks1/Ds1 \
                               + rho*(SPECIFIC_HEAT_AIR/MOLAR_MASS_AIR + LATENT_HEAT_SUBLIMATION * Dsrf * wsrf) * ga)
                    dEs = rho * wsrf * ga * Dsrf * dTs
                    dGs = 2 * ks1 * dTs/Ds1
                    dHs = SPECIFIC_HEAT_AIR/MOLAR_MASS_AIR * rho * ga * dTs
                    if (Tsrf + dTs < T_MELT):
                        Qsrf = self.qsat(Ps=Ps,T=T_MELT)
                        Esrf = rho * wsrf * ga * (Qsrf - Qa)
                        Gsrf = 2 * ks1 * (T_MELT - Ts1)/Ds1
                        Hsrf = SPECIFIC_HEAT_AIR/MOLAR_MASS_AIR * rho * ga * (T_MELT - Ta)
                        Rsrf = SWsrf + LW - STEFAN_BOLTZMANN * T_MELT**4 
                        Melt = (Rsrf - Gsrf - Hsrf - Lsrf * Esrf) / LATENT_HEAT_FUSION
                        Melt = np.maximum(Melt, 0.)
                        dEs = 0.
                        dGs = 0.
                        dHs = 0.
                        dTs = T_MELT - Tsrf
                
                # Update surface temperature and fluxes
                Esrf = Esrf + dEs
                Gsrf = Gsrf + dGs
                Hsrf = Hsrf + dHs
                Tsrf = Tsrf + dTs
                if not np.isfinite(Tsrf):
                    raise ValueError("Tsrf non-finite after update")
                
                # Diagnostics
                ebal = SWsrf + LW - STEFAN_BOLTZMANN * Tsrf**4 - Gsrf - Hsrf - Lsrf * Esrf - LATENT_HEAT_FUSION * Melt
                LWout = STEFAN_BOLTZMANN * Tsrf**4
                LWsub = LW                
                Usub = (ustar / VON_KARMAN) * (np.log(self.zsub/self.z0g) - self.psim(self.zsub,rL) + self.psim(self.z0g,rL))

                if (ne > 4) and (abs(ebal) < 0.01):
                    break
    
        # Sublimation limited by available snow
        subl = 0
        self.Ssub = np.sum(Sice[:]) - Melt * dt
        if (self.Ssub > 0) or (Tsrf < T_MELT):
            Esrf = np.minimum(Esrf, self.Ssub / dt)
            subl = Esrf
                
        # Fluxes to the atmosphere
        E = float(Esrf + np.sum(self.Eveg[:]))
        H = float(Hsrf + np.sum(self.Hveg[:]))
        LE = float(Lsrf * Esrf) #+ sum(Lcan[:]*self.Eveg[:])

        # store iteration state
        self.iteration_state =  {'Tsrf': Tsrf
                                 }

        fluxes = {'Esrf': Esrf,
                  'Gsrf': Gsrf,
                  'H': H,
                  'LE': LE,
                  'Rsrf': Rsrf,
                  'LWout': LWout,
                  'LWsub': LWsub,
                  'Melt': Melt,
                  'subl': subl,
                  'Usub': Usub,
                  'ustar': ustar,
                  'ga': ga,
                  'ebal': ebal
                 }

        states = {'Tsrf': Tsrf,
                  'rL': rL}
                    
        return fluxes, states



    def qsat(self, Ps, T):
        """
        Computes the saturation specific humidity, corresponding exactly to the Fortran version.
        
        Args:
            P (float): Air pressure (Pa)
            T (float): Temperature (K)
            
        Returns:
            Qs (float): Saturation specific humidity
        """
        Tc = T - T_MELT  # Convert to Celsius

        if Tc > 0:
            es = SATURATION_VAPOR_PRESSURE_MELT * np.exp(17.5043 * Tc / (241.3 + Tc))
        else:
            es = SATURATION_VAPOR_PRESSURE_MELT * np.exp(22.4422 * Tc / (272.186 + Tc))

        Qsrf = R_RATIO * es / Ps  # Compute specific humidity
        
        return Qsrf



    def psim(self, z, rL):
        """
        Stability function for momentum.
        """
        zeta = float(np.clip(z * rL, -2.0, 1.0))

        if zeta > 0.0:
            return -5.0 * zeta  # stable
        else:
            x = (1.0 - 16.0 * zeta) ** 0.25
            return (2.0 * np.log((1.0 + x) / 2.0) +
                    np.log((1.0 + x**2) / 2.0) -
                    2.0 * np.arctan(x) + np.pi / 2.0)

    
    def psih(self, z, rL):
        """
        Stability function for heat.
        """
        zeta = float(np.clip(z * rL, -2.0, 1.0))

        if zeta > 0.0:
            return -5.0 * zeta  # stable
        else:
            x = (1.0 - 16.0 * zeta) ** 0.25
            return 2.0 * np.log((1.0 + x**2) / 2.0)
