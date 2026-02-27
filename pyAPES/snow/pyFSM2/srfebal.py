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
                    'ZOFFST': (int): 0,1,2
                    'EXCHNG': (int): 0,1,2
                'params' (dict):
                    'z0sn' (float): # Snow roughness length (m)
                'initial_conditions' (dict):
                    'Tsrf' (float): Initial surface temperature (K)

        Returns:
            self (object)
        """        

        self.ZOFFST = properties['physics_options']['ZOFFST']
        self.EXCHNG = properties['physics_options']['EXCHNG']
        self.z0sn = properties['params']['z0sn']
        self.Tsrf = properties['initial_conditions']['Tsrf']

        # temporary storage of iteration results
        self.iteration_state = None  

    def update(self):
        """ 
        Updates srfebal state.
        """
        self.Tsrf = self.iteration_state['Tsrf']


    def run(self, dt: float, forcing: Dict) -> Tuple:
        """
        Calculates one timestep

        Args:
            dt (float): timestep [s]
            forcing' (dict):
                'Ds1' (float): Surface layer thickness (m)
                'fsnow' (float): Ground snowcover fraction
                'gs1' (float): Surface moisture conductance (m/s)
                'ks1' (float): Surface layer thermal conductivity (W/m/K)
                'LW' (float): Incoming longwave radiation (W/m2)
                'Ps' (float): Surface pressure (Pa)
                'RH' (float): Relative humidity (%)
                'SWsrf' (float): SW absorbed by snow/ground surface (W/m^2)
                'Ta' (float): Air temperature (K)
                'Ts1' (float): Surface layer temperature (K)
                'Ua' (float): Wind speed (m/s)
                'Sice' (np.ndarray): Ice content of snow layers (kg/m^2)
                'z0sf' (float): Surface roughness length [m]
        Returns
            (tuple):
            fluxes (dict):
                'Esrf' (float): Moisture flux from the surface (kg/m^2/s)
                'Gsrf' (float): Heat flux into snow/ground surface (W/m^2)
                'H' (float): Sensible heat flux to the atmosphere (W/m^2)
                'LE' (float): Latent heat flux to the atmosphere (W/m^2)
                'LWout' (float): Outgoing LW radiation (W/m^2)
                'Melt' (float): Surface melt rate (kg/m^2/s)
                'subl' (float): Sublimation rate (kg/m^2/s)
            states (dict):
                'Tsrf' (float): Surface temperature (K)
                'rL' (float): Monin-Obukhov length (m)
        """
        # read forcings
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
        Sice = forcing['Sice']
        zU = forcing['reference_height']
        zT = forcing['reference_height']
        Tsrf = forcing['Tsrf']
        z0sf = forcing['z0sf']
        
        if self.ZOFFST == 0:
            # Heights specified above ground
            self.zU1 = zU
            self.zT1 = zT

        if self.ZOFFST == 1:
            # Heights specified above canopy top
            self.zU1 = zU + self.vegh
            self.zT1 = zT + self.vegh

        # Convert relative to specific humidity
        Tc = Ta - T_MELT
        es = SATURATION_VAPOR_PRESSURE_MELT * np.exp(17.5043*Tc/(241.3 + Tc))
        Qa = (RH/100.)*R_RATIO*es/Ps

        # Roughness lengths
        self.z0g = (self.z0sn**fsnow) * (z0sf**(1 - fsnow))
        self.z0h = 0.1 * self.z0g
        
        # Saturation humidity and air density
        Qsrf = self.qsat(Ps=Ps, T=Tsrf)
        Lsrf = np.array(LATENT_HEAT_SUBLIMATION)
        if (Tsrf > T_MELT):
            Lsrf = LATENT_HEAT_VAPORISATION
        Dsrf = Lsrf * Qsrf / (GAS_CONSTANT/MOLAR_MASS_H2O * Tsrf**2)
        rho = Ps / (GAS_CONSTANT/MOLAR_MASS_AIR * Ta)

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
            Esrf = rho * wsrf * ga * (Qsrf - Qa)
            Gsrf = 2 * ks1 * (Tsrf - Ts1) / Ds1
            Hsrf = SPECIFIC_HEAT_AIR/MOLAR_MASS_AIR * rho * ga * (Tsrf - Ta)
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

            if (ne > 4) and (abs(ebal) < 0.01):
                break
    
        # Sublimation limited by available snow
        subl = 0
        Ssub = np.sum(Sice[:]) - Melt * dt
        if (Ssub > 0.) or (Tsrf < T_MELT):
            Esrf = np.minimum(Esrf, Ssub / dt)
            subl = Esrf
                
        # Fluxes to the atmosphere
        E = float(Esrf)
        H = float(Hsrf)
        LE = float(Lsrf * Esrf)

        # store iteration state
        self.iteration_state =  {'Tsrf': Tsrf
                                 }

        fluxes = {'Esrf': Esrf,
                  'Gsrf': Gsrf,
                  'H': H,
                  'LE': LE,
                  'Rsrf': Rsrf,
                  'LWout': LWout,
                  'Melt': Melt,
                  'subl': subl,
                  'ustar': ustar,
                  'ga': ga,
                  'ebal': ebal
                 }

        states = {'Tsrf': Tsrf,
                  'rL': rL}
                    
        return fluxes, states



    def qsat(self, Ps, T):
        """
        Computes the saturation specific humidity.
        
        Args:
            'Ps' (float): Air pressure (Pa)
            'T' (float): Temperature (K)
            
        Returns:
            'Qs' (float): Saturation specific humidity
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

        Args:
            'z' (float): Height above surface (m)
            'rL' (float): Monin-Obukhov length (m)

        Returns:
             stability function for momentum
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

        Args:
            'z' (float): Height above surface (m)
            'rL' (float): Monin-Obukhov length (m)

        Returns:
             stability function for heat
        """
        zeta = float(np.clip(z * rL, -2.0, 1.0))

        if zeta > 0.0:
            return -5.0 * zeta  # stable
        else:
            x = (1.0 - 16.0 * zeta) ** 0.25
            return 2.0 * np.log((1.0 + x**2) / 2.0)
