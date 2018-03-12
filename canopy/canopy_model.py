# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 11:01:50 2017

@author: slauniai

******************************************************************************
CanopyModel:

Gridded canopy and snow hydrology model for SpatHy -integration
Uses simple schemes for computing water flows and storages within vegetation
canopy and snowpack at daily or sub-daily timesteps.

(C) Samuli Launiainen, 2016-2017
last edit: 1.11.2017: Added CO2-response to dry_canopy_et
******************************************************************************

"""
import numpy as np
eps = np.finfo(float).eps
from matplotlib import pyplot as plt
from photo import leaf_interface

#: [kg mol-1, molar mass of H2O
MH2O = 18.02e-3
#: [J/mol], latent heat of vaporization at 20 deg C
L_MOLAR = 44100.0 

class CanopyModel():
    def __init__(self, cpara):
        """
        initializes CanopyModel -object

        Args:
            cpara (dict): see canopy_parameters.py
        Returns:
            self (object):
                parameters/state variables:
                    Switch_MLM (boolean): control for multilayer computation
                    location (dict): 
                        'lat'(float): latitude
                        'lon'(float): longitude
                    LAI (float): total leaf area index [m2 m-2]
                    lad (float): total leaf area density [m2 m-3]
                    hc (float): canopy heigth [m]
                    cf (float): canopy closure [-]
                    gsref (float): LAI-weigthed average gsref of canopy [m s-1]
                    pheno_state (float): LAI-weigthed average phenological state of canopy (0..1)
                grid (only for multilayer computation):
                    z, dz, Nlayers (floats)
                objects:
                    Ptypes (.Pheno_Model, .LAI_model), Forestfloor (.Evap_Model, ...)
                    Radi_Model, Aero_Model, Interc_model, Canopy_Tr, Snow_Model
        """

        from radiation import Radiation
        from micromet import Aerodynamics
        from interception import Interception
        from evapotranspiration import Canopy_Transpiration
        from snow import Snowpack

        # --- control switches ---
        self.Switch_MLM = cpara['ctr']['multilayer_model']['ON']

        # --- site location ---
        self.location = cpara['loc']

        # --- grid ---
        if self.Switch_MLM:
            self.Nlayers = cpara['grid']['Nlayers']  # number of layers
            self.z = np.linspace(0, cpara['grid']['zmax'], self.Nlayers)  # grid [m] above ground
            self.dz = self.z[1] - self.z[0]  # gridsize [m]
            self.Switch_Eflow = cpara['ctr']['multilayer_model']['Eflow']
            self.Switch_WMA = cpara['ctr']['multilayer_model']['WMA']
        else:
            self.z = np.array([-1])

        # --- Plant types (with phenoligical models) ---
        ptypes = []
        stand_lai, stand_lad = 0.0, 0.0
        # para = [pinep, shrubp]
        for k in range(len(cpara['plant_types'])):
            p = cpara['plant_types'][k]
            for n in range(len(p['LAImax'])):
                pp = p.copy()
                pp['LAImax'] = p['LAImax'][n]
                if self.Switch_MLM:
                    pp['lad'] = p['lad'][:, n]
                else:
                    pp['lad'] = 0.0
                ptypes.append(PlantType(self.z, pp,
                        Switch_pheno=cpara['ctr']['pheno_cylcle'],
                        Switch_lai=cpara['ctr']['seasonal_LAI'],
                        MLM_ctr=cpara['ctr']['multilayer_model']))
                stand_lai += pp['LAImax']
                stand_lad += pp['lad']
        self.Ptypes = ptypes
        del p, pp, k, ptypes

        # --- stand characteristics ---
        self.LAI = stand_lai  # canopy total 1-sided leaf area index [m2m-2]
        self.lad = stand_lad  # canopy total 1-sided leaf area density [m2m-3]
        if self.Switch_MLM:
            f = np.where(self.lad > 0)[0][-1]
            self.hc = self.z[f].copy()  # canopy height [m]
        else:
            self.hc = cpara['canopy_para']['hc']  # canopy height [m]

            # --- initialize canopy evapotranspiration computation ---
            self.Canopy_Tr = Canopy_Transpiration(cpara['phys_para'])

        # --- initialize radiation model ---
        self.Radi_Model = Radiation(cpara['radi'], self.Switch_MLM)

        # --- initialize canopy flow model ---
        self.Aero_Model = Aerodynamics(self.z, self.lad, self.hc, cpara['aero'], self.Switch_MLM)

        # --- initialize interception model---
        self.Interc_Model = Interception(cpara['interc_snow'], self.LAI)

        # --- initialize snow model ---
        self.Snow_Model = Snowpack(cpara['interc_snow'], self.Interc_Model.cf)

        # --- forest floor (models for evaporation...)---
        self.ForestFloor = ForestFloor(cpara['ffloor'])  ### --- soilrp to parameters?

    def _run_daily(self, doy, Ta, PsiL=0.0):
        """
        Computatations done at daily timestep. Updates 
        LAI and phenology
        Args:
            doy: day of year [days]
            Ta: mean daily air temperature [degC]
            PsiL: leaf water potential [MPa]
        Returns:
            None
        """

        """ update physiology and leaf area of planttypes & canopy"""
        stand_lai, stand_lad, gsref, pheno_state = 0.0, 0.0, 0.0, 0.0
        for pt in self.Ptypes:
            pt._update_daily(doy, Ta, PsiL=PsiL)  # updates pt properties
            stand_lad += pt.lad
            stand_lai += pt.LAI
            gsref += pt.LAI * pt.gsref
            pheno_state += pt.LAI * pt.pheno_state
        self.lad = stand_lad
        self.LAI = stand_lai
        self.gsref = gsref / stand_lai
        self.pheno_state = pheno_state / stand_lai

        """ normalized flow statistics in canopy with new lad """
        if self.Switch_MLM and self.Switch_Eflow and self.Ptypes[0].Switch_lai:
            self.Aero_Model.normalized_flow_stats(self.z, self.lad, self.hc)

    def _run_timestep(self, dt, forcing, Rew=1.0, beta=1.0):
        """
        Runs CanopyModel instance for one timestep
        Args:
            dt: timestep [s]
            forcing (dataframe): meteorological forcing data
                'Prec': precipitation rate [m s-1]
                'Tair': air temperature [degC]
                'Rg': global radiation [W m-2]
                'vpd': vapor pressure deficit [kPa]
                'Par': fotosynthetically active radiation [W m-2]
                'U': wind speed [m s-1]
                'CO2': atm. CO2 mixing ratio [ppm]
                'P': pressure [Pa]
            Rew - relative extractable water [-], scalar or matrix
            beta - term for soil evaporation resistance (Wliq/FC) [-]
        Returns:
            fluxes (dict):
            state (dict):
        """

        T = np.array([forcing['Tair']])
        U = np.array([forcing['U']])
        VPD = np.array([forcing['vpd']])  # in multilayer model from H2O somehow?
        CO2 = np.array([forcing['CO2']])
        P = forcing['P']

        """ --- radiation --- """
        if 'Rnet' in forcing:
            Rn = forcing['Rnet']
        else:
            # estimate net radiation based on global radiation [W/m2]
            # Launiainen et al. 2016 GCB, fit to Fig 2a
            Rn = np.maximum(2.57 * self.LAI / (2.57 * self.LAI + 0.57) - 0.2, 0.55) * forcing['Rg']
        # Rnet available at canopy and ground [W/m2]
        Rnet_c, Rnet_gr = self.Radi_Model.layerwise_Rnet(
                LAI=self.LAI,
                Rnet=Rn)

        """ --- aerodynamics --- """
        if self.Switch_MLM is False:
            # aerodynamic resistance and wind speed at ground and canopy
            ra, U = self.Aero_Model._run(
                    LAI=self.LAI,
                    hc=self.hc,
                    Uo=U)
        else:
            # multilayer canopy flow statistics
            if self.Switch_Eflow is False:
                # recompute normalized flow statistics in canopy with current meteo
                self.Aero_Model.normalized_flow_stats(
                        z=self.z,
                        lad=self.lad,
                        hc=self.hc,
                        Utop=forcing['U']/(forcing['Ustar'] + eps))
            # update U and initialize canopy flow statistics
            U, T, H2O, CO2 = self.Aero_Model.update_state(
                    ustaro=forcing['Ustar'], 
                    To=forcing['Tair'], 
                    H2Oo=forcing['H2O'], 
                    CO2o=forcing['CO2'])
            # aerodynamic resistance
            ra, _ = self.Aero_Model._run(   ## tähän joku Uta ja ladia hyödyntävä tapa?
                    LAI=self.LAI,
                    hc=self.hc,
                    Uo=U[-1])

        """ --- interception and interception storage evaporation --- """
        Trfall_rain, Trfall_snow, Interc, Evap, MBE_interc = self.Interc_Model._run(
                dt=dt,
                LAI=self.LAI,
                T=T[-1],
                Prec=forcing['Prec'],
                AE=Rnet_c,
                VPD=VPD,
                Ra=ra[-1],
                U=U[-1])

        """ --- snowpack ---"""
        PotInf, MBE_snow = self.Snow_Model._run(
                dt=dt,
                T=T[0],
                Trfall_rain=Trfall_rain,
                Trfall_snow=Trfall_snow)

        """ --- compute Par profiles with canopy --- """
        if self.Switch_MLM:
            Q_sl1, Q_sh1, f_sl, Par_gr = self.Radi_Model.PAR_profiles(
                    LAIz=self.lad * self.dz,
                    zen=forcing['Zen'],
                    dirPar=forcing['dirPar'],
                    diffPar=forcing['diffPar'])

        # multi-layer computations
        max_err = 0.01  # maximum relative error
        max_iter = 10  # maximum iterations
        gam = 0.5  # weight for iterations
        err_h2o, err_co2 = 999., 999.

        if self.Switch_MLM:
            Switch_WMA = self.Switch_WMA
        else:
            Switch_WMA = True

        iter_no = 1

        while (err_h2o > max_err or err_co2 > max_err) and iter_no < max_iter:
            if self.Switch_MLM:
                # --- h2o and co2 sink/source terms
                qsource = np.zeros(self.Nlayers)
                csource = np.zeros(self.Nlayers)
                # dark respiration
                Rstand = 0.0
                # PlantType results, create empty list to append
                pt_stats = []
                """ --- compute leaf gas-exchange at each layer and for each PlantType """
                # here assume Tleaf = T
                for pt in self.Ptypes:
                    # --- sunlit leaves
                    pt_stats_i, dqsource, dcsource, dRstand = pt.leaf_gasexchange(
                            f_sl, H2O, CO2, T, U, P, Q_sl1, Q_sh1,
                            SWabs_sl=0.0, SWabs_sh=0.0, LWl=np.zeros(self.Nlayers)) # only used if Ebal=True
                    # --- update ---
                    # h2o and co2 sink/source terms
                    qsource += dqsource  # mol m-3 s-1
                    csource += dcsource  #umol m-3 s-1
                    # dark respiration umol m-2 s-1
                    Rstand +=  dRstand
                    # PlantType results
                    pt_stats.append(pt_stats_i)
            else:
                "" "--- canopy transpiration --- """
                Tr = self.Canopy_Tr._run(  # DRY ??
                        dt=dt,
                        LAI=self.LAI,
                        gsref=self.gsref,
                        T=T[-1],
                        AE=Rnet_c,
                        Qp=forcing['Par'],
                        VPD=VPD,
                        Ra=ra[-1],
                        CO2=CO2[-1],
                        fPheno=self.pheno_state,
                        Rew=Rew)

            """ --- forest floor --- """
            if self.Switch_MLM:
                E_gr, LE_gr, An_gr, R_gr, trfall_gr = 0.0, 0.0, 0.0, 0.0, 0.0
                # CO2 flux at forest floor
                Fc_gr = An_gr + R_gr
            else:
                Efloor = self.ForestFloor._run(
                        dt=dt,
                        T=T[0],
                        AE=Rnet_gr,
                        VPD=VPD,
                        Ra=ra[0],
                        beta=beta,
                        SWE=self.Snow_Model.swe)

            """  --- solve scalar profiles (H2O, CO2) """
            # we need to add here T-profile as well
            if Switch_WMA is False:
                H2O, CO2, err_h2o, err_co2 = self.Aero_Model.scalar_profiles(
                        gam, H2O, CO2, T, P,
                        source={'H2O': qsource, 'CO2': csource},
                        lbc={'H2O': E_gr, 'CO2': Fc_gr})
                iter_no += 1
                if iter_no == max_iter:  # if no convergence, re-compute with WMA -assumption
                    Switch_WMA = True
                    iter_no = 1
            else:
                err_h2o, err_co2 = 0.0, 0.0

        if self.Switch_MLM:
            # ecosystem fluxes
            Fc = np.cumsum(csource)*self.dz + Fc_gr  # umolm-2s-1
            LE = (np.cumsum(qsource)*self.dz + E_gr) * L_MOLAR  # Wm-2

            # net ecosystem exchange umolm-2s-1
            NEE = Fc[-1]
            # ecosystem respiration umolm-2s-1
            Reco = Rstand + R_gr
            # ecosystem GPP umolm-2s-1
            GPP = - NEE + Reco
            # stand transpiration [m/dt]
            Tr = (LE[-1] - LE_gr) / L_MOLAR * MH2O * dt * 1e-3
            # evaporation from moss layer [m/dt]
            Efloor = E_gr * MH2O * dt * 1e-3   # on jo HIAHTUNUT? Ei haihduteta maasta..?

        # return state and fluxes in dictionary
        state = {'SWE': self.Snow_Model.swe,
                 'LAI': self.LAI,
                 'Phenof': self.pheno_state
                 }
        fluxes = {'PotInf': PotInf,
                  'Trfall': Trfall_rain + Trfall_snow,
                  'Interc': Interc,
                  'CanEvap': Evap,
                  'Transp': Tr,
                  'Efloor': Efloor,
                  'MBE_interc': MBE_interc,
                  'MBE_snow': MBE_snow
                  }

        return fluxes, state

class PlantType():
    """
    PlantType -class.
    Contains plant-specific properties, state variables and phenology functions
    """

    def __init__(self, z, p, Switch_pheno, Switch_lai,  MLM_ctr, Switch_WaterStress=False):
        """
        Creates PlantType
        Args:
            z - grid, evenly spaced, np.array
            p - parameters (dict)
            Switch_x - controls
        Returns:
            PlantType instance
        """

        from phenology import Photo_cycle, LAI_cycle

        self.Switch_pheno = Switch_pheno  # include phenology
        self.Switch_lai = Switch_lai  # seasonal LAI
        self.Switch_WaterStress = Switch_WaterStress  # water stress affects stomata

        # phenology model
        if self.Switch_pheno:
            self.Pheno_Model = Photo_cycle(p['phenop'])  # phenology model instance
            self.pheno_state = self.Pheno_Model.f  # phenology state [0...1]
        else:
            self.pheno_state = 1.0

        # dynamic LAI model
        if self.Switch_lai:
            # seasonality of leaf area
            self.LAI_Model = LAI_cycle(p['laip'])  # LAI model instance
            self.relative_LAI = self.LAI_Model.f  # LAI relative to annual maximum [..1]
        else:
            self.relative_LAI = 1.0

        # physical structure
        self.LAImax = p['LAImax']  # maximum annual 1-sided LAI [m2m-2]
        self.LAI = self.LAImax * self.relative_LAI  # current LAI
        self.lad_normed = p['lad']  # normalized leaf-area density [m-1]
        self.lad = self.LAI * self.lad_normed  # current leaf-area density [m2m-3]

        if MLM_ctr['ON']:
            self.dz = z[1] - z[0]
            # plant height [m]
            f = np.where(self.lad_normed > 0)[0][-1]
            self.hc = z[f]
            # leaf gas-exchange parameters
            self.photop0 = p['photop']   # A-gs parameters at pheno_state = 1.0 (dict)
            self.photop = self.photop0.copy()  # current A-gs parameters (dict)
            # leaf properties
            self.leafp = p['leafp']  # leaf properties (dict)
            self.StomaModel = MLM_ctr['StomaModel']
            self.Switch_Ebal = MLM_ctr['Ebal']
            self.gsref = 0.0
        else:
            self.gsref = p['gsref']
 
    def _update_daily(self, doy, T, PsiL=0.0):
        """
        Updates PlantType pheno_state, gas-exchange parameters, LAI & lad
        Args:
            doy - day of year
            T - daily air temperature [degC]
            Psi_leaf - leaf (or soil) water potential, <0 [MPa]
        NOTE: CALL ONCE PER DAY
        """
        PsiL = np.minimum(-1e-5, PsiL)

        if self.Switch_pheno:
            self.pheno_state = self.Pheno_Model._run(T, out=True)

        if self.Switch_lai:
            self.relative_LAI =self.LAI_Model._run(doy, T, out=True)
            self.LAI = self.relative_LAI * self.LAImax
            self.lad = self.lad_normed * self.LAI
        """
        # scale photosynthetic capacity using vertical N gradient
        f = 1.0
        if 'kn' in self.photop0:
            kn = self.photop0['kn']
            Lc = np.flipud(np.cumsum(np.flipud(self.lad*self.dz)))
            Lc = Lc / Lc[0]
            f = np.exp(-kn*Lc)
            # print f
            # plt.plot(f, z, 'k', Lc, z, 'g-')
        # preserve proportionality of Jmax and Rd to Vcmax
        self.photop['Vcmax'] = f * self.pheno_state * self.photop0['Vcmax']
        self.photop['Jmax'] =  f * self.pheno_state * self.photop0['Jmax']
        self.photop['Rd'] =  f * self.pheno_state * self.photop0['Rd']

        if self.Switch_WaterStress:
            b = self.photop0['drp']
            if 'La' in self.photop0:
                # lambda increases with decreasing Psi as in Manzoni et al., 2011 Funct. Ecol.
                self.photop['La'] = self.photop0['La'] * np.exp(-b*PsiL)
            if 'm' in self.photop0:  # medlyn g1-model, decrease with decreasing Psi  
                self.photop['m'] = self.photop0['m'] * np.maximum(0.05, np.exp(b*PsiL))
        """

    def leaf_gasexchange(self, f_sl, H2O, CO2, T, U, P, Q_sl1, Q_sh1, SWabs_sl, SWabs_sh, LWl):
        """
        Compute leaf gas-exchange for PlantType
        """

        # --- sunlit leaves
        sl = leaf_interface(self.photop, self.leafp, H2O, CO2, T, Q_sl1,
                            SWabs_sl, LWl, U, P=P, model=self.StomaModel,
                            Ebal=self.Switch_Ebal, dict_output=True)

        # --- shaded leaves
        sh = leaf_interface(self.photop, self.leafp, H2O, CO2, T, Q_sh1, 
                            SWabs_sh, LWl, U, P=P, model=self.StomaModel,
                            Ebal=self.Switch_Ebal, dict_output=True)

        # integrate water and C fluxes over all leaves in PlantType, store resuts
        pt_stats = self._integrate(sl, sh, f_sl)

        # --- sink/source terms
        dqsource = (f_sl*sl['E'] + (1.0 - f_sl)*sh['E'])*self.lad  # mol m-3 s-1
        dcsource = - (f_sl*sl['An'] + (1.0 - f_sl)*sh['An'])*self.lad  #umol m-3 s-1
        dRstand = np.sum((f_sl*sl['Rd'] + (1.0 - f_sl)*sh['Rd'])*self.lad*self.dz)  # add dark respiration umol m-2 s-1

        return pt_stats, dqsource, dcsource, dRstand

    def _integrate(self, sl, sh, f_sl):
        """
        integrates layerwise statistics (per unit leaf area) to plant level
        Arg:
            sl, sh - dict of leaf-level outputs for sunlit and shaded leaves:         
            
            x = {'An': An, 'Rd': Rd, 'E': fe, 'H': H, 'Fr': Fr, 'Tl': Tl, 'Ci': Ci,
                 'Cs': Cs, 'gs_v': gsv, 'gs_c': gs_opt, 'gb_v': gb_v}
        Returns:
            y - plant level statistics
        """
        # plant fluxes, weight factors is sunlit and shaded LAI at each layer
        f1 = f_sl*self.lad*self.dz
        f2 = (1.0 - f_sl)*self.lad*self.dz
        
        keys = ['An', 'Rd', 'E']
        y = {k: np.nansum(sl[k]*f1 + sh[k]*f2) for k in keys}
        del keys

        # effective statistics; layerwise fluxes weighted by An
        g1 = f1 * sl['An'] / np.nansum(f1 * sl['An'] + f2 * sh['An'])
        g2 = f2 * sh['An'] / np.nansum(f1 * sl['An'] + f2 * sh['An'])
        # print sum(g1 + g2)        
        keys = ['Tl', 'Ci', 'Cs', 'gs_v', 'gb_v']
        
        y.update({k: np.nansum(sl[k]*g1 + sh[k]*g2) for k in keys})
        # print y
        return y

class ForestFloor():
    """
    Forest floor
    """
    def __init__(self, p):

        from evapotranspiration import ForestFloor_Evap

        self.Evap = ForestFloor_Evap(p)

    def _run(self, dt, T, AE, VPD, Ra, beta, SWE):
        """
        Args:
            T: air temperature [degC]
            AE: available energy at forest floor [W m-2]
            VPD: vapor pressure deficit in [kPa]
            Ra: soil aerodynamic resistance [s m-1]
            beta: 
        Returns:
            Efloor: forest floor evaporation [m]
        """
        
        if SWE > 0:  # snow on the ground
            Efloor = 0.0
        else:
            Efloor = self.Evap._run(dt, T, AE, VPD, Ra, beta)

        # Interception?

        return Efloor