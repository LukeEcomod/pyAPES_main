#include <math.h>

#include "leaf.h"

double max(double a, double b) { return a > b ? a : b; }

void photo_c3_medlyn_farquhar(PhotoParams *params, double *Qp, double *T,
                              double *VPD, double *ca, double *gb_c,
                              double *gb_v, double P, size_t size,
                              Outputs *outputs) {
    /*
    Leaf gas-exchange by Farquhar-Medlyn Unified Stomatal Optimality (USO)
    -model (Medlyn et al., 2011 GCB), where co-limitation as in standard
    Farquhar-model

    Args:
        photop - parameter dict with keys: Vcmax, Jmax, Rd, alpha, theta, beta,
    g1, g0, tresp can be scalars or arrays. tresp - dictionary with keys: Vcmax,
    Jmax, Rd: temperature sensitivity parameters. OMIT key 'tresp' if no
    temperature adjustments for photoparameters! Qp - incident PAR at leaves
    [umolm-2s-1] T - leaf temperature [degC] VPD - leaf-air vapor pressure
    difference [mol mol-1] ca - ambient CO2 [ppm] gb_c - boundary-layer
    conductance for CO2 [mol m-2 s-1] gb_v - boundary-layer conductance for H2O
    [mol m-2 s-1]

    Returns:
        An - net CO2 flux [umol m-2 s-1]
        Rd - dark respiration [umol m-2 s-1]
        fe - leaf transpiration rate [mol m-2 s-1]
        gs - stomatal conductance for CO2 [mol m-2 s-1]
        ci - leaf internal CO2 [ppm]
        cs - leaf surface CO2 [ppm]
    */

    for (size_t i = 0; i < size; i++) {
        const double Tk = T[i] + DEG_TO_KELVIN;

        double vpd = 1e-3 * VPD[i] * P;
        vpd = vpd < EPS ? EPS : vpd; // kPa
                                     //
        const double TN_GAS_CONSTANT_Tk = TN * GAS_CONSTANT * Tk;
        const double exp_scaling_tk = (Tk - TN) / TN_GAS_CONSTANT_Tk;

        // --- CO2 compensation point -------
        double Tau_c = 42.75 * exp(37830 * exp_scaling_tk);

        // ---- Kc & Ko (umol/mol), Rubisco activity for CO2 & O2 ------
        const double Kc = 404.9 * exp(79430.0 * exp_scaling_tk);
        const double Ko = 2.784e5 * exp(36380.0 * exp_scaling_tk);

        double Jmax = params->Jmax[i];
        double Vcmax = params->Vcmax[i];
        double Rd = params->Rd[i];
        if (0 != params->tresp) {
            const double ONE_OVER_T_GAS_CONSTANT = 1.0 / (Tk * GAS_CONSTANT);
            const double VcHa = 1e3 * params->Vcmax_T[0];
            const double VcHd = 1e3 * params->Vcmax_T[1];
            const double VcSd = params->Vcmax_T[2];
            double NOM =
                exp(VcHa * exp_scaling_tk) *
                (1.0 + exp((TN * VcSd - VcHd) * ONE_OVER_TN_GAS_CONSTANT));
            double DENOM =
                1.0 + exp((Tk * VcSd - VcHd) * ONE_OVER_T_GAS_CONSTANT);
            Vcmax = Vcmax * NOM / DENOM;

            const double JHa = 1e3 * params->Jmax_T[0];
            const double JHd = 1e3 * params->Jmax_T[1];
            const double JSd = params->Jmax_T[2];
            NOM = exp(JHa * exp_scaling_tk) *
                  (1.0 + exp((TN * JSd - JHd) * ONE_OVER_TN_GAS_CONSTANT));
            DENOM = 1.0 + exp((Tk * JSd - JHd) * ONE_OVER_T_GAS_CONSTANT);
            Jmax = Jmax * NOM / DENOM;

            const double RdHa = 1e3 * params->Rd_T[0];
            Rd = Rd * exp(RdHa * exp_scaling_tk);
            Tau_c = 42.75 * exp(37830.0 * exp_scaling_tk);
        }

        // --- model parameters k1_c, k2_c [umol/m2/s]
        const double Km = Kc * (1.0 + O2_IN_AIR / Ko);
        const double qp = Qp[i];
        const double jaqp = Jmax + params->alpha * qp;
        const double jaqp2 = jaqp * jaqp;
        const double J =
            (jaqp -
             sqrt(jaqp2 - (4.0 * params->theta * Jmax * params->alpha * qp))) /
            (2.0 * params->theta);

        // --- iterative solution for cs and ci
        double err = 9999.0;
        size_t cnt = 1;
        const double cai = ca[i];
        double cs = cai;       // leaf surface CO2
        double ci = 0.8 * cai; // internal CO2
        double An = 0.0;
        double gs_opt = 0.0;

        const size_t MaxIter = 50;
        while (err > 0.0001 && cnt < MaxIter) {
            //  -- rubisco -limited rate
            const double Av = Vcmax * (ci - Tau_c) / (ci + Km);
            // -- RuBP -regeneration limited rate
            const double Aj = J / 4.0 * (ci - Tau_c) / (ci + 2.0 * Tau_c);

            const double x = Av + Aj;
            const double y = Av * Aj;
            An = (x - sqrt(x * x - 4.0 * params->beta * y)) /
                     (2.0 * params->beta) -
                 Rd; // co-limitation

            const double An1 = max(An, 0.0);
            // stomatal conductance
            gs_opt = params->g0 + (1.0 + params->g1 / sqrt(vpd)) * An1 / cs;
            gs_opt = max(gs_opt, params->g0); // gcut is the lower limit
            // CO2 supply
            cs = max(cai - An1 / gb_c[i], 0.5 * cai); // through boundary layer
            const double ci0 = ci;
            ci = max(cs - An1 / gs_opt, 0.1 * cai); // through stomata

            err = (ci0 - ci) * (ci0 - ci);
            cnt += 1;
        }

        // when Rd > photo, assume stomata closed and ci == ca
        if (An < 0) {
            ci = cai;
            cs = cai;
            gs_opt = params->g0;
        }

        const double gb_vi = gb_v[i];
        const double gs_v = H2O_CO2_RATIO * gs_opt;
        const double geff = (gb_vi * gs_v) / (gb_vi + gs_v); // molm-2s-1
        const double fe = geff * vpd / (1e-3 * P); // leaf transpiration rate

        outputs->An[i] = An;
        outputs->Rd[i] = Rd;
        outputs->fe[i] = fe;
        outputs->gs_opt[i] = gs_opt;
        outputs->ci[i] = ci;
        outputs->cs[i] = cs;
    }
}
