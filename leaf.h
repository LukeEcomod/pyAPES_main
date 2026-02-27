#include <float.h>
#include <stdlib.h>

//: machine epsilon
const double EPS = DBL_EPSILON;
//: [K], zero degrees celsius in Kelvin
const double DEG_TO_KELVIN = 273.15;
//: [J mol-1 K-1], universal gas constant
const double GAS_CONSTANT = 8.314;
//: [K] reference temperature for photosynthetic parameters, 283.15 [K]
const double TN = 25.0 + DEG_TO_KELVIN;
//: [J mol-1] universal gas constant times 25C reference temperature in Kelvin
const double TN_GAS_CONSTANT = TN * GAS_CONSTANT;
// Reciprocal of the previous
const double ONE_OVER_TN_GAS_CONSTANT = 1.0 / (TN_GAS_CONSTANT);
//: [umol m-3], molar O2 concentration in air
const double O2_IN_AIR = 2.10e5;
//: [-] H2O to CO2 diffusivity ratio
const double H2O_CO2_RATIO = 1.6;

typedef struct PhotoParams {
    double *Vcmax;
    double *Jmax;
    double *Rd;

    double alpha;
    double theta;
    double g1;
    double g0;
    double beta;

    double Vcmax_T[3];
    double Jmax_T[3];
    double Rd_T[1];

    int tresp;
} PhotoParams;

typedef struct Outputs {
    double *An;
    double *Rd;
    double *fe;
    double *gs_opt;
    double *ci;
    double *cs;
} Outputs;

double max(double a, double b);

void photo_c3_medlyn_farquhar(PhotoParams *params, double *Qp, double *T,
                              double *VPD, double *ca, double *gb_c,
                              double *gb_v, double P, size_t size,
                              Outputs *outputs);
