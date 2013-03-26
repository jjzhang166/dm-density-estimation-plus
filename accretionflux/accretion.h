#ifndef __LY__ACCRETION_SPHERE__
#define __LY__ACCRETION_SPHERE__
#include "types.h"
#include "tetrahedron.h"
    double accreation_sphere_rate(int counts, Point * posdata, Point * veldata, double mass, Point &halocenter, double r1, double r2);
#endif
