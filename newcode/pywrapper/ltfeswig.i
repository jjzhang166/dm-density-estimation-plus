%module ltfepy

%{
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <stdint.h>
#include "../ltfeheader.h"
#include "../ltfereader.h"
%}

%include <typemaps.i>
%include <carrays.i>
%include <std_string.i>
%include <std_vector.i>
%include "../ltfeheader.h"
%include "../ltfereader.h"

#double array, use this one to wap the array
%array_class(double, dArray);
%array_class(float, fArray);

%include "array_util.i"
