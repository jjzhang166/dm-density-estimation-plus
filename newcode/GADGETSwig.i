%module GADGETPy

%{
#include <sstream>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cmath>
#include <cstring>
#include <vector>
#include "GadgetReader/gadgetreader.hpp"
#include "GadgetReader/gadgetheader.h"
%}

%include <typemaps.i>
%include <carrays.i>
%include <std_string.i>
%include <std_vector.i>
%include "array_util.i"

%typemap(in) int32_t = int;
%typemap(in) int64_t = long long;
%typemap(in) uint32_t = unsigned int;
%typemap(in) uint64_t = unsigned long long;

/* Convert from Python --> C */
%typemap(in) int, long, int32_t, uint32_t, int16_t, uint16_t, uint64_t, int64_t{
    $1 = PyInt_AsLong($input);
}

/* Convert from C --> Python */
%typemap(out) int, long, int64_t, uint64_t{
    $result = PyInt_FromLong($1);
}

%include "GadgetReader/gadgetheader.h"
%include "GadgetReader/gadgetreader.hpp"

#double array, use this one to wap the array
%array_class(double, dArray);
%array_class(int, intArray);
%array_class(long, longArray);



#%array_class(uint32_t, uInt32Array);
#%array_class(int32_t, int32Array);
#%array_class(uint64_t, uInt64Array);
#%array_class(int64_t, int64Array);
