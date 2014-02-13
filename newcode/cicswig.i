%module cicpy

%{
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cic.h"
%}

%include "carrays.i"
%include "cic.h"

#double array, use this one to wap the array
%array_class(double, dArray);

%include "array_util.i"