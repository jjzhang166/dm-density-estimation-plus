%module GADGETPy

%{
#include <sstream>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cmath>
#include <cstring>
#include "GadgetReader/gadgetreader.hpp"
#include "GadgetReader/gadgetheader.h"
%}

%include "GadgetReader/gadgetheader.h"
%include "GadgetReader/gadgetreader.hpp"

#double array, use this one to wap the array
%array_class(double, dArray);