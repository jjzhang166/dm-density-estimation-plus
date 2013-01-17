/*
 * grids.h
 * This file defines the grids system. The class will create a grids file of certain name.
 * Each time it loads a certain part of the whole grid using the standard grid system.
 *  Created on: Dec 17, 2012
 *      Author: lyang
 */

#ifndef GRIDS_MANAGER_H_
#define GRIDS_MANAGER_H_
#include <string>
#include <vector>

using namespace std;

#include "types.h"
#include "grid.h"

typedef Grids<REAL> GridManager;
typedef Grids<Point> GridVelocityManager;

#endif /* GRIDS_H_ */
