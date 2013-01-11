/*
 * grids.h
 * This file defines the grids system. The class will create a grids file of certain name.
 * Each time it loads a certain part of the whole grid using the standard grid system.
 *  Created on: Dec 17, 2012
 *      Author: lyang
 */

#ifndef GRIDS_H_
#define GRIDS_H_
#include <string>
#include <vector>

using namespace std;

#include "gadgetheader.h"
#include "tetrahedron.h"
#include "types.h"

class GridManager{
public:
	GridManager(std::string filename, int gridsize, int subgridsize);
	GridManager(std::string filename, int gridsize, int subgridsize,
			Point boxStartPoint, Point boxEndPoint);


	bool loadGrid(int i, int j, int k);		// load a sub grid at (i, j, k) of size subgridsize
											// the current memory will be directly cleared
											// if failed, returns false, keep the current grid
											// NOTE: THIS IS NOT THE SAME WITH loadGridByActualCoor
											// i-j-k here stands for the coordinates of subgrids in
											// larger grids


	bool loadGrid(int ind);					// load the grid at ind with ind = k * subgridsize * subgridsize + j * subgridsize + i

	bool loadGridByActualCoor(int i, int j, int k); //load the subgrid which has the current point
											// if the point is already in current block, do nothing
											// otherwise reload without saving

	//deprecated
	void saveGrid();						// save the current grid into the file on the harddrive
	
	//io
	void saveToFile();						// save the whole grid into file
	void saveToFileOld();					// save the whole grid into file in an old format
	void loadFromFile(string filename);		// load the grid data from file
	void loadFromFileOld(string filename);	// load the grid data from an old format

	int getGridSize();
	int getSubGridSize();
	int getSubGridNum();					// get total subgrids in the main grids

	REAL getValueByActualCoor(int i, int j, int k); //get the value of the actual point. If current
											// block does not has the point, save and reload

	REAL getValue(int i, int j, int k);	// get the value of current block

	void setValue(int i, int j, int k, REAL value);// set the value of current block
	void setValueByActualCoor(int i, int j, int k, REAL value); //set the value of the actual coor
											// if the value is not in current block, save and reload
	bool isInCurrentBlock(int i, int j, int k); //check whether coor is in current block

											//convert the actual index to current index
	void actual2Current(int ai, int aj, int ak, int &ci, int &cj, int &ck);
	void current2Actual(int ci, int cj, int ck, int &ai, int &aj, int &ak);

	int getCurrentInd();					// get current index number

	REAL * getSubGrid();					// get the actual subgrid memory

	Point getStartPoint();
	Point getEndPoint();

	Point getPoint(int i, int j, int k);	// get the point in units of the box of current index
	Point getPointByActualCoor(int i, int j, int k);

	~GridManager();
private:
	std::string filename_;
	vector<REAL *> grid_lists;
	int gridsize_;
	int subgridsize_;
	int subgrid_num;
	int current_block_ind;					// the current block index
	int convertToIndex(int i, int j, int k);// convert the i-j-k index of the sub grid to a index
	int convertToIndexByActualCoor(int i, int j, int k);// convert the i-j-k index of the sub grid to a index

	void initialize(string filename, int gridsize, int subgridsize);

	std::string getSubGridFileName(int ind);// get the filename of the subgrid
											// file format: index subgridsize gridsize [data block]

	Point boxStartPoint_, boxEndPoint_;		// the two 2 specifies two diagonal points

	REAL * grid_;							// the actual grid pointer

	gadget_header grid_header;

};



#endif /* GRIDS_H_ */
