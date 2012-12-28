/*
 * grids.cpp
 *
 *  Created on: Dec 18, 2012
 *      Author: lyang
 */

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

using namespace std;

#include "tetrahedron.h"
#include "grids.h"

GridManager::GridManager(string filename, int gridsize, int subgridsize) {
	filename_ = filename;
	gridsize_ = gridsize;
	subgridsize_ = subgridsize;
	if (gridsize % subgridsize != 0) {
		printf("Subgridsize (%d) must be a dividor of Gridsize(%d)!.\n",
				subgridsize, gridsize);
		exit(1);
	}
	subgrid_num = gridsize / subgridsize; // + 1;
	subgrid_num = subgrid_num * subgrid_num * subgrid_num;
	grid_ = new double[subgridsize_ * subgridsize_ * subgridsize_];
	int loop_i;
	//clear
	for (loop_i = 0; loop_i < subgridsize_ * subgridsize_ * subgridsize_;
			loop_i++) {
		grid_[loop_i] = 0;
	}
	for (loop_i = 0; loop_i < subgrid_num; loop_i++) {
		current_block_ind = loop_i;
		saveGrid();
	}
	current_block_ind = 0;

	boxStartPoint_.x = 0;
	boxStartPoint_.y = 0;
	boxStartPoint_.z = 0;

	boxEndPoint_.x = 32000;
	boxEndPoint_.y = 32000;
	boxEndPoint_.z = 32000;

}

GridManager::GridManager(std::string filename, int gridsize, int subgridsize,
		Point boxStartPoint, Point boxEndPoint) {
	GridManager(filename, gridsize, subgridsize);
	boxStartPoint_ = boxStartPoint;
	boxEndPoint_ = boxEndPoint;
}

GridManager::~GridManager() {
	delete grid_;
}

int GridManager::getSubGridNum() {
	return subgrid_num;
}

int GridManager::getGridSize() {
	return this->gridsize_;
}

int GridManager::getSubGridSize() {
	return this->subgridsize_;
}

int GridManager::getCurrentInd() {
	return this->current_block_ind;
}

bool GridManager::loadGrid(int ind) {
	if (ind < this->subgrid_num) {
		if (ind == this->current_block_ind) {
			return true;
		} else {
			string filename = getSubGridFileName(ind);
			ifstream gridFile(filename.c_str(), ios::in | ios::binary);
			int c_ind = 0;
			int c_sub = 0;
			int c_grid = 0;
			gridFile.read((char *) &c_ind, sizeof(int));
			gridFile.read((char *) &c_sub, sizeof(int));
			gridFile.read((char *) &c_grid, sizeof(int));
			if (c_ind != ind || c_sub != subgridsize_ || c_grid != gridsize_) {
				printf(
						"File format incorrect! File is changed by other programs.\n");
				gridFile.close();
				exit(1);
			}
			gridFile.read((char *) grid_,
					sizeof(double) * subgridsize_ * subgridsize_
							* subgridsize_);
			current_block_ind = ind;
			gridFile.close();
			return true;
		}
	} else {
		return false;
	}
}

bool GridManager::loadGrid(int i, int j, int k) {
	return loadGrid(this->convertToIndex(i, j, k));
}

void GridManager::saveGrid() {
	string filename = getSubGridFileName(this->current_block_ind);
	ofstream gridFile(filename.c_str(), ios::out | ios::binary);
	gridFile.write((char *) &(this->current_block_ind), sizeof(int));
	gridFile.write((char *) &(this->subgridsize_), sizeof(int));
	gridFile.write((char *) &(this->gridsize_), sizeof(int));
	gridFile.write((char *) this->grid_,
			sizeof(double) * subgridsize_ * subgridsize_ * subgridsize_);
	gridFile.close();
}

double * GridManager::getSubGrid() {
	return this->grid_;
}

bool GridManager::loadGridByActualCoor(int i, int j, int k) {
	if (isInCurrentBlock(i, j, k)) {
		return true;
	} else {
		int ind = convertToIndexByActualCoor(i, j, k);
		//printf("ind %d\n", ind);
		if (ind >= this->subgrid_num) {
			//printf("afdafadfasf %d\n", subgrid_num);
			return false;
		} else {
			//printf("ind %d\n", ind);
			return this->loadGrid(ind);
		}

	}
}

int GridManager::convertToIndex(int i, int j, int k) {
	int l = gridsize_ / subgridsize_; // + 1;
	return k * l * l + j * l + i;
}

int GridManager::convertToIndexByActualCoor(int i, int j, int k) {
	int i_ = i / subgridsize_;
	int j_ = j / subgridsize_;
	int k_ = k / subgridsize_;
	int ind = convertToIndex(i_, j_, k_);
	return ind;
}

bool GridManager::isInCurrentBlock(int i, int j, int k) {
	int ind = convertToIndexByActualCoor(i, j, k);
	//printf("dddd %d \n", ind);
	if (ind == current_block_ind) {
		return true;
	} else {
		return false;
	}
}

string GridManager::getSubGridFileName(int ind) {
	string suffix =
			static_cast<ostringstream*>(&(ostringstream() << ind))->str();
	return this->filename_ + suffix;
}

double GridManager::getValueByActualCoor(int i, int j, int k) {
	if (!isInCurrentBlock(i, j, k)) {
		//printf("%d %d %d", i, j, k);
		saveGrid();
		loadGridByActualCoor(i, j, k);
	}
	int i_, j_, k_;
	actual2Current(i, j, k, i_, j_, k_);
	return getValue(i_, j_, k_);
}

double GridManager::getValue(int i, int j, int k) {
	return grid_[k * subgridsize_ * subgridsize_ + j * subgridsize_ + i];
}

void GridManager::actual2Current(int ai, int aj, int ak, int & ci, int & cj,
		int & ck) {
	int i0 = ai / subgridsize_ * subgridsize_;
	int j0 = aj / subgridsize_ * subgridsize_;
	int k0 = ak / subgridsize_ * subgridsize_;
	ci = ai - i0;
	cj = aj - j0;
	ck = ak - k0;
}

void GridManager::current2Actual(int ci, int cj, int ck, int & ai, int & aj,
		int & ak) {
	int nsubg = gridsize_ / subgridsize_; // + 1;

	int i0 = (current_block_ind % nsubg) * subgridsize_;
	int j0 = (current_block_ind / nsubg % nsubg) * subgridsize_;
	int k0 = (current_block_ind / nsubg / nsubg % nsubg) * subgridsize_;

	//printf("%d %d %d %d %d\n", current_block_ind, i0, j0, k0, nsubg);

	ai = ci + i0;
	aj = cj + j0;
	ak = ck + k0;
}

void GridManager::setValue(int i, int j, int k, double value) {
	grid_[k * subgridsize_ * subgridsize_ + j * subgridsize_ + i] = value;
}

void GridManager::setValueByActualCoor(int i, int j, int k, double value) {
	if (!isInCurrentBlock(i, j, k)) {
		saveGrid();
		loadGridByActualCoor(i, j, k);
	}
	int ci, cj, ck;
	actual2Current(i, j, k, ci, cj, ck);
	//cout << "ci cj ck currentind: " << ci << " " << cj << " " << ck << " " << this->current_block_ind << endl;
	setValue(ci, cj, ck, value);
}

Point GridManager::getStartPoint() {
	return boxStartPoint_;
}
Point GridManager::getEndPoint() {
	return boxEndPoint_;
}

Point GridManager::getPointByActualCoor(int i, int j, int k) {
	double fx, fy, fz;
	Point retP;
	fx = (double) i / (double) gridsize_;
	fy = (double) j / (double) gridsize_;
	fz = (double) k / (double) gridsize_;
	retP.x = fx * (boxEndPoint_.x - boxStartPoint_.x) + boxStartPoint_.x;
	retP.y = fy * (boxEndPoint_.y - boxStartPoint_.y) + boxStartPoint_.y;
	retP.z = fz * (boxEndPoint_.z - boxStartPoint_.z) + boxStartPoint_.z;
	return retP;
}

Point GridManager::getPoint(int i, int j, int k) {
	int ai, aj, ak;
	current2Actual(i, j, k, ai, aj, ak);
	return getPointByActualCoor(ai, aj, ak);
}

