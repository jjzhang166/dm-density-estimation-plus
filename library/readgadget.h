/*
 * readgadget.h
 *
 *  Created on: Dec 27, 2012
 *      Author: lyang
 */

#ifndef READGADGET_H_
#define READGADGET_H_
#include <fstream>
#include "types.h"
#include "tetrahedron.h"
#include "gadgetheader.h"

/************WARNNING*****************************************
 *! LOW MEMORY VERSION CAN ONLY WORK IN REGULAR GRID
 *! high memory version can work in any grid
 *! ORDERED DATA has not been implemented
 *************************************************************/

class GSnap {
public:
    //isHighMem - if store all the data in the memory.
    //If gridsize == -1, then set up the gridsize to be (Npart)^(1/3)
    //parttype =? six kinds
	GSnap(std::string filename, bool isHighMem = true,
          int parttype =1, int gridsize = 512);
	~GSnap();
	gadget_header header;
	uint32_t Npart;

	void readPosBlock(Point * &posblock,
                    int imin, int jmin, int kmin, 
                    int imax, int jmax, int kmax, 
                    bool isPeriodical = true, 
                    bool isOrdered = false
                    );

	void readBlock(Point * &posblock, Point * &velocityblock, 
                    int imin, int jmin, int kmin, 
                    int imax, int jmax, int kmax, 
			        bool isPeriodical = true, 
                    bool isOrdered = false
                   );

    /*****************SLOW, COPY DATA***************************/
    // read a list of points, from position ptr
    void readPos(std::fstream &file, Point * pos, long ptr, long count);
	void readVel(std::fstream &file, Point * vel, long ptr, long count);
    /***********************************************************/


    //recomended to use this version, if high memory
    /******************RECOMENDED USE THIS**********************/
    //Data are all sorted along indexes
    //get all the position data
    Point * getAllPos(){
        return allpos_;
    };

    //get all the velocity data
    Point * getAllVel(){
        return allvel_;
    }
    /***********************************************************/

private:
	//uint32_t * ids;
	string filename_;
	int grid_size;

    bool isHighMem_;
    int startind;
    int endind;
    
    //total number of particles in the gadget file
    int totalparts;
    
    Point * allpos_;
    Point * allvel_;
    uint32_t * allind_;
        
    // read a point, from position ptr
    Point readPos(std::fstream &file, long ptr);
	Point readVel(std::fstream &file, long ptr);


	void readIndex(std::fstream &file, int *block,
			int imin, int jmin, int kmin, int imax, 
            int jmax, int kmax, bool isPeriodical = true, 
            bool isOrdered = false);
};

#endif /* READGADGET_H_ */
