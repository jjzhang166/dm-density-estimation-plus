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
    
    /*deprecated*/
    //isHighMem - if store all the data in the memory.
    //If gridsize == -1, then set up the gridsize to be (Npart)^(1/3)
    //parttype =? six kinds
#ifdef _MSC_VER
	__declspec(deprecated)
#endif
	GSnap(std::string filename,
          bool isHighMem,
          int parttype =1,
          int gridsize = 512)
#ifndef _MSC_VER
			__attribute__ ((deprecated));
#else
					;
#endif
    
    //read all data into memory
    GSnap(
          std::string filename,
          int parttype =1,
          int gridsize = 512
          );
    
    //this reads a multi-file into memory
    GSnap(
          std::string prefix,
          std::string basename,
          int numfiles,
          int parttype =1,
          int gridsize = 512
          );
    
	~GSnap();
	gadget_header header;
	uint32_t Npart;

#ifdef _MSC_VER
	__declspec(deprecated)
#endif
	void readPosBlock(Point * &posblock,
                    int imin, int jmin, int kmin, 
                    int imax, int jmax, int kmax, 
                    bool isPeriodical = true, 
                    bool isOrdered = false
                    )
#ifndef _MSC_VER
			__attribute__ ((deprecated));
#else
					;
#endif

#ifdef _MSC_VER
	__declspec(deprecated)
#endif
	void readBlock(Point * &posblock,
                   Point * &velocityblock,
                   int imin, int jmin, int kmin,
                   int imax, int jmax, int kmax,
                   bool isPeriodical = true,
                   bool isOrdered = false
                   )
				   
#ifndef _MSC_VER
			__attribute__ ((deprecated));
#else
					;
#endif

    /***********************************************************/
    // read a list of points, from position ptr
    void readPos(std::fstream &file,
                 Point * pos,
                 long ptr,
                 long count
                 );
	void readVel(std::fstream &file,
                 Point * vel,
                 long ptr,
                 long count
                 );
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

    //bool isHighMem_;
    int startind;
    int endind;
    
    //total number of particles in the gadget file
    int totalparts;
    
    Point * allpos_;
    Point * allvel_;
    uint32_t * allind_;
        
    // read a point, from position ptr
#ifdef _MSC_VER
	__declspec(deprecated)
#endif
    Point readPos(std::fstream &file, long ptr)
#ifndef _MSC_VER
			__attribute__ ((deprecated));
#else
					;
#endif


#ifdef _MSC_VER
	__declspec(deprecated)
#endif
	Point readVel(std::fstream &file, long ptr)
#ifndef _MSC_VER
			__attribute__ ((deprecated));
#else
					;
#endif

    
    bool isHighMem_;


	void readIndex(
                   std::fstream &file,
                   int *block,
                   int imin,
                   int jmin,
                   int kmin,
                   int imax,
                   int jmax,
                   int kmax,
                   bool isPeriodical = true,
                   bool isOrdered = false
                   );
};

#endif /* READGADGET_H_ */
