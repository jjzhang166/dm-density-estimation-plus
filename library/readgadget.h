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


/*
This class suppose to read gadget file in any form: single file or multiple file.
It works in a low memory mode and a high memory mode. In low memory mode, it reads the 
data block by block (each time it reads a block of data to the memory). In high memory
mode, it reads all the data into the memory. The low memory mode requires the data grid
be 2^n.
*/

class GSnap {
public:
    
    /*deprecated*/
    //isHighMem - if store all the data in the memory.
    //If gridsize == -1, then set up the gridsize to be (Npart)^(1/3)
    //parttype =? six kinds
	GSnap(std::string filename,
          bool isHighMem,
          int parttype =1,
          int gridsize = -1)
					;
    
    //read all data into memory
    GSnap(
          std::string filename,
          int parttype =1,
          int gridsize = -1
          );
    
    //this reads a multi-file into memory
    GSnap(
          std::string prefix,
          std::string basename,
          int numfiles,
		  bool isHighMem = false,
          int parttype =1,
          int gridsize = -1
          );
    
	~GSnap();
	gadget_header header;
	uint32_t Npart;

	void readPosBlock(Point * &posblock,
                    int imin, int jmin, int kmin, 
                    int imax, int jmax, int kmax, 
                    bool isPeriodical = true, 
                    bool isOrdered = false
                    );

	void readBlock(Point * &posblock,
                   Point * &velocityblock,
                   int imin, int jmin, int kmin,
                   int imax, int jmax, int kmax,
                   bool isPeriodical = true,
                   bool isOrdered = false
                   );
				   

    /***********************************************************/
    // read a list of points, from position ptr
    void readPos(std::fstream &file,
                 Point * pos,
                 long ptr,
                 long count
                 );
    
    void readPos(std::string &file,
                 Point * pos,
                 long ptr,
                 long count
                 );

	void readVel(std::fstream &file,
                 Point * vel,
                 long ptr,
                 long count
                 );
    
    void readVel(std::string &file,
                 Point * vel,
                 long ptr,
                 long count
                 );
    /***********************************************************/


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

	bool isMultifile_;
	string prefix_;
	int numOfFiles_;
	string basename_;
	int * numOfParts_;
    int * multStartInd_;
    int * multEndInd_;
    int * minInd_;
    int * maxInd_;


    //bool isHighMem_;
    int startind;
    int endind;
    
    //total number of particles in the gadget file
    int totalparts;
    
    Point * allpos_;
    Point * allvel_;
    uint32_t * allind_;
        
    // read a point, from position ptr
    Point readPos(std::fstream &file, long ptr);
    Point readPos(std::string &file, long ptr);

	Point readVel(std::fstream &file, long ptr);
    Point readVel(std::string &file, long ptr);

    
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
    
    void readIndex(
                   std::string &file,
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

	void init_singlefile(
		string filename,
        bool isHighMem,
        int parttype,
        int gridsize
		);

};

#endif /* READGADGET_H_ */
