#ifndef __LY__DENS_CIC
#define __LY__DENS_CIC

class CIC{
public:
    CIC(double boxSize, int gridsize, bool isVelocityField = false);
    
    // render a single particle into the mesh
    void render_particle(double * pos, double * vel, double mass = 1.0);
    
    // render a list of numParts particles into the mesh
    void render_particle(double * pos, double * vel, int numParts,
                         double mass = 1.0);
    
    double * getDensityField();
    double * getVelocityXField();
    double * getVelocityYField();
    double * getVelocityZField();
    
private:
    double boxSize_;
    double mass_;
    double dx_;
    int gridsize_;
    bool isVelocityField_;
    double * densityField;
    double * velocityXField;
    double * velocityYField;
    double * velocityZField;
    
    // initialize a grid
    void clearGrid();
    
    // add a value to the grid cells
    void addToGridCells(double * grids, double * pos, double value);
};



#endif
