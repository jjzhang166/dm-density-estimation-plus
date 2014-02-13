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


/*// python wrapper
extern "C" {
    CIC* CIC_new(double boxSize, int gridsize, bool isVelocityField){
        return new CIC(boxSize, gridsize, isVelocityField);
    };
    
    void CIC_render_particle(CIC* cic, double * pos,
                         double * vel, int numParts,
                         double mass){
        cic->render_particle(pos, vel, numParts, mass);
    };
    
    double * CIC_getDensityField(CIC * cic){
        return cic->getDensityField();
    };
    double * CIC_getVelocityXField(CIC * cic){
        return cic->getVelocityXField();
    };
    double * CIC_getVelocityYField(CIC * cic){
        return cic->getVelocityYField();
    };
    double * CIC_getVelocityZField(CIC * cic){
        return cic->getVelocityZField();
    };
}*/

#endif
