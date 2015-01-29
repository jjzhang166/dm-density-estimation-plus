#include <cstring>
#include <cstdio>
#include <cstdlib>     /* atoi */
#include <fitsio.h>
#include <string>
#include <sstream>
#include "haloread.h"

using namespace std;


int getTotalHaloNum(const char * fitsname){
    int status = 0;
    char keyname[FLEN_KEYWORD], colname[FLEN_VALUE], coltype[FLEN_VALUE];
    //int status = 0;   /* CFITSIO status value MUST be initialized to zero! */
    int hdupos, hdutype, bitpix, naxis, ncols, ii;
    int haloid = 0;
    float floatvalue;
    int intvalue;
    char ctype[FLEN_VALUE];
    long naxes[10], nrows;
    int leng = 0;
    fitsfile *fptr;
    
    
    if (!fits_open_table(&fptr, fitsname, READONLY, &status)){
        ii = 1;
        fits_make_keyn("TFORM", ii, keyname, &status); /* make keyword */
        fits_read_key(fptr, TSTRING, keyname, coltype, NULL, &status);
        strncpy(ctype, coltype, strlen(coltype) - 1);
        leng = atoi(ctype);
        fits_close_file(fptr, &status);
    }
    
    if(status > 0){
        fits_report_error(stderr, status);
    }
    
    return leng;
}

int getHaloById(const char * fitsname, int halonum, Halo * halo){
    int status = 0;   /* CFITSIO status value MUST be initialized to zero! */
    
    char keyname[FLEN_KEYWORD], colname[FLEN_VALUE], coltype[FLEN_VALUE];
    int hdupos, hdutype, bitpix, naxis, ncols, ii;
    float floatvalue;
    int intvalue;
    char ctype[FLEN_VALUE];
    long naxes[10], nrows;
    fitsfile *fptr;
    
    int numhalo = getTotalHaloNum(fitsname);
    
    if(halonum > numhalo){
        fprintf(stderr, "Halo number incorrect!");
        status = 1;
        return status;
    }
    
    if (!fits_open_table(&fptr, fitsname, READONLY, &status)){
        status = getHaloById(fptr, halonum, halo);
        fits_close_file(fptr, &status);
    }
    
    
    if(status > 0){
        fits_report_error(stderr, status);
    }
    return status;
}


int getHaloById(fitsfile * fptr, int halonum, Halo * phalo){
    char keyname[FLEN_KEYWORD], colname[FLEN_VALUE], coltype[FLEN_VALUE];
    int hdupos, hdutype, bitpix, naxis, ncols, ii;
    float floatvalue;
    int intvalue;
    char ctype[FLEN_VALUE];
    char cstriptype[FLEN_VALUE];
    long naxes[10], nrows;
    Halo &halo = *phalo;
    int status = 0;
    int haloid = halonum;
    
    fits_get_num_rows(fptr, &nrows, &status);
    fits_get_num_cols(fptr, &ncols, &status);
    
    //printf("nrows=%ld, ncols=%d\n", nrows, ncols);
    
    for (ii = 1; ii <= ncols; ii++)
    {
        fits_make_keyn("TTYPE", ii, keyname, &status); /* make keyword */
        fits_read_key(fptr, TSTRING, keyname, colname, NULL, &status);
        fits_make_keyn("TFORM", ii, keyname, &status); /* make keyword */
        fits_read_key(fptr, TSTRING, keyname, coltype, NULL, &status);
        
        //printf(" %3d %-16s %-16s\n", ii, colname, coltype);
        strncpy(ctype, coltype, strlen(coltype) - 1);
        int leng = atoi(ctype);
        char type = coltype[strlen(coltype) - 1];
        
        //fscanf("%s", string)
        stringstream astm;
        string astr;
        astm << colname;
        astm >> astr;       //trim the string, so that no spaces
        //printf("%s", astr.c_str());
        
        if(type == 'E'){
            fits_read_col(fptr, TFLOAT, ii, 1, haloid, 1, NULL, &floatvalue, NULL, &status);
            if(astr.compare("X") == 0){
                halo.x = floatvalue;
            }else if(astr.compare("Y") == 0){
                halo.y = floatvalue;
            }else if(astr.compare("Z") == 0){
                halo.z = floatvalue;
            }else if(astr.compare("VX") == 0){
                halo.vx = floatvalue;
            }else if(astr.compare("VY") == 0){
                halo.vy = floatvalue;
            }else if(astr.compare("VZ") == 0){
                halo.vz = floatvalue;
            }else if(astr.compare("LX") == 0){
                halo.lx = floatvalue;
            }else if(astr.compare("LY") == 0){
                halo.ly = floatvalue;
            }else if(astr.compare("LZ") == 0){
                halo.lz = floatvalue;
            }else if(astr.compare("I1") == 0){
                halo.i1 = floatvalue;
            }else if(astr.compare("I2") == 0){
                halo.i2 = floatvalue;
            }else if(astr.compare("I3") == 0){
                halo.i3 = floatvalue;
            }else if(astr.compare("I1X") == 0){
                halo.i1x = floatvalue;
            }else if(astr.compare("I1Y") == 0){
                halo.i1y = floatvalue;
            }else if(astr.compare("I1Z") == 0){
                halo.i1z = floatvalue;
            }else if(astr.compare("I2X") == 0){
                halo.i2x = floatvalue;
            }else if(astr.compare("I2Y") == 0){
                halo.i2y = floatvalue;
            }else if(astr.compare("I2Z") == 0){
                halo.i2z = floatvalue;
            }else if(astr.compare("I3X") == 0){
                halo.i3x = floatvalue;
            }else if(astr.compare("I3Y") == 0){
                halo.i3y = floatvalue;
            }else if(astr.compare("I3Z") == 0){
                halo.i3z = floatvalue;
            }else if(astr.compare("MASS") == 0){
                halo.mass = floatvalue;
            }else if(astr.compare("RADIUS") == 0){
                halo.radius = floatvalue;
            }else if(astr.compare("V_MAX") == 0){
                halo.v_max = floatvalue;
            }else if(astr.compare("M_MAX") == 0){
                halo.m_max = floatvalue;
            }else if(astr.compare("R_MAX") == 0){
                halo.r_max = floatvalue;
            }else if(astr.compare("PARENT_COMMON") == 0){
                halo.parent_common = floatvalue;
            }
        }else if(type == 'J'){
            fits_read_col(fptr, TINT32BIT, ii, 1, haloid, 1, NULL, &intvalue, NULL, &status);
            //printf(" %-16s = %d\n", colname, intvalue);
            if(astr.compare("MULTI") == 0){
                halo.multi = intvalue;
            }else if(astr.compare("PARENT_ID") == 0){
                halo.parent_id = intvalue;
            }
        }
    }
    
    return status;
}
