#include <cstring>
#include <cstdio>
#include <cstdlib>     /* atoi */
#include <fitsio.h>

using namespace std;

int main(int argc, char *argv[])
{
    fitsfile *fptr;         /* FITS file pointer, defined in fitsio.h */
    char keyname[FLEN_KEYWORD], colname[FLEN_VALUE], coltype[FLEN_VALUE];
    int status = 0;   /* CFITSIO status value MUST be initialized to zero! */
    int single = 1, hdupos, hdutype, bitpix, naxis, ncols, ii;
    int haloid = 0;
    float floatvalue;
    int intvalue;
    char ctype[FLEN_VALUE];
    long naxes[10], nrows;


    if (argc != 3) {
      printf("Usage:  haloprop filename #halo\n");
      printf("\n");
      printf("The the halo information of the specific halo numer\n");
      printf("The input file must be a correct halo catalog file\n");
      return(0);
    }
    
    haloid = atoi(argv[2]);
    if (!fits_open_table(&fptr, argv[1], READONLY, &status))
    {
      fits_get_hdu_num(fptr, &hdupos);  /* Get the current HDU position */
        fits_get_hdu_type(fptr, &hdutype, &status);  /* Get the HDU type */

        printf("\nHDU #%d  ", hdupos);
        fits_get_num_rows(fptr, &nrows, &status);
        fits_get_num_cols(fptr, &ncols, &status);
        //printf("%d columns x %ld rows\n", ncols, nrows);
        
        ii = 1;
        fits_make_keyn("TFORM", ii, keyname, &status); /* make keyword */
        fits_read_key(fptr, TSTRING, keyname, coltype, NULL, &status);
        strncpy(ctype, coltype, strlen(coltype) - 1); 
        int leng = atoi(ctype);

        printf(" Halo #%d / %d: \n", haloid, leng);

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
            if(type == 'E'){
                fits_read_col(fptr, TFLOAT, ii, 1, haloid, 1, NULL, &floatvalue, NULL, &status);
                printf(" %-16s = %f\n", colname, floatvalue);
            }else if(type == 'J'){
                fits_read_col(fptr, TINT32BIT, ii, 1, haloid, 1, NULL, &intvalue, NULL, &status);
                printf(" %-16s = %d\n", colname, intvalue);
            }
        }

        fits_movrel_hdu(fptr, 1, NULL, &status);  /* try move to next ext */

      if (status == END_OF_FILE) status = 0; /* Reset normal error */
      fits_close_file(fptr, &status);
    }

    if (status) fits_report_error(stderr, status); /* print any error message */
    return(status);            
}
