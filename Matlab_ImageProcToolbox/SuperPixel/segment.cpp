/*
Copyright (C) 2006 Pedro Felzenszwalb

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/

#include <cstdio>
#include <cstdlib>
#include <image.h>
#include <misc.h>
#include <pnmfile.h>
#include "segment-image.h"
#include "mex.h"
/*
int main(int argc, char **argv) {
  if (argc != 6) {
    fprintf(stderr, "usage: %s sigma k min input(ppm) output(ppm)\n", argv[0]);
    return 1;
  }
  
  float sigma = atof(argv[1]);
  float k = atof(argv[2]);
  int min_size = atoi(argv[3]);
	
  printf("loading input image.\n");
  image<rgb> *input = loadPPM(argv[4]);
	
  printf("processing\n");
  int num_ccs; 
  image<rgb> *seg = segment_image(input, sigma, k, min_size, &num_ccs); 
  savePPM(seg, argv[5]);

  printf("got %d components\n", num_ccs);
  printf("done! uff...thats hard work.\n");

  return 0;
}
*/

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]){
    double* input_image = (double*)(mxGetData(prhs[0]));
    const mwSize* image_size = mxGetDimensions(prhs[0]);
    size_t height = image_size[0];
    size_t width = image_size[1];
    size_t dims = mxGetNumberOfDimensions(prhs[0]);

    double sigma = mxGetScalar(prhs[1]);
    double min_size = mxGetScalar(prhs[2]);
    double sp_num = mxGetScalar(prhs[3]);

    image<rgb>* new_image = new image<rgb>(width, height);
    for(size_t i = 0;i < height;i++){
        for(size_t j = 0;j < width;j++){
            rgb color;
            color.r = input_image[i + height * (j + width * 0)];
            color.g = input_image[i + height * (j + width * 1)];
            color.b = input_image[i + height * (j + width * 2)];
            new_image->access[i][j] = color;
        }
    }

    int num_ccs; 
    // image<int>* pixel_label = new image<int>(width,height);    
    plhs[0] = mxCreateDoubleMatrix(height,width,mxREAL);
    double* pixel_label = (double*)mxGetPr(plhs[0]);
    segment_image(new_image, sigma, sp_num, min_size, &num_ccs, pixel_label);
}

