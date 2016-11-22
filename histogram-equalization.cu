#include <stdio.h>
#include <string.h>
#include <stdlib.h>


__global__ void construct_histogram_gpu(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin) {
	int i = blockIdx.x + blockDim.x + threadIdx.x;
	if (i < nbr_bin) {
		hist_out[i] = 0;
	}
	
	if (i < img_size) {
		atomicAdd(hist_out[img_in[i]], 1);
	}
}

void histogram_gpu(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin) {
}

__global__ void construct_lut() {
	int i = blockIdx.x + blockDim.x + threadIdx.x;
	if (i < nbr_bin) {
	
	}
}

__global__ void get_result_image() {
	int i = blockIdx.x + blockDim.x + threadIdx.x;
	if (i < img_size) {
	
	}
}


void histogram_equalization_gpu(unsigned char * img_out, unsigned char * img_in,
				int * hist_in, int img_size, int nbr_bin) {

}
