#include <stdio.h>
#include <string.h>
#include <stdlib.h>


// Device code
__global__ void construct_histogram_gpu(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin) {
	int i = blockIdx.x + blockDim.x + threadIdx.x;
	if (i < nbr_bin) {
		hist_out[i] = 0;
	}
	
	if (i < img_size) {
		atomicAdd(hist_out[img_in[i]], 1);
	}
}
// Host code
void histogram_gpu(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin) {
	// Allocate vectors in device memory
	int* hist_out_gpu;
	cudaMalloc(&hist_out_gpu, nbr_bin*(sizeof(int));	
	unsigned char* img_in_gpu;
	cudaMalloc(&img_in_gpu, img_size*(sizeof(unsigned char));
	int img_size_gpu;
	cudaMalloc(&img_size_gpu, sizeof(img_size));
	int nbr_bin_gpu;
	cudaMalloc(&nbr_bin_gpu, sizeof(nbr_bin));
	// Copy vectors from host memory to device memory
	cudaMemcpy(hist_out_gpu, hist_out, nbr_bin*(sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(img_in_gpu, img_in, img_size*(sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(img_size_gpu, img_size, size(img_size), cudaMemcpyHostToDevice):
	cudaMemcpy(nbr_bin_gpu, nbr_bin, size(nbr_bin), cudaMemcpyHostToDevice);
	// Invoke kernel
	int blocksPerGrid = (img_size + nbr_bin - 1) / nbr_bin;
	construct_histogram_gpu<<blocksPerGrid, nbr_bin>>
		(hist_out_gpu, img_in_gpu, img_size_gpu, nbr_bin_gpu);
	// Copy result from device memory to host memory
	cudaMemcpy(hist_out, hist_out_gpu, nbr*(sizeof(int)), cudaMemcpyDeviceToHost);
	// Free device memory
	cudaFree(hist_out_gpu);
	cudaFree(img_in_gpu);
	cudaFree(img_size_gpu);
	cudaFree(nbr_bin_gpu);
}
// Device code
__global__ void construct_lut_gpu(int * cdf, int * lut, int * hist_in, 
				  int nbr_bin, int min, int d) {
	int i = blockIdx.x + blockDim.x + threadIdx.x;
	if (i < nbr_bin) {
		cdf += hist_in[i];
		lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
		if(lut[i] < 0) {
			lut[i] = 0;
		}	
	}
}
// Device code
__global__ void get_result_image_gpu() {
	int i = blockIdx.x + blockDim.x + threadIdx.x;
	if (i < img_size) {
		if(lut[img_in[i]] > 255) {
			img_out[i] = 255;
		}
		else {
			img_out[i] = (unsigned char)lut[img_in[i]];
		}
	}
}
// Host code
void histogram_equalization_gpu(unsigned char * img_out, unsigned char * img_in,
				int * hist_in, int img_size, int nbr_bin) {
	int* lut = (int*)malloc(sizeof(int)*nbr_bin);
	int* cdf = (int*)malloc(sizeof(int)*nbr_bin);
	int i, min, d;
	min = 0;
	i = 0;
	while (min == 0) {
		min = hist_in[i++];
	}
	d = img_size - min;
	// Allocate vectors in device memory
	int* lut_gpu;
	cudaMalloc(&lut_gpu, nbr_bin*(sizeof(int));
	int* cdf_gpu;	
	cudaMalloc(&cdf_gpu, nbr_bin*(sizeof(int));
	unsigned char* img_out_gput;
	cudaMalloc(&img_out_gpu, img_size*(sizeof(unsigned char));
	// Copy vectors from host memory to device memory
	cudaMemcpy(lut_gpu, lut, nbr_bin*(sizeof(int)), cudaMemcpyHostToDevice);
	cudaMemcpy(cdf_gpu, cdf, nbr_bin*(sizeof(int)), cudaMemcpyHostToDevice);
	// Invoke kernel
	int blocksPerGrid = (img_size + nbr_bin - 1) / nbr_bin;
	construct_lut_gpu<<blocksPerGrid, nbr_bin>>
		(cdf_gpu, lut_gpu, hist_in, nbr_bin, min, d);
	// Copy result from device memory to host memory
	cudaMemcpy(lut, lut_gpu, nbr*(sizeof(int)), cudaMemcpyDeviceToHost);
	
}
