#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.cuh"

// Device code
__global__ void construct_histogram_gpu(int * hist_out, unsigned char * img_in, int * img_size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	//__shared__ int s_hist_out[256];
	if (i < *img_size) {
	//s_hist_out[img_in[i]] = hist_out[img_in[i]];
	//__syncthreads();
	atomicAdd(&hist_out[img_in[i]], 1);
	//s_hist_out[img_in[i]] = 0;
	//hist_out[img_in[i]] = s_hist_out[img_in[i]];
	}
}
// Host code
void histogram_gpu(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin) {
	int i;
	for(i = 0; i < nbr_bin; i++) {
		hist_out[i] = 0;
	}
	// Allocate vectors in device memory
	int* hist_out_gpu;
	cudaMalloc(&hist_out_gpu, nbr_bin*(sizeof(int)));	
	unsigned char* img_in_gpu;
	cudaMalloc(&img_in_gpu, img_size*(sizeof(unsigned char)));
	int* img_size_gpu;
	cudaMalloc(&img_size_gpu, sizeof(int));
	// Copy vectors from host memory to device memory
	cudaMemcpy(hist_out_gpu, hist_out, nbr_bin*(sizeof(int)), cudaMemcpyHostToDevice);
	cudaMemcpy(img_in_gpu, img_in, img_size*(sizeof(unsigned char)), cudaMemcpyHostToDevice);
	cudaMemcpy(img_size_gpu, &img_size, sizeof(int), cudaMemcpyHostToDevice);
	// Invoke kernel
	int blocksPerGrid = (img_size + nbr_bin - 1) / nbr_bin;
	construct_histogram_gpu<<<blocksPerGrid, nbr_bin>>>
		(hist_out_gpu, img_in_gpu, img_size_gpu);
	// Copy result from device memory to host memory
	cudaMemcpy(hist_out, hist_out_gpu, nbr_bin*(sizeof(int)), cudaMemcpyDeviceToHost);
	// Free device memory
	cudaFree(hist_out_gpu);
	cudaFree(img_in_gpu);
	cudaFree(img_size_gpu);
	/*printf("\n\n");
	for(i = 0; i < nbr_bin; i++) {
		printf("%d ", hist_out[i]);
	}*/
}
// Device code
__global__ void construct_lut_gpu(int * cdf, int * lut, int * hist_in, 
				  int * nbr_bin, int * min, int * d) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < *nbr_bin) {
		lut[i] = (int)(((float)(cdf[i]) - (*min))*255/(*d) + 0.5);
		if(lut[i] < 0) {
			lut[i] = 0;
		}	
	}
}
// Device code
__global__ void get_result_image_gpu(int * lut, unsigned char * img_out,
				     unsigned char * img_in, int * img_size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < *img_size) {
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
	cdf[0] = hist_in[0];
	for(i = 1; i < nbr_bin; i++) {
		cdf[i] = cdf[i-1] + hist_in[i];
	}
	d = img_size - min;
	// Allocate vectors in device memory
	int* lut_gpu;
	cudaMalloc(&lut_gpu, nbr_bin*(sizeof(int)));
	int* cdf_gpu;	
	cudaMalloc(&cdf_gpu, nbr_bin*(sizeof(int)));
	unsigned char* img_out_gpu;
	cudaMalloc(&img_out_gpu, img_size*(sizeof(unsigned char)));
	unsigned char* img_in_gpu;
	cudaMalloc(&img_in_gpu, img_size*(sizeof(unsigned char)));
	int* hist_in_gpu;
	cudaMalloc(&hist_in_gpu, nbr_bin*(sizeof(int)));
	int* img_size_gpu;
	cudaMalloc(&img_size_gpu, sizeof(int));
	int* nbr_bin_gpu;
	cudaMalloc(&nbr_bin_gpu, sizeof(int));
	int* min_gpu;
	cudaMalloc(&min_gpu, sizeof(int));
	int* d_gpu;
	cudaMalloc(&d_gpu, sizeof(int));
	// Copy vectors from host memory to device memory
	cudaMemcpy(lut_gpu, lut, nbr_bin*(sizeof(int)), cudaMemcpyHostToDevice);	
	cudaMemcpy(cdf_gpu, cdf, nbr_bin*(sizeof(int)), cudaMemcpyHostToDevice);
	cudaMemcpy(img_out_gpu, img_out, img_size*(sizeof(unsigned char)), cudaMemcpyHostToDevice);
	cudaMemcpy(img_in_gpu, img_in, img_size*(sizeof(unsigned char)), cudaMemcpyHostToDevice);
	cudaMemcpy(img_size_gpu, &img_size, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(nbr_bin_gpu, &nbr_bin, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(min_gpu, &min, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_gpu, &d, sizeof(int), cudaMemcpyHostToDevice);
	// Invoke kernel
	int blocksPerGrid = (img_size + nbr_bin - 1) / nbr_bin;
	construct_lut_gpu<<<blocksPerGrid, nbr_bin>>>
		(cdf_gpu, lut_gpu, hist_in_gpu, nbr_bin_gpu, min_gpu, d_gpu);
	// Copy result from device memory to host memory
	cudaMemcpy(lut, lut_gpu, nbr_bin*(sizeof(int)), cudaMemcpyDeviceToHost);
	// Copy vectors from host memory to device memory
	cudaMemcpy(lut_gpu, lut, nbr_bin*(sizeof(int)), cudaMemcpyHostToDevice);
	// Invoke kernel
	get_result_image_gpu<<<blocksPerGrid, nbr_bin>>>
		(lut_gpu, img_out_gpu, img_in_gpu, img_size_gpu);
	// Copy vectors from device memory to host memory
	cudaMemcpy(img_out, img_out_gpu, img_size*(sizeof(unsigned char)), cudaMemcpyDeviceToHost);
	// Free device memory
	cudaFree(lut_gpu);
	cudaFree(cdf_gpu);
	cudaFree(img_out_gpu);
	cudaFree(img_in_gpu);
	cudaFree(hist_in_gpu);
	cudaFree(img_size_gpu);
	cudaFree(nbr_bin_gpu);
	cudaFree(min_gpu);
	cudaFree(d_gpu);	
}
