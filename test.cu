using namespace std;
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


__global__ void is_odd( int * d_in){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	printf("hello from thread %d, data is %d\n", idx, d_in[idx]);
}
 

int main(int argc, char ** argv) {

  int nDevices;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf(" Memory Global Mem: %d\n", prop.totalGlobalMem);
    printf(" Memory shared per block: %d\n", prop.sharedMemPerBlock);
    printf(" can map host memory: %d\n", prop.canMapHostMemory);
    printf(" device overlap: %d\n", prop.deviceOverlap);
    printf(" max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  }
printf("entering\n");

	int  input1[1024];
        //input1 =  (int*) malloc(sizeof(int)*1024);
	int  input2[1024];
	//input2 = (int*) malloc(sizeof(int)*1024);

	int output_a[1024];// = malloc(sizeof(int)*2048);
	int output_b[1024];
	int * d_in;
	int * d_out;

	cudaMalloc((void**) &d_in, sizeof(int)*2048);
	cudaMalloc((void**) &d_out, sizeof(int)*2048);

	for(int i=0; i< 1024; i++){
		input1[i] = i*2;
		input2[i] = i*2+1;
	}

	cudaMemcpy2D(d_in, 2*sizeof(int), input1, sizeof(int), sizeof(int), 1024, cudaMemcpyHostToDevice);
	cudaMemcpy2D(d_in + 1, 2*sizeof(int), input2, sizeof(int), sizeof(int), 1024, cudaMemcpyHostToDevice);
	is_odd<<<2, 1024>>>(d_in);
	cudaDeviceSynchronize();
	cudaMemcpy2D(output_a, sizeof(int), d_in, 2*sizeof(int), sizeof(int), 1024, cudaMemcpyDeviceToHost);
	cudaMemcpy2D(output_b, sizeof(int), d_in+1, 2*sizeof(int), sizeof(int), 1024, cudaMemcpyDeviceToHost);
	printf("index at 10: a is %d, b is %d\n", output_a[10], output_b[10]);
	cudaFree(d_in);
	cudaFree(d_out);
	return 0;
}
