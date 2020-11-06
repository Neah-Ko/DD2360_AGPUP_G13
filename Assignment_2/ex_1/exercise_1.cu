#include <stdio.h>

#define TPB 256
#define NB 1

__global__ void helloKernel(){
	/* not necessary since we have 1 block */
	// const int i = blockIdx.x * blockDim.x + threadIdx.x;
	printf("Hello World! My threadId is %d\n", threadIdx.x);
}


int main(){
	helloKernel<<<NB, TPB>>>();
	cudaDeviceSynchronize();
	return 0;
}