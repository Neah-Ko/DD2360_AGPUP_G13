#include <stdio.h>

#define TPB 256
#define NB 1

__global__ void helloKernel(){
	/* not necessary since we have only 1 block */
	// const int i = blockIdx.x * blockDim.x + threadIdx.x;
	printf("Hello World! My threadId is %d\n", threadIdx.x);
}


int main(){
	/* Launch kernel */
	helloKernel<<<NB, TPB>>>();
	/* To display printf executed by the GPU */
	cudaDeviceSynchronize();
	return 0;
}