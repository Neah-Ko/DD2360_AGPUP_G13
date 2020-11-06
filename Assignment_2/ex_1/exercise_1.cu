#include <stdio.h>


__global__ void helloKernel(){
	// const int i = blockIdx.x * blockDim.x + threadIdx.x;
	printf("Hello World! My threadId is %d\n", threadIdx.x);
}



int main(){
	helloKernel<<<256, 1>>>();
	cudaDeviceSynchronize();
	return 0;
}