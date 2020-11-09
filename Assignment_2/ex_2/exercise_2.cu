#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define TPB 256
#define ARRAY_SIZE 100000 // default
#define BOUND_RAND 100
#define FLOAT_TH 1e-4

__global__ void saxpyKernel(size_t n, const float a, const float * d_x, float * d_y){
	/* get index */
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i > n) return;
	/* compute */
	d_y[i] += a * d_x[i];
}

void saxpyLauncher(size_t n, const float a, const float * x, float * y){
	/* Copies input arrays on the GPU and calls saxpy kernel */
	float * d_x;
	float * d_y;

	cudaMalloc(&d_x, n*sizeof(float));
	cudaMalloc(&d_y, n*sizeof(float));

	cudaMemcpy(d_x, x, n*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, n*sizeof(float), cudaMemcpyHostToDevice);

	saxpyKernel<<<(n + TPB - 1)/TPB, TPB>>>(n, a, d_x, d_y);
	cudaMemcpy(y, d_y, n*sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_x);
	cudaFree(d_y);
}


void saxpy(size_t n, const float a, const float * x, float * y){
	for(size_t i = 0; i < n; i++){
		y[i] += a*x[i];
	}
}

float randFloat(){
	return ((float)(rand() / (float)RAND_MAX)) * 2 * BOUND_RAND - BOUND_RAND;
}

double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}


int main(int argc, char **argv){
	/* variables */
	int n = (argc > 1) ? strtol(argv[1], NULL, 10) : ARRAY_SIZE;

	float x[n];
	float y[n];
	float y_k[n];
	float a;
	printf("n = %d\n", n);

	/* generate data */
	srand(time(NULL)); // seed
	a = randFloat();
	for(size_t i = 0; i < n; i++){
		x[i] = randFloat(); 
		y[i] = randFloat();
		y_k[i] = y[i];
	}

	/* CPU version */
	printf("Computing SAXPY on the CPU… ");
	double cpu_iStart = cpuSecond();
	saxpy(n, a, x, y);
	double cpu_iElaps = cpuSecond() - cpu_iStart;
	printf("Done! in %f seconds\n", cpu_iElaps);


	/* GPU version */
	printf("Computing SAXPY on the GPU… ");
	double gpu_iStart = cpuSecond();
	saxpyLauncher(n, a, x, y_k);
	cudaDeviceSynchronize();
	double gpu_iElaps = cpuSecond() - gpu_iStart;
	printf("Done! in %f seconds\n", gpu_iElaps);

	/* Compare */
	printf("Comparing the output for each implementation… ");
	float avgE = 0;
	for(size_t i = 0; i < n; i++){
		avgE += fabs((y[i] - y_k[i]) / min(y[i], y_k[i]));
	}
	avgE /= n;
	if(avgE > FLOAT_TH){
		printf("Incorrect!\n");
		return -1;
	}
	printf("Correct!\n");
	return 0;
}