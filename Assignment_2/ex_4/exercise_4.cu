#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <curand_kernel.h>
#include <curand.h>

#define NUM_ITER 1000000000
#define NUM_THREADS 2048
#define TPB 128

double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

__global__ void piKernel(size_t n, int c_num_iter, int * d_counts, curandState * states){
	/* get index */
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i > n) return;

	double x;
	double y;
	double z;
	int c = 0;

    curand_init(i, i, 0, &states[i]);
    for(int j = 0; j < (c_num_iter / NUM_THREADS); j++){
    	x = curand_uniform(&states[i]);
    	y = curand_uniform(&states[i]);
    	z = sqrt((x*x) + (y*y));
    	if (z <= 1.0) {
            c++;
        }
    }
    d_counts[i] = c;
}

void piLauncher(size_t n, int c_num_iter, int c_tpb, int * counts){
	int * d_counts;
	curandState *dev_random;


	cudaMalloc((void**)&dev_random, n*sizeof(curandState));
	cudaMalloc(&d_counts, n*sizeof(int));
	cudaMemset(d_counts, 0, n*sizeof(int));

	piKernel<<<NUM_THREADS/c_tpb, c_tpb>>>(n, c_num_iter, d_counts, dev_random);

	cudaMemcpy(counts, d_counts, n*sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(dev_random);
	cudaFree(d_counts);
}


int main(int argc, char **argv){
	/* variables */
	int c_num_iter = (argc > 1) ? strtol(argv[1], NULL, 10) : NUM_ITER;
	int c_tpb = (argc > 2) ? strtol(argv[2], NULL, 10) : TPB;

	int n = NUM_THREADS;
	int counts[n];
	int count = 0;
	double pi;

	/* Compute pi on the GPU */
	printf("Estimating pi on the GPU… ");
	double gpu_iStart = cpuSecond();
	piLauncher(n, c_num_iter, c_tpb, counts);
	cudaDeviceSynchronize();

	for(int i = 0; i < n; i++){
		count += counts[i];
	}
	pi = ((double)count / (double)c_num_iter) * 4.0;


	double gpu_iElaps = cpuSecond() - gpu_iStart;
	printf("Done! in %f seconds\n", gpu_iElaps);

	printf("The result is %f\n", pi);

	return 0;
}