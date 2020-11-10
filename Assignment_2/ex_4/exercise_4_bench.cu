#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <curand_kernel.h>
#include <curand.h>

// #define NUM_ITER 1000000000
// #define NUM_ITER 10000000
/* iterations per thread */
#define NUM_ITER_KERNEL 1000000
/* threads in each blocks */
#define TPB 256
/* blocks in the grid */
#define NB 4

double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

__global__ void piKernel(size_t n, int * d_counts, curandState * states, int num_iter_kernel) {
	/* get index */
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	double x;
	double y;
	double z;

	curand_init(i, i, 0, &states[i]);

	// place count on the register to gotta go fast.
	// It may be optimized nevertheless but anyway.
	int count = 0;

	for(int j = 0; j < num_iter_kernel; j++){
		x = curand_uniform(&states[i]);
		y = curand_uniform(&states[i]);
		z = sqrt((x*x) + (y*y));
		if (z <= 1.0)
			count++;
	}
	
	d_counts[i] = count;
}

void piLauncher(size_t n, int * counts, int itr_per_thr, int thr_per_blk) {
	int * d_counts;
	curandState *dev_random;


	cudaMalloc((void**)&dev_random, n*sizeof(curandState));
	cudaMalloc(&d_counts, n*sizeof(int));

	piKernel<<<NB, thr_per_blk>>>(n, d_counts, dev_random, itr_per_thr);

	cudaMemcpy(counts, d_counts, n*sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(dev_random);
	cudaFree(d_counts);
}


int main(int argc, char **argv){
	/* variables */
	int n;
	int count = 0;
	double pi;
	int itr_per_thr;
	int thr_per_blk;

	/* initialize variables from the arguments */
	if (argc >= 2 && argv[1][0] == 'I') // iterations per thread
		itr_per_thr = atoi(argv[1]+1);
	else
		itr_per_thr = NUM_ITER_KERNEL;

        if (argc >= 2 && argv[1][0] == 'T') // threads per block
		thr_per_blk = atoi(argv[1]+1);
	else
		thr_per_blk = TPB;

	/* variables again */
	n = NB * thr_per_blk;
	int counts[n];

	/* Compute pi on the GPU */
	fprintf(stderr, "Estimating pi on the GPU with %d iter per thread and %d thread per block.\n",
			itr_per_thr, thr_per_blk);
	double gpu_iStart = cpuSecond();
	piLauncher(n, counts, itr_per_thr, thr_per_blk);
	cudaDeviceSynchronize();

        double gpu_iElaps = cpuSecond() - gpu_iStart;

	// This could be improved by calculating the sum on the GPU with branching.
	for(int i = 0; i < n; i++){
		count += counts[i];
	}
	pi = (((double)count) / ((double) n) / ((double) NUM_ITER_KERNEL)) * 4.0;

	printf("%d %d %lf %lf\n", itr_per_thr, thr_per_blk, gpu_iElaps, pi);

	//printf("Done! in %f seconds\n", gpu_iElaps);
	//printf("The result is %f\n", pi);

	return 0;
}
