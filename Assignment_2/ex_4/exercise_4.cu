/*
 * Head comment
 */

#include <stdio.h>

#include <curand_kernel.h>
#include <curand.h>

// Oh boi, I hate those #define preprocessor directives.
#define TPB 1024
#define NB 4
#define ITER_PER_THREAD 1000000



/*
 * This code is not a 1:1 copy of ORNL's (which is linked on the assignment's
 * page. Due to the fact that our code is bound to the GPL licence.
 */

__global__ void count_pi(uint32_t * storage, curandState * rand_states) {

	const int id = threadIdx.x + blockDim.x * blockIdx.x;
	
	// initialize the random state of this very thread.
	curand_init(id, id, 0, rand_states + id);

	uint32_t count = 0;

	for (unsigned int i = 0; i < ITER_PER_THREAD; i++) {
		float x = curand_uniform (&rand_states[id]);
		float y = curand_uniform (&rand_states[id]);

		if (x*x + y*y <= 1)
			count++;
	}
	
	storage[id] = count;
}


int main() {

	// The amount of threads.
	// There is no need to round up with (NB+TPB-1) / TPB here.
	const size_t size = NB * TPB;

	uint32_t *counts = NULL;
	uint32_t *d_counts = NULL;
	cudaMalloc(&d_counts, size * sizeof(uint32_t));

	curandState *dev_random;
	cudaMalloc(&dev_random, size * sizeof(curandState));

	count_pi<<<NB, TPB>>>(d_counts, dev_random);

	counts = (uint32_t*)  malloc(size * sizeof(uint32_t));
	
	cudaDeviceSynchronize();
	// TODO remove that mem copy. The GPU should compute the sum and send one number.
	cudaMemcpy(counts, d_counts, size * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	cudaFree(dev_random);
	cudaFree(d_counts);

	uint32_t total = 0;
	for (int i=0; i<size; i++)
		total += counts[i];

	printf("%lf\n",  4.0 * ((double) total) / ((double) (size*ITER_PER_THREAD) ));

	return 0;
}
