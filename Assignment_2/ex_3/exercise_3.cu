#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// defaults 
#define NUM_PARTICLES 1
#define NUM_ITERATION 1000
#define BLOCK_SIZE 256
#define BOUND_RAND 100



typedef struct Particle {
	float3 position;
	float3 velocity;
} Particle;

float randFloat(){
	/* This time only positive values */
	return ((float)(rand() / (float)RAND_MAX)) * BOUND_RAND;
}

double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

__global__ void timestepKernel(size_t n, int t, Particle * d_vec){
	/* get index */
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i > n) return;

	/* compute */
	d_vec[i].velocity = make_float3(
		((t & 0xF435) / 100.0),
		((t & 0xF445) / 100.0),
		((t & 0xF465) / 100.0));
	d_vec[i].position = make_float3(
		d_vec[i].position.x + d_vec[i].velocity.x,
		d_vec[i].position.y + d_vec[i].velocity.y,
		d_vec[i].position.z + d_vec[i].velocity.z
	);
}

void simulationLauncher(size_t n, Particle * vec){
	Particle * d_vec;

	cudaMalloc(&d_vec, n*sizeof(Particle));


	cudaMemcpy(d_vec, vec, n*sizeof(Particle), cudaMemcpyHostToDevice);


	for(int t = 0; t < NUM_ITERATION; t++){
		timestepKernel<<<(n + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE>>>(n, t, d_vec);
	}

	cudaMemcpy(vec, d_vec, n*sizeof(Particle), cudaMemcpyDeviceToHost);

	cudaFree(d_vec);
}

void timestep(size_t n, int t, Particle * vec){
	for(int i = 0; i < n; i++){
		// Weak pseudo-random velocities that depend on time parameter
		vec[i].velocity = make_float3(
			((t & 0xF435) / 100.0),
			((t & 0xF445) / 100.0),
			((t & 0xF465) / 100.0));
		// Update pos
		vec[i].position = make_float3(
			vec[i].position.x + vec[i].velocity.x,
			vec[i].position.y + vec[i].velocity.y,
			vec[i].position.z + vec[i].velocity.z
		);
	}
}

int main(){
	/* variables */
	int n = NUM_PARTICLES;
	Particle gVec[n];
	Particle cVec[n];
	
	/* generate data */
	srand(time(NULL)); // seed
	for(int i = 0; i < n; i++){
		gVec[i].position = make_float3(randFloat(), randFloat(), randFloat());
		cVec[i].position = gVec[i].position;
	}

	// /* print */
	// for(int i = 0; i < n; i ++){
	// 	printf("%d: (%f, %f, %f) - (%f, %f, %f)\n",
	// 		i,
	// 		cVec[i].position.x,
	// 		cVec[i].position.y,
	// 		cVec[i].position.z,
	// 		cVec[i].velocity.x,
	// 		cVec[i].velocity.y,
	// 		cVec[i].velocity.z
	// 	);
	// }

	/* simulate: CPU */
	printf("Simulating on the CPU… ");
	double cpu_iStart = cpuSecond();
	for(int t = 0; t < NUM_ITERATION; t++){
		timestep(n, t, cVec);
	}
	double cpu_iElaps = cpuSecond() - cpu_iStart;
	printf("Done! in %f seconds\n", cpu_iElaps);

	/* simulate: GPU */
	printf("Simulating on the GPU… ");
	double gpu_iStart = cpuSecond();
	simulationLauncher(n,  gVec);
	cudaDeviceSynchronize();
	double gpu_iElaps = cpuSecond() - gpu_iStart;
	printf("Done! in %f seconds\n", gpu_iElaps);


	/* Compare */
	// printf("Comparing the output for each implementation… ");
	// for(size_t i = 0; i < ARRAY_SIZE; i++){
	// 	if((y[i] - y_k[i]) > FLOAT_TH) {
	// 		printf("Incorrect!\n");
	// 		return -1;
	// 	}
	// }
	// printf("Correct!\n");
	return 0;
}