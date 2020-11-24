#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// defaults 
#define NUM_PARTICLES 10000
#define NUM_ITERATION 500
#define BLOCK_SIZE 256
#define BOUND_RAND 100
#define FLOAT_TH 1e-4
#define FDIFF(a, b) fabs((a - b) / min(a, b))


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

void simulationLauncher(size_t n, int m, int tpb, Particle * vec){
	/* Data is copied back and forth at each timestep */
	for(int t = 0; t < m; t++){
		timestepKernel<<<(n + tpb - 1)/tpb, tpb>>>(n, t, vec);
	}
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


float f3fdiff(float3 a, float3 b){
	return (FDIFF(a.x, b.x) + FDIFF(a.y, b.y) + FDIFF(a.z, b.z)) / 3.0;
}

int main(int argc, char **argv){
	/* variables */
	int n = (argc > 1) ? strtol(argv[1], NULL, 10) : NUM_PARTICLES;
	int m = (argc > 2) ? strtol(argv[2], NULL, 10) : NUM_ITERATION;
	int tpb = (argc > 3) ? strtol(argv[3], NULL, 10) : BLOCK_SIZE;

	printf("npart: %d, nit: %d, bsz: %d\n", n, m, tpb);


	Particle * gVec;
	cudaMallocManaged(&gVec, n*sizeof(Particle));

	if(gVec == NULL){
		printf("Particle array allocation failed\n");
		return -1;
	}

	Particle cVec[n];
	
	/* generate data */
	srand(time(NULL)); // seed
	for(int i = 0; i < n; i++){
		gVec[i].position = make_float3(randFloat(), randFloat(), randFloat());
		cVec[i].position = gVec[i].position;
	}

	/* simulate: CPU */
	printf("Simulating on the CPU… ");
	double cpu_iStart = cpuSecond();
	for(int t = 0; t < m; t++){
		timestep(n, t, cVec);
	}
	double cpu_iElaps = cpuSecond() - cpu_iStart;
	printf("Done! in %f seconds\n", cpu_iElaps);

	/* simulate: GPU */
	printf("Simulating on the GPU… ");
	double gpu_iStart = cpuSecond();
	simulationLauncher(n, m, tpb, gVec);
	cudaDeviceSynchronize();
	double gpu_iElaps = cpuSecond() - gpu_iStart;
	printf("Done! in %f seconds\n", gpu_iElaps);

	/* Compare */
	printf("Comparing the output for each implementation… ");
	float avgE = 0;
	for(size_t i = 0; i < n; i++){
		avgE += f3fdiff(cVec[i].position, gVec[i].position);
	}
	avgE /= n;
	if(avgE > FLOAT_TH){
		printf("Incorrect!\n");
		return -1;
	}
	printf("Correct!\n");

	cudaFree(gVec);

	return 0;
}
