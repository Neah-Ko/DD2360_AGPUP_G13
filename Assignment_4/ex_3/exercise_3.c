#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <CL/cl.h>

// defaults 
#define NUM_PARTICLES 10000
#define NUM_ITERATION 500
#define WG_SIZE 128
#define BOUND_RAND 100
#define FLOAT_TH 1e-4
#define MIN(a,b) (((a)<(b))?(a):(b))
#define FDIFF(a, b) fabs((a - b) / MIN(a, b))
// This is a macro for checking the error variable.
#define CHK_ERROR(err) if (err != CL_SUCCESS) fprintf(stderr,"Error: %s\n",clGetErrorString(err));
// A errorCode to string converter (forward declaration)
const char* clGetErrorString(int);

struct Particle {
	float posx, posy, posz;
	float velx, vely, velz;
};

/* We need to define structure in the kernel else it is not seen */
const char * timestepKernel_program =
"__kernel void timestepKernel(int n,           \n" 
"                             int t,           \n"
"                             __global struct Particle * vec){ 	\n"
"	struct Particle {           				\n" 
"		float posx, posy, posz; 				\n"
"		float velx, vely, velz; 				\n"
"	};                          				\n"
"	int i = get_global_id(0);   				\n"
"   if (i > n) return;                          \n"
" 	vec[i].velx = ((t & 0xF435) / 100.0);		\n"
"	vec[i].vely = ((t & 0xF445) / 100.0);		\n"
"	vec[i].velz = ((t & 0xF465) / 100.0);		\n"
"	vec[i].posx = vec[i].posx + vec[i].velx;	\n"
"	vec[i].posy = vec[i].posy + vec[i].vely;	\n"
"	vec[i].posz = vec[i].posz + vec[i].velz;	\n"
"}";

float randFloat(){
	/* This time only positive values */
	return ((float)(rand() / (float)RAND_MAX)) * BOUND_RAND;
}

double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}


void simulationLauncher(size_t n, int wgs, int nts, struct Particle * vec){
    cl_platform_id * platforms;
    cl_uint n_platform;

    // Find OpenCL Platforms
    cl_int err = clGetPlatformIDs(0, NULL, &n_platform);
    CHK_ERROR(err);
    platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id)*n_platform);
    err = clGetPlatformIDs(n_platform, platforms, NULL);
    CHK_ERROR(err);

    // Find and sort devices
    cl_device_id *device_list;
    cl_uint n_devices;
    err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0,NULL, &n_devices);
    CHK_ERROR(err);
    device_list = (cl_device_id *) malloc(sizeof(cl_device_id) * n_devices);
    err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, n_devices, device_list, NULL);
    CHK_ERROR(err);

    // Create and initialize an OpenCL context
    cl_context context = clCreateContext(NULL, n_devices, device_list, NULL, NULL, &err);
    CHK_ERROR(err);

    // Create a command queue
    cl_command_queue cmd_queue = clCreateCommandQueue(context, device_list[0], 0, &err);
    CHK_ERROR(err); 

    /* v Insert your own code here */
    size_t byte_size = n * sizeof(struct Particle); 

    /* Build Kernel */
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&timestepKernel_program, NULL, &err);
    CHK_ERROR(err);
    err = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);
    CHK_ERROR(err);
    if(err != CL_SUCCESS){
    	printf("%d\n", err);
        size_t len;
        char buffer[2048];
        clGetProgramBuildInfo(program, device_list[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        
        fprintf((stderr), "Build error: %s\n", buffer);
        exit(1);
    }
    cl_kernel kernel = clCreateKernel(program, "timestepKernel", &err);
    CHK_ERROR(err);

    /* Declare buffer */
    cl_mem vec_dev = clCreateBuffer(context, CL_MEM_READ_WRITE, byte_size, NULL, &err);
    CHK_ERROR(err);

    /* write in buffer */
    
	double copy_iStart = cpuSecond();
    err = clEnqueueWriteBuffer(cmd_queue, vec_dev, CL_TRUE, 0, byte_size, vec, 0, NULL, NULL);
    CHK_ERROR(err);
	double copy_iElaps = cpuSecond() - copy_iStart;
    printf("\n\tCopy time: %f seconds", copy_iElaps);

    err = clFinish(cmd_queue); CHK_ERROR(err);
    /* set fixed parameters */
    err = clSetKernelArg(kernel, 0, sizeof(cl_uint), &n);
    CHK_ERROR(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &vec_dev);
    CHK_ERROR(err);

    /* set dimensions */
    cl_uint work_dim = 1;
    const size_t workgroup_size = wgs;
	size_t n_workitem = ((n-1)/workgroup_size + 1) * workgroup_size;
    // size_t n_workitem = n;
	// n_workitem += (n_workitem % workgroup_size) > 0 ? (workgroup_size - (n_workitem % workgroup_size)) : 0;
	double kernel_iStart = cpuSecond();
	/* simulate */
    for(cl_uint t = 0; t < nts; t++){

		err = clSetKernelArg(kernel, 1, sizeof(cl_uint), &t);
		CHK_ERROR(err);

		/* launch kernel */
	    err = clEnqueueNDRangeKernel(cmd_queue,         // cl_command_queue command_queue,
	                               kernel,              // cl_kernel kernel
	                               work_dim,            // cl_uint work_dim
	                               NULL,                // const size_t *global_work_offset
	              (const size_t *) &n_workitem,         // const size_t *global_work_size
	              (const size_t *) &workgroup_size,     // const size_t *local_work_size
	                               0,                   // cl_uint num_events_in_wait_list
	                               NULL,                // const cl_event *event_wait_list
	                               NULL);               // cl_event *event

	    CHK_ERROR(err);
    }
	/* Wait for it to finish */
    err = clFinish(cmd_queue); CHK_ERROR(err);
	double kernel_iElaps = cpuSecond() - kernel_iStart;
    printf("\n\tKernel time: %f seconds\n", kernel_iElaps);
    /* copy back on host */
    err = clEnqueueReadBuffer(cmd_queue, vec_dev, CL_TRUE, 0, byte_size, vec, 0, NULL, NULL);
    /* Flush */
    err = clFlush(cmd_queue);  CHK_ERROR(err);
    err = clReleaseKernel(kernel);    CHK_ERROR(err);
    err = clReleaseProgram(program);  CHK_ERROR(err);
    err = clReleaseMemObject(vec_dev);  CHK_ERROR(err);
    
    /* ^ end of own code */

    // Finally, release all that we have allocated.
    err = clReleaseCommandQueue(cmd_queue); CHK_ERROR(err);
    err = clReleaseContext(context);        CHK_ERROR(err);
    free(platforms);
    free(device_list);
}


void timestep(size_t n, int t, struct Particle * vec){
	for(int i = 0; i < n; i++){
		// Weak pseudo-random velocities that depend on time parameter
		vec[i].velx = ((t & 0xF435) / 100.0);
		vec[i].vely = ((t & 0xF445) / 100.0);
		vec[i].velz = ((t & 0xF465) / 100.0);
		// Update pos
		vec[i].posx = vec[i].posx + vec[i].velx;
		vec[i].posy = vec[i].posy + vec[i].vely;
		vec[i].posz = vec[i].posz + vec[i].velz;
	}
}


float f3fdiff(struct Particle * a, struct Particle * b){
	return (FDIFF(a->posx, b->posx) + FDIFF(a->posy, b->posy) + FDIFF(a->posz, b->posz)) / 3.0;
}

int main(int argc, char **argv){
	/* variables */
	int n = (argc > 1) ? strtol(argv[1], NULL, 10) : NUM_PARTICLES;
	int m = (argc > 2) ? strtol(argv[2], NULL, 10) : NUM_ITERATION;
	int wgs = (argc > 3) ? strtol(argv[3], NULL, 10) : WG_SIZE;

	struct Particle gVec[n];
	struct Particle cVec[n];
	
	/* generate data */
	srand(time(NULL)); // seed
	for(int i = 0; i < n; i++){
		gVec[i].posx = randFloat();
		gVec[i].posy = randFloat();
		gVec[i].posz = randFloat();
		cVec[i].posx = gVec[i].posx;
		cVec[i].posy = gVec[i].posy;
		cVec[i].posz = gVec[i].posz;
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
	simulationLauncher(n, wgs, m, gVec);
	double gpu_iElaps = cpuSecond() - gpu_iStart;
	printf("\tDone! in %f seconds\n", gpu_iElaps);

	/* Print */
	// for(int i = 0; i < n; i++){
	// 	printf("(%.2f, %.2f, %.2f)x(%.2f, %.2f, %.2f) - (%.2f, %.2f, %.2f)x(%.2f, %.2f, %.2f)\n",
	// 		gVec[i].velx, gVec[i].vely, gVec[i].velz,
	// 		gVec[i].posx, gVec[i].posy, gVec[i].posz,
	// 		cVec[i].velx, cVec[i].vely, cVec[i].velz,
	// 		cVec[i].posx, cVec[i].posy, cVec[i].posz);
	// }

	/* Compare */
	printf("Comparing the output for each implementation… ");
	float avgE = 0;
	for(size_t i = 0; i < n; i++){
		avgE += f3fdiff(&cVec[i], &gVec[i]);
	}
	avgE /= n;
	if(avgE > FLOAT_TH){
		printf("Incorrect!\n");
		return -1;
	}
	printf("Correct!\n");

	return 0;
}


// The source for this particular version is from: https://stackoverflow.com/questions/24326432/convenient-way-to-show-opencl-error-codes
const char* clGetErrorString(int errorCode) {
  switch (errorCode) {
      case 0: return "CL_SUCCESS";
      case -1: return "CL_DEVICE_NOT_FOUND";
      case -2: return "CL_DEVICE_NOT_AVAILABLE";
      case -3: return "CL_COMPILER_NOT_AVAILABLE";
      case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
      case -5: return "CL_OUT_OF_RESOURCES";
      case -6: return "CL_OUT_OF_HOST_MEMORY";
      case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
      case -8: return "CL_MEM_COPY_OVERLAP";
      case -9: return "CL_IMAGE_FORMAT_MISMATCH";
      case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
      case -12: return "CL_MAP_FAILURE";
      case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
      case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
      case -15: return "CL_COMPILE_PROGRAM_FAILURE";
      case -16: return "CL_LINKER_NOT_AVAILABLE";
      case -17: return "CL_LINK_PROGRAM_FAILURE";
      case -18: return "CL_DEVICE_PARTITION_FAILED";
      case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
      case -30: return "CL_INVALID_VALUE";
      case -31: return "CL_INVALID_DEVICE_TYPE";
      case -32: return "CL_INVALID_PLATFORM";
      case -33: return "CL_INVALID_DEVICE";
      case -34: return "CL_INVALID_CONTEXT";
      case -35: return "CL_INVALID_QUEUE_PROPERTIES";
      case -36: return "CL_INVALID_COMMAND_QUEUE";
      case -37: return "CL_INVALID_HOST_PTR";
      case -38: return "CL_INVALID_MEM_OBJECT";
      case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
      case -40: return "CL_INVALID_IMAGE_SIZE";
      case -41: return "CL_INVALID_SAMPLER";
      case -42: return "CL_INVALID_BINARY";
      case -43: return "CL_INVALID_BUILD_OPTIONS";
      case -44: return "CL_INVALID_PROGRAM";
      case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
      case -46: return "CL_INVALID_KERNEL_NAME";
      case -47: return "CL_INVALID_KERNEL_DEFINITION";
      case -48: return "CL_INVALID_KERNEL";
      case -49: return "CL_INVALID_ARG_INDEX";
      case -50: return "CL_INVALID_ARG_VALUE";
      case -51: return "CL_INVALID_ARG_SIZE";
      case -52: return "CL_INVALID_KERNEL_ARGS";
      case -53: return "CL_INVALID_WORK_DIMENSION";
      case -54: return "CL_INVALID_WORK_GROUP_SIZE";
      case -55: return "CL_INVALID_WORK_ITEM_SIZE";
      case -56: return "CL_INVALID_GLOBAL_OFFSET";
      case -57: return "CL_INVALID_EVENT_WAIT_LIST";
      case -58: return "CL_INVALID_EVENT";
      case -59: return "CL_INVALID_OPERATION";
      case -60: return "CL_INVALID_GL_OBJECT";
      case -61: return "CL_INVALID_BUFFER_SIZE";
      case -62: return "CL_INVALID_MIP_LEVEL";
      case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
      case -64: return "CL_INVALID_PROPERTY";
      case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
      case -66: return "CL_INVALID_COMPILER_OPTIONS";
      case -67: return "CL_INVALID_LINKER_OPTIONS";
      case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";
      case -69: return "CL_INVALID_PIPE_SIZE";
      case -70: return "CL_INVALID_DEVICE_QUEUE";
      case -71: return "CL_INVALID_SPEC_ID";
      case -72: return "CL_MAX_SIZE_RESTRICTION_EXCEEDED";
      case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
      case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
      case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
      case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
      case -1006: return "CL_INVALID_D3D11_DEVICE_KHR";
      case -1007: return "CL_INVALID_D3D11_RESOURCE_KHR";
      case -1008: return "CL_D3D11_RESOURCE_ALREADY_ACQUIRED_KHR";
      case -1009: return "CL_D3D11_RESOURCE_NOT_ACQUIRED_KHR";
      case -1010: return "CL_INVALID_DX9_MEDIA_ADAPTER_KHR";
      case -1011: return "CL_INVALID_DX9_MEDIA_SURFACE_KHR";
      case -1012: return "CL_DX9_MEDIA_SURFACE_ALREADY_ACQUIRED_KHR";
      case -1013: return "CL_DX9_MEDIA_SURFACE_NOT_ACQUIRED_KHR";
      case -1093: return "CL_INVALID_EGL_OBJECT_KHR";
      case -1092: return "CL_EGL_RESOURCE_NOT_ACQUIRED_KHR";
      case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
      case -1057: return "CL_DEVICE_PARTITION_FAILED_EXT";
      case -1058: return "CL_INVALID_PARTITION_COUNT_EXT";
      case -1059: return "CL_INVALID_PARTITION_NAME_EXT";
      case -1094: return "CL_INVALID_ACCELERATOR_INTEL";
      case -1095: return "CL_INVALID_ACCELERATOR_TYPE_INTEL";
      case -1096: return "CL_INVALID_ACCELERATOR_DESCRIPTOR_INTEL";
      case -1097: return "CL_ACCELERATOR_TYPE_NOT_SUPPORTED_INTEL";
      case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
      case -1098: return "CL_INVALID_VA_API_MEDIA_ADAPTER_INTEL";
      case -1099: return "CL_INVALID_VA_API_MEDIA_SURFACE_INTEL";
      case -1100: return "CL_VA_API_MEDIA_SURFACE_ALREADY_ACQUIRED_INTEL";
      case -1101: return "CL_VA_API_MEDIA_SURFACE_NOT_ACQUIRED_INTEL";
      default: return "CL_UNKNOWN_ERROR";
  }
}
