#pragma once
//libs
#include "cuda_runtime.h" 
#include <stdio.h>
#include <iostream> 
#include "device_launch_parameters.h"  
#include <time.h>
#include <math.h>
#include <stdlib.h> 


#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
static void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
	if (result) 
	{
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}

//ray tracing basics
#include "vec3.h"
#include "ray.h"
#include "Triangle.h"
#include "Add Kernel.cuh"

//#include "sphere.cu"
//logging


