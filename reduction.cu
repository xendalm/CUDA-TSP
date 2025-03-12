#include "reduction.h"
#include <cuda_runtime.h>

__global__ void reduce_min_fitness(float* d_fitness, float* d_min_fitness, int overal_size) {
	extern __shared__ float shared_data[];

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;

	shared_data[tid] = (idx < overal_size) ? d_fitness[idx] : INFINITY;
	__syncthreads();

	int active_threads = blockDim.x;
	while (active_threads > 1) {
		int half = (active_threads + 1) / 2;

		if (tid < half && tid + half < active_threads) {
			shared_data[tid] = fminf(shared_data[tid], shared_data[tid + half]);
		}
		__syncthreads();

		active_threads = half;
	}

	if (tid == 0) {
		d_min_fitness[blockIdx.x] = shared_data[0];
	}
}

// nvcc -c reduction.cu -o reduction.o
// ar rcs libreduction.a reduction.o
// nvcc tsp.cu -o tsp -L. -lreduction