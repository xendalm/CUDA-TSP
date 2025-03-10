#ifndef REDUCTION_H
#define REDUCTION_H

#include <cuda_runtime.h>

__global__ void reduce_min_fitness(float* d_min_fitness_blocks, float* d_final_min_fitness,
								   int overal_size);

#endif