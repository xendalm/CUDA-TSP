#include "reduction.h"
#include <cassert>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#define CITIES 52
#define BLOCK_SIZE 256
#define POP_SIZE 10000
#define GENERATIONS 5000
#define MAX_GENERATIONS_WITHOUT_IMPROVEMENT 500
#define MUTATION_RATE 0.1f
#define TOURNAMENT_SIZE 4

struct City {
	float x, y;
};

std::vector<City> cities;

__device__ __host__ float distance(City a, City b) {
	return sqrtf((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

__global__ void init_rand(curandState* d_states, int seed) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed + idx, idx, 0, &d_states[idx]);
}

__global__ void initialize_population(int* d_population, curandState* d_states) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= POP_SIZE)
		return;

	curandState localState = d_states[idx];
	for (int i = 0; i < CITIES; i++) {
		d_population[idx * CITIES + i] = i;
	}

	for (int i = 0; i < CITIES; i++) {
		int j = curand(&localState) % CITIES;
		int temp = d_population[idx * CITIES + i];
		d_population[idx * CITIES + i] = d_population[idx * CITIES + j];
		d_population[idx * CITIES + j] = temp;
	}
	d_states[idx] = localState;
}

__global__ void evaluate_fitness(City* d_cities, int* d_population, float* d_fitness) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= POP_SIZE)
		return;

	bool check_used[CITIES] = {false};

	float total_distance = 0;
	for (int i = 0; i < CITIES - 1; i++) {
		assert(!check_used[d_population[idx * CITIES + i]]);
		check_used[d_population[idx * CITIES + i]] = true;
		total_distance += distance(d_cities[d_population[idx * CITIES + i]],
								   d_cities[d_population[idx * CITIES + i + 1]]);
	}
	assert(!check_used[d_population[idx * CITIES + CITIES - 1]]);
	total_distance += distance(d_cities[d_population[idx * CITIES + CITIES - 1]],
							   d_cities[d_population[idx * CITIES]]);
	d_fitness[idx] = total_distance;
}

__global__ void tournament_selection(int* d_population, float* d_fitness, int* d_selected,
									 curandState* d_states) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= POP_SIZE)
		return;

	curandState localState = d_states[idx];
	int best = curand(&localState) % POP_SIZE;
	for (int i = 1; i < TOURNAMENT_SIZE; i++) {
		int candidate = curand(&localState) % POP_SIZE;
		if (d_fitness[candidate] < d_fitness[best]) {
			best = candidate;
		}
	}

	for (int i = 0; i < CITIES; i++) {
		d_selected[idx * CITIES + i] = d_population[best * CITIES + i];
	}
}

__global__ void crossover(int* d_population, int* d_selected, curandState* d_states) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= POP_SIZE / 2)
		return;

	curandState localState = d_states[idx];
	int parent1 = 2 * idx;
	int parent2 = 2 * idx + 1;

	int start = curand(&localState) % CITIES;
	int end = start + (curand(&localState) % (CITIES - start));

	int offspring1[CITIES], offspring2[CITIES];
	bool used1[CITIES] = {false}, used2[CITIES] = {false};

	for (int i = start; i <= end; i++) {
		offspring1[i] = d_selected[parent2 * CITIES + i];
		offspring2[i] = d_selected[parent1 * CITIES + i];
		used1[offspring1[i]] = true;
		used2[offspring2[i]] = true;
	}

	int pos1 = 0, pos2 = 0;
	for (int i = 0; i < CITIES; i++) {
		int city = d_selected[parent1 * CITIES + i];
		if (!used1[city]) {
			while (pos1 >= start && pos1 <= end)
				pos1++;
			offspring1[pos1++] = city;
		}

		city = d_selected[parent2 * CITIES + i];
		if (!used2[city]) {
			while (pos2 >= start && pos2 <= end)
				pos2++;
			offspring2[pos2++] = city;
		}
	}

	for (int i = 0; i < CITIES; i++) {
		d_population[parent1 * CITIES + i] = offspring1[i];
		d_population[parent2 * CITIES + i] = offspring2[i];
	}
}

__global__ void mutate(int* d_population, curandState* d_states) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= POP_SIZE)
		return;

	curandState localState = d_states[idx];
	if (curand_uniform(&localState) < MUTATION_RATE) {
		int a = curand(&localState) % CITIES;
		int b = curand(&localState) % CITIES;
		int temp = d_population[idx * CITIES + a];
		d_population[idx * CITIES + a] = d_population[idx * CITIES + b];
		d_population[idx * CITIES + b] = temp;
	}
	d_states[idx] = localState;
}

void tsp_genetic_algorithm_gpu() {
	City* d_cities;
	int *d_population, *d_selected;
	float* d_fitness;
	curandState* d_states;

	cudaMalloc(&d_cities, CITIES * sizeof(City));
	cudaMalloc(&d_population, POP_SIZE * CITIES * sizeof(int));
	cudaMalloc(&d_selected, POP_SIZE * CITIES * sizeof(int));
	cudaMalloc(&d_fitness, POP_SIZE * sizeof(float));
	cudaMalloc(&d_states, POP_SIZE * sizeof(curandState));

	cudaMemcpy(d_cities, cities.data(), CITIES * sizeof(City), cudaMemcpyHostToDevice);

	int threads_per_block = BLOCK_SIZE;
	int num_blocks = (POP_SIZE + threads_per_block - 1) / threads_per_block;

	init_rand<<<num_blocks, threads_per_block>>>(d_states, time(NULL));
	initialize_population<<<num_blocks, threads_per_block>>>(d_population, d_states);

	float* d_min_fitness_blocks;
	float* d_final_min_fitness;
	cudaMalloc(&d_min_fitness_blocks, num_blocks * sizeof(float));
	cudaMalloc(&d_final_min_fitness, sizeof(float));

	float best_fitness = INFINITY;
	float gen_best_fitness;
	int generations_wout_improvement = 0;

	cudaEvent_t event_start, event_stop;
	cudaEventCreate(&event_start);
	cudaEventCreate(&event_stop);
	cudaEventRecord(event_start, 0);
	// GA
	for (int gen = 0; gen < GENERATIONS; gen++) {
		evaluate_fitness<<<num_blocks, threads_per_block>>>(d_cities, d_population, d_fitness);
		cudaDeviceSynchronize();

		reduce_min_fitness<<<num_blocks, threads_per_block, threads_per_block * sizeof(float)>>>(
			d_fitness, d_min_fitness_blocks, POP_SIZE);
		cudaDeviceSynchronize();

		reduce_min_fitness<<<1, num_blocks, num_blocks * sizeof(float)>>>(
			d_min_fitness_blocks, d_final_min_fitness, num_blocks);
		cudaDeviceSynchronize();

		cudaMemcpy(&gen_best_fitness, d_final_min_fitness, sizeof(float), cudaMemcpyDeviceToHost);

		if (gen_best_fitness < best_fitness) {
			printf("Generation %d: best fitness = %f\n", gen, gen_best_fitness);
			best_fitness = gen_best_fitness;
			generations_wout_improvement = 0;
		} else {
			generations_wout_improvement++;
			if (generations_wout_improvement == MAX_GENERATIONS_WITHOUT_IMPROVEMENT) {
				printf("Generation %d: no improvement for %d generations. Stopping.\n",
					   gen,
					   MAX_GENERATIONS_WITHOUT_IMPROVEMENT);
				break;
			}
		}

		tournament_selection<<<num_blocks, threads_per_block>>>(
			d_population, d_fitness, d_selected, d_states);
		cudaDeviceSynchronize();
		crossover<<<num_blocks, threads_per_block>>>(d_population, d_selected, d_states);
		cudaDeviceSynchronize();
		mutate<<<num_blocks, threads_per_block>>>(d_population, d_states);
		cudaDeviceSynchronize();
	}

	cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, event_start, event_stop);
	printf("Time: %f ms\n", milliseconds);

	cudaFree(d_cities);
	cudaFree(d_population);
	cudaFree(d_selected);
	cudaFree(d_fitness);
	cudaFree(d_states);
}

void load_cities(const std::string& filename) {
	std::ifstream file(filename);
	std::string line;
	while (std::getline(file, line)) {
		std::istringstream iss(line);
		int id;
		City city;
		if (iss >> id >> city.x >> city.y) {
			cities.push_back(city);
		}
	}
}

int main(int argc, char* argv[]) {
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " <tsp_data_file>" << std::endl;
		return 1;
	}

	load_cities(argv[1]);

	tsp_genetic_algorithm_gpu();

	return 0;
}
