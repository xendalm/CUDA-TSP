#include "reduction.h"
#include <cassert>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#define BLOCK_SIZE 256
#define POP_SIZE 10000
#define GENERATIONS 5000
#define MAX_GENERATIONS_WITHOUT_IMPROVEMENT 500
#define MUTATION_RATE 0.1f
#define TOURNAMENT_SIZE 4

struct City {
	float x, y;
};

__device__ __host__ float distance(City a, City b) {
	return sqrtf((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

__global__ void init_rand(curandState* d_states, int seed) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed + idx, idx, 0, &d_states[idx]);
}

__global__ void initialize_population(int* d_population, curandState* d_states, int cities_num) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= POP_SIZE)
		return;

	curandState localState = d_states[idx];
	for (int i = 0; i < cities_num; i++) {
		d_population[idx * cities_num + i] = i;
	}

	for (int i = 0; i < cities_num; i++) {
		int j = curand(&localState) % cities_num;
		int temp = d_population[idx * cities_num + i];
		d_population[idx * cities_num + i] = d_population[idx * cities_num + j];
		d_population[idx * cities_num + j] = temp;
	}
	d_states[idx] = localState;
}

__global__ void evaluate_fitness(City* d_cities, int* d_population, float* d_fitness,
								 int cities_num) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= POP_SIZE)
		return;

	// bool check_used[52] = {false};

	float total_distance = 0;
	for (int i = 0; i < cities_num - 1; i++) {
		// assert(!check_used[d_population[idx * cities_num + i]]);
		// check_used[d_population[idx * cities_num + i]] = true;
		total_distance += distance(d_cities[d_population[idx * cities_num + i]],
								   d_cities[d_population[idx * cities_num + i + 1]]);
	}
	// assert(!check_used[d_population[idx * cities_num + cities_num - 1]]);
	total_distance += distance(d_cities[d_population[idx * cities_num + cities_num - 1]],
							   d_cities[d_population[idx * cities_num]]);
	d_fitness[idx] = total_distance;
}

__global__ void tournament_selection(int* d_population, float* d_fitness, int* d_selected,
									 curandState* d_states, int cities_num) {
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

	for (int i = 0; i < cities_num; i++) {
		d_selected[idx * cities_num + i] = d_population[best * cities_num + i];
	}
}

__global__ void crossover(int* d_population, int* d_selected, int* offspring, char* used,
						  curandState* d_states, int cities_num) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= POP_SIZE / 2)
		return;

	curandState localState = d_states[idx];
	int parent1 = 2 * idx;
	int parent2 = 2 * idx + 1;

	int start = curand(&localState) % cities_num;
	int end = start + (curand(&localState) % (cities_num - start));

	for (int i = 0; i < cities_num; i++) {
		used[parent1 * cities_num + i] = 0;
		used[parent2 * cities_num + i] = 0;
	}

	for (int i = start; i <= end; i++) {
		offspring[parent1 * cities_num + i] = d_selected[parent2 * cities_num + i];
		offspring[parent2 * cities_num + i] = d_selected[parent1 * cities_num + i];
		used[parent1 * cities_num + offspring[parent1 * cities_num + i]] = 1;
		used[parent2 * cities_num + offspring[parent2 * cities_num + i]] = 1;
	}

	int pos1 = 0, pos2 = 0;
	for (int i = 0; i < cities_num; i++) {
		int city = d_selected[parent1 * cities_num + i];
		if (!used[parent1 * cities_num + city]) {
			while (pos1 >= start && pos1 <= end)
				pos1++;
			offspring[parent1 * cities_num + pos1++] = city;
		}

		city = d_selected[parent2 * cities_num + i];
		if (!used[parent2 * cities_num + city]) {
			while (pos2 >= start && pos2 <= end)
				pos2++;
			offspring[parent2 * cities_num + pos2++] = city;
		}
	}

	for (int i = 0; i < cities_num; i++) {
		d_population[parent1 * cities_num + i] = offspring[parent1 * cities_num + i];
		d_population[parent2 * cities_num + i] = offspring[parent2 * cities_num + i];
	}
}

__global__ void mutate(int* d_population, curandState* d_states, int cities_num) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= POP_SIZE)
		return;

	curandState localState = d_states[idx];
	if (curand_uniform(&localState) < MUTATION_RATE) {
		int a = curand(&localState) % cities_num;
		int b = curand(&localState) % cities_num;
		int temp = d_population[idx * cities_num + a];
		d_population[idx * cities_num + a] = d_population[idx * cities_num + b];
		d_population[idx * cities_num + b] = temp;
	}
	d_states[idx] = localState;
}

void tsp_genetic_algorithm_gpu(const std::vector<City>& cities) {
	City* d_cities;
	int *d_population, *d_selected;
	float* d_fitness;
	curandState* d_states;
	int* d_offspring;
	char* d_used;

	int cities_num = cities.size();

	cudaMalloc(&d_cities, cities_num * sizeof(City));
	cudaMalloc(&d_population, POP_SIZE * cities_num * sizeof(int));
	cudaMalloc(&d_selected, POP_SIZE * cities_num * sizeof(int));
	cudaMalloc(&d_fitness, POP_SIZE * sizeof(float));
	cudaMalloc(&d_states, POP_SIZE * sizeof(curandState));
	cudaMalloc(&d_offspring, POP_SIZE * cities_num * sizeof(int));
	cudaMalloc(&d_used, POP_SIZE * cities_num * sizeof(char));

	cudaMemcpy(d_cities, cities.data(), cities_num * sizeof(City), cudaMemcpyHostToDevice);

	int threads_per_block = BLOCK_SIZE;
	int num_blocks = (POP_SIZE + threads_per_block - 1) / threads_per_block;

	init_rand<<<num_blocks, threads_per_block>>>(d_states, time(NULL));
	initialize_population<<<num_blocks, threads_per_block>>>(d_population, d_states, cities_num);

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
		evaluate_fitness<<<num_blocks, threads_per_block>>>(
			d_cities, d_population, d_fitness, cities_num);
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
			d_population, d_fitness, d_selected, d_states, cities_num);
		cudaDeviceSynchronize();
		crossover<<<num_blocks, threads_per_block>>>(
			d_population, d_selected, d_offspring, d_used, d_states, cities_num);
		cudaDeviceSynchronize();
		mutate<<<num_blocks, threads_per_block>>>(d_population, d_states, cities_num);
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
	cudaFree(d_offspring);
	cudaFree(d_used);
}

std::vector<City> load_cities(const std::string& filename) {
	std::vector<City> cities;

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

	return cities;
}

int main(int argc, char* argv[]) {
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " <tsp_data_file>" << std::endl;
		return 1;
	}

	std::vector<City> cities = load_cities(argv[1]);

	tsp_genetic_algorithm_gpu(cities);

	return 0;
}
