import math
import sys
import time
import warnings

import numpy as np
from numba import cuda, int32
from numba.core.errors import NumbaPerformanceWarning
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

warnings.simplefilter("ignore", category=NumbaPerformanceWarning)

BLOCK_SIZE = 512
POP_SIZE = 50000
GENERATIONS = 500
MAX_GENERATIONS_WITHOUT_IMPROVEMENT = 500
MUTATION_RATE = 0.1
TOURNAMENT_SIZE = 4


def load_cities(filename):
    cities = []
    with open(filename, 'r') as f:
        for line in f:
            _, x, y = line.strip().split()
            cities.append((x, y))
    return cities


@cuda.jit(device=True)
def cuda_distance(city1_x, city1_y, city2_x, city2_y):
    return math.sqrt((city1_x - city2_x) ** 2 + (city1_y - city2_y) ** 2)


@cuda.jit
def initialize_population_kernel(population, rng_states, cities_num):
    idx = cuda.grid(1)
    if idx >= POP_SIZE:
        return

    for i in range(cities_num):
        population[idx, i] = i

    for i in range(cities_num):
        j = int(xoroshiro128p_uniform_float32(rng_states, idx) * cities_num) % cities_num
        temp = population[idx, i]
        population[idx, i] = population[idx, j]
        population[idx, j] = temp


@cuda.jit
def evaluate_fitness_kernel(population, cities_x, cities_y, fitness, cities_num):
    idx = cuda.grid(1)
    if idx >= POP_SIZE:
        return

    total_distance = 0.0
    for i in range(cities_num - 1):
        city1_idx = population[idx, i]
        city2_idx = population[idx, i + 1]
        total_distance += cuda_distance(cities_x[city1_idx], cities_y[city1_idx],
                                        cities_x[city2_idx], cities_y[city2_idx])

    city1_idx = population[idx, cities_num - 1]
    city2_idx = population[idx, 0]
    total_distance += cuda_distance(cities_x[city1_idx], cities_y[city1_idx],
                                    cities_x[city2_idx], cities_y[city2_idx])
    fitness[idx] = total_distance


@cuda.jit
def tournament_selection_kernel(population, fitness, selected, rng_states, cities_num):
    idx = cuda.grid(1)
    if idx >= POP_SIZE:
        return

    best_candidate = int(xoroshiro128p_uniform_float32(rng_states, idx) * POP_SIZE) % POP_SIZE

    for i in range(1, TOURNAMENT_SIZE):
        candidate = int(xoroshiro128p_uniform_float32(rng_states, idx) * POP_SIZE) % POP_SIZE
        if fitness[candidate] < fitness[best_candidate]:
            best_candidate = candidate

    for i in range(cities_num):
        selected[idx, i] = population[best_candidate, i]


@cuda.jit
def crossover_kernel(population, selected, offspring, used, rng_states, cities_num):
    idx = cuda.grid(1)
    if idx >= POP_SIZE // 2:
        return

    parent1_idx = 2 * idx
    parent2_idx = 2 * idx + 1

    start = int(xoroshiro128p_uniform_float32(rng_states, idx) * cities_num) % cities_num
    end = start + int(xoroshiro128p_uniform_float32(rng_states, idx) * (cities_num - start))

    for i in range(cities_num):
        used[parent1_idx, i] = 0
        used[parent2_idx, i] = 0

    for i in range(0, cities_num):
        offspring[parent1_idx, i] = selected[parent2_idx, i]
        offspring[parent2_idx, i] = selected[parent1_idx, i]
        used[parent1_idx, offspring[parent1_idx, i]] = 1
        used[parent2_idx, offspring[parent2_idx, i]] = 1

    pos1 = 0
    pos2 = 0
    for i in range(cities_num):
        city = selected[parent1_idx, i]
        if not used[parent1_idx, city]:
            while start <= pos1 <= end:
                pos1 += 1
            offspring[parent1_idx, pos1] = city
            pos1 += 1

        city = selected[parent2_idx, i]
        if not used[parent2_idx, city]:
            while start <= pos2 <= end:
                pos2 += 1
            offspring[parent2_idx, pos2] = city
            pos2 += 1

    for i in range(cities_num):
        population[parent1_idx, i] = offspring[parent1_idx, i]
        population[parent2_idx, i] = offspring[parent2_idx, i]


@cuda.jit
def mutate_kernel(population, rng_states, cities_num):
    idx = cuda.grid(1)
    if idx >= POP_SIZE:
        return

    if cuda.random.xoroshiro128p_uniform_float32(rng_states, idx) < MUTATION_RATE:
        a = int(xoroshiro128p_uniform_float32(rng_states, idx) * cities_num) % cities_num
        b = int(xoroshiro128p_uniform_float32(rng_states, idx) * cities_num) % cities_num
        temp = population[idx, a]
        population[idx, a] = population[idx, b]
        population[idx, b] = temp


def tsp_genetic_algorithm_cuda(cities):
    cities_x = np.array([city[0] for city in cities], dtype=np.float32)
    cities_y = np.array([city[1] for city in cities], dtype=np.float32)

    d_cities_x = cuda.to_device(cities_x)
    d_cities_y = cuda.to_device(cities_y)

    cities_num = len(cities)

    d_population = cuda.to_device(np.zeros((POP_SIZE, cities_num), dtype=np.int32))
    d_selected = cuda.to_device(np.zeros((POP_SIZE, cities_num), dtype=np.int32))
    d_fitness = cuda.to_device(np.zeros(POP_SIZE, dtype=np.float32))
    d_offspring = cuda.to_device(np.zeros((POP_SIZE, cities_num), dtype=np.int32))
    d_used = cuda.to_device(np.zeros((POP_SIZE, cities_num), dtype=np.int32))

    threads_per_block = BLOCK_SIZE
    blocks_per_grid = (POP_SIZE + threads_per_block - 1) // threads_per_block
    print(f"Blocks per grid: {blocks_per_grid}, Threads per block: {threads_per_block}")

    rng_states = create_xoroshiro128p_states(POP_SIZE, seed=int(time.time()))
    initialize_population_kernel[blocks_per_grid, threads_per_block](d_population, rng_states, cities_num)

    best_fitness = float('inf')
    generations_wout_improvement = 0

    start_time = time.time()

    for gen in range(GENERATIONS):
        evaluate_fitness_kernel[blocks_per_grid, threads_per_block](d_population, d_cities_x, d_cities_y, d_fitness, cities_num)
        cuda.synchronize()

        h_fitness = d_fitness.copy_to_host()
        gen_best_fitness = np.min(h_fitness)

        if gen_best_fitness < best_fitness:
            print(f"Generation {gen}: best fitness = {gen_best_fitness}")
            best_fitness = gen_best_fitness
            generations_wout_improvement = 0
        else:
            generations_wout_improvement += 1
            if generations_wout_improvement == MAX_GENERATIONS_WITHOUT_IMPROVEMENT:
                print(
                    f"Generation {gen}: no improvement for {MAX_GENERATIONS_WITHOUT_IMPROVEMENT} generations. Stopping.")
                break

        tournament_selection_kernel[blocks_per_grid, threads_per_block](d_population, d_fitness, d_selected, rng_states, cities_num)
        cuda.synchronize()

        crossover_kernel[blocks_per_grid, threads_per_block](d_population, d_selected, d_offspring, d_used, rng_states, cities_num)
        cuda.synchronize()

        mutate_kernel[blocks_per_grid, threads_per_block](d_population, rng_states, cities_num)
        cuda.synchronize()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time: {:.4f} seconds".format(elapsed_time))
    print("Final best fitness:", best_fitness)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tsp_ga_cuda.py <tsp_data_file>")
        sys.exit(1)

    city_filename = sys.argv[1]
    cities = load_cities(city_filename)

    tsp_genetic_algorithm_cuda(cities)
