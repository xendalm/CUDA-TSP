import math
import sys
import time
import warnings

import numpy as np
from numba import cuda, int32
from numba.core.errors import NumbaPerformanceWarning
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

warnings.simplefilter("ignore", category=NumbaPerformanceWarning)

CITIES = 52
BLOCK_SIZE = 512
POP_SIZE = 5000
GENERATIONS = 700
MAX_GENERATIONS_WITHOUT_IMPROVEMENT = 500
MUTATION_RATE = 0.1
TOURNAMENT_SIZE = 4


def load_cities(filename):
    cities = []
    with open(filename, 'r') as f:
        for line in f:
            id, x, y = line.strip().split()
            cities.append((x, y))
    return cities


@cuda.jit(device=True)
def cuda_distance(city1_x, city1_y, city2_x, city2_y):
    return math.sqrt((city1_x - city2_x) ** 2 + (city1_y - city2_y) ** 2)


@cuda.jit
def initialize_population_kernel(population, rng_states):
    idx = cuda.grid(1)
    if idx >= POP_SIZE:
        return

    for i in range(CITIES):
        population[idx, i] = i

    for i in range(CITIES):
        j = int(xoroshiro128p_uniform_float32(rng_states, idx) * CITIES) % CITIES
        temp = population[idx, i]
        population[idx, i] = population[idx, j]
        population[idx, j] = temp


@cuda.jit
def evaluate_fitness_kernel(population, cities_x, cities_y, fitness):
    idx = cuda.grid(1)
    if idx >= POP_SIZE:
        return

    total_distance = 0.0
    for i in range(CITIES - 1):
        city1_idx = population[idx, i]
        city2_idx = population[idx, i + 1]
        total_distance += cuda_distance(cities_x[city1_idx], cities_y[city1_idx],
                                        cities_x[city2_idx], cities_y[city2_idx])

    city1_idx = population[idx, CITIES - 1]
    city2_idx = population[idx, 0]
    total_distance += cuda_distance(cities_x[city1_idx], cities_y[city1_idx],
                                    cities_x[city2_idx], cities_y[city2_idx])
    fitness[idx] = total_distance


@cuda.jit
def tournament_selection_kernel(population, fitness, selected, rng_states):
    idx = cuda.grid(1)
    if idx >= POP_SIZE:
        return

    best_candidate = int(xoroshiro128p_uniform_float32(rng_states, idx) * POP_SIZE) % POP_SIZE

    for i in range(1, TOURNAMENT_SIZE):
        candidate = int(xoroshiro128p_uniform_float32(rng_states, idx) * POP_SIZE) % POP_SIZE
        if fitness[candidate] < fitness[best_candidate]:
            best_candidate = candidate

    for i in range(CITIES):
        selected[idx, i] = population[best_candidate, i]


@cuda.jit
def crossover_kernel(population, selected, rng_states):
    idx = cuda.grid(1)
    if idx >= POP_SIZE // 2:
        return

    parent1_idx = 2 * idx
    parent2_idx = 2 * idx + 1

    start = int(xoroshiro128p_uniform_float32(rng_states, idx) * CITIES) % CITIES
    end = start + int(xoroshiro128p_uniform_float32(rng_states, idx) * (CITIES - start))

    offspring1 = cuda.local.array(shape=(CITIES,), dtype=int32)
    offspring2 = cuda.local.array(shape=(CITIES,), dtype=int32)
    used1 = cuda.local.array(shape=(CITIES,), dtype=int32)
    used2 = cuda.local.array(shape=(CITIES,), dtype=int32)

    for i in range(CITIES):
        used1[i] = 0
        used2[i] = 0

    for i in range(0, CITIES):
        offspring1[i] = selected[parent2_idx, i]
        offspring2[i] = selected[parent1_idx, i]
        used1[offspring1[i]] = 1
        used2[offspring2[i]] = 1

    pos1 = 0
    pos2 = 0
    for i in range(CITIES):
        city = selected[parent1_idx, i]
        if not used1[city]:
            while start <= pos1 <= end:
                pos1 += 1
            offspring1[pos1] = city
            pos1 += 1

        city = selected[parent2_idx, i]
        if not used2[city]:
            while start <= pos2 <= end:
                pos2 += 1
            offspring2[pos2] = city
            pos2 += 1

    for i in range(CITIES):
        population[parent1_idx, i] = offspring1[i]
        population[parent2_idx, i] = offspring2[i]


@cuda.jit
def mutate_kernel(population, rng_states):
    idx = cuda.grid(1)
    if idx >= POP_SIZE:
        return

    if cuda.random.xoroshiro128p_uniform_float32(rng_states, idx) < MUTATION_RATE:
        a = int(xoroshiro128p_uniform_float32(rng_states, idx) * CITIES) % CITIES
        b = int(xoroshiro128p_uniform_float32(rng_states, idx) * CITIES) % CITIES
        temp = population[idx, a]
        population[idx, a] = population[idx, b]
        population[idx, b] = temp


def tsp_genetic_algorithm_cuda(cities):
    cities_x = np.array([city[0] for city in cities], dtype=np.float32)
    cities_y = np.array([city[1] for city in cities], dtype=np.float32)

    d_cities_x = cuda.to_device(cities_x)
    d_cities_y = cuda.to_device(cities_y)

    population = np.zeros((POP_SIZE, CITIES), dtype=np.int32)
    selected = np.zeros((POP_SIZE, CITIES), dtype=np.int32)
    fitness = np.zeros(POP_SIZE, dtype=np.float32)

    d_population = cuda.to_device(population)
    d_selected = cuda.to_device(selected)
    d_fitness = cuda.to_device(fitness)

    threads_per_block = BLOCK_SIZE
    blocks_per_grid = (POP_SIZE + threads_per_block - 1) // threads_per_block
    print(f"Blocks per grid: {blocks_per_grid}, Threads per block: {threads_per_block}")

    rng_states = create_xoroshiro128p_states(POP_SIZE, seed=int(time.time()))
    initialize_population_kernel[blocks_per_grid, threads_per_block](d_population, rng_states)

    best_fitness = float('inf')
    generations_wout_improvement = 0

    start_time = time.time()

    for gen in range(GENERATIONS):
        evaluate_fitness_kernel[blocks_per_grid, threads_per_block](d_population, d_cities_x, d_cities_y, d_fitness)
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

        tournament_selection_kernel[blocks_per_grid, threads_per_block](d_population, d_fitness, d_selected, rng_states)
        cuda.synchronize()

        crossover_kernel[blocks_per_grid, threads_per_block](d_population, d_selected, rng_states)
        cuda.synchronize()

        mutate_kernel[blocks_per_grid, threads_per_block](d_population, rng_states)
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

    CITIES = len(cities)

    tsp_genetic_algorithm_cuda(cities)
