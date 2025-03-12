import math
import warnings

from numba import cuda
from numba.core.errors import NumbaPerformanceWarning
from numba.cuda.random import xoroshiro128p_uniform_float32

warnings.simplefilter("ignore", category=NumbaPerformanceWarning)

MUTATION_RATE = 0.1
TOURNAMENT_SIZE = 4


@cuda.jit(device=True)
def cuda_distance(city1_x, city1_y, city2_x, city2_y):
    return math.sqrt((city1_x - city2_x) ** 2 + (city1_y - city2_y) ** 2)


@cuda.jit
def tournament_selection_kernel(selected, global_population, global_fitness, rng_states, population_size, cities_size):
    idx = cuda.grid(1)
    if idx >= population_size:
        return

    global_population_size = len(global_population)
    best_candidate = int(
        xoroshiro128p_uniform_float32(rng_states, idx) * global_population_size) % global_population_size

    for i in range(1, TOURNAMENT_SIZE):
        candidate = int(
            xoroshiro128p_uniform_float32(rng_states, idx) * global_population_size) % global_population_size
        if global_fitness[candidate] < global_fitness[best_candidate]:
            best_candidate = candidate

    for i in range(cities_size):
        selected[idx, i] = global_population[best_candidate, i]


@cuda.jit
def crossover_kernel(selected, offspring1, offspring2, rng_states, population_size, cities_size):
    idx = cuda.grid(1)
    if idx >= population_size // 2:
        return

    parent1_idx = 2 * idx
    parent2_idx = 2 * idx + 1

    start = int(xoroshiro128p_uniform_float32(rng_states, idx) * cities_size) % cities_size
    end = start + int(xoroshiro128p_uniform_float32(rng_states, idx) * (cities_size - start))

    for i in range(cities_size):
        offspring1[idx, i] = -1
        offspring2[idx, i] = -1

    for i in range(0, cities_size):
        offspring1[idx, i] = selected[parent2_idx, i]
        offspring2[idx, i] = selected[parent1_idx, i]

    pos1 = 0
    pos2 = 0
    for i in range(cities_size):
        city = selected[parent1_idx, i]
        if offspring1[idx, city] == -1:
            while start <= pos1 <= end:
                pos1 += 1
            offspring1[idx, pos1] = city
            pos1 += 1

        city = selected[parent2_idx, i]
        if offspring2[idx, i] == -1:
            while start <= pos2 <= end:
                pos2 += 1
            offspring2[idx, pos2] = city
            pos2 += 1

    for i in range(cities_size):
        selected[parent1_idx, i] = offspring1[idx, i]
        selected[parent2_idx, i] = offspring2[idx, i]


@cuda.jit
def mutate_kernel(population, rng_states, population_size, cities_size):
    idx = cuda.grid(1)
    if idx >= population_size:
        return

    if cuda.random.xoroshiro128p_uniform_float32(rng_states, idx) < MUTATION_RATE:
        a = int(xoroshiro128p_uniform_float32(rng_states, idx) * cities_size) % cities_size
        b = int(xoroshiro128p_uniform_float32(rng_states, idx) * cities_size) % cities_size
        temp = population[idx, a]
        population[idx, a] = population[idx, b]
        population[idx, b] = temp


@cuda.jit
def evaluate_fitness_kernel(population, cities_x, cities_y, fitness, population_size, cities_size):
    idx = cuda.grid(1)
    if idx >= population_size:
        return

    total_distance = 0.0
    for i in range(cities_size - 1):
        city1_idx = population[idx, i]
        city2_idx = population[idx, i + 1]
        total_distance += cuda_distance(cities_x[city1_idx], cities_y[city1_idx],
                                        cities_x[city2_idx], cities_y[city2_idx])

    city1_idx = population[idx, cities_size - 1]
    city2_idx = population[idx, 0]
    total_distance += cuda_distance(cities_x[city1_idx], cities_y[city1_idx],
                                    cities_x[city2_idx], cities_y[city2_idx])
    fitness[idx] = total_distance
