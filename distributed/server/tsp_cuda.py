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
def tournament_selection_kernel(selected, global_population, global_fitness, rng_states, population_size, cities_num):
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

    for i in range(cities_num):
        selected[idx, i] = global_population[best_candidate, i]


@cuda.jit
def crossover_kernel(selected, offspring, used, rng_states, population_size, cities_num):
    idx = cuda.grid(1)
    if idx >= population_size // 2:
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
        selected[parent1_idx, i] = offspring[parent1_idx, i]
        selected[parent2_idx, i] = offspring[parent2_idx, i]


@cuda.jit
def mutate_kernel(population, rng_states, population_size, cities_num):
    idx = cuda.grid(1)
    if idx >= population_size:
        return

    if cuda.random.xoroshiro128p_uniform_float32(rng_states, idx) < MUTATION_RATE:
        a = int(xoroshiro128p_uniform_float32(rng_states, idx) * cities_num) % cities_num
        b = int(xoroshiro128p_uniform_float32(rng_states, idx) * cities_num) % cities_num
        temp = population[idx, a]
        population[idx, a] = population[idx, b]
        population[idx, b] = temp


@cuda.jit
def evaluate_fitness_kernel(population, cities_x, cities_y, fitness, population_size, cities_num):
    idx = cuda.grid(1)
    if idx >= population_size:
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
