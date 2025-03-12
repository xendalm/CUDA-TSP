import sys
from concurrent.futures import ThreadPoolExecutor

import grpc
import numpy as np

from generated.api.tsp import tsp_pb2, tsp_pb2_grpc


def load_cities(filename):
    cities = []
    with open(filename, 'r') as f:
        for line in f:
            id, x, y = line.strip().split()
            cities.append((float(x), float(y)))
    return cities


def run_client(cities, addresses, POP_SIZE, GENERATIONS, MAX_GENERATIONS_WITHOUT_IMPROVEMENT):
    num_servers = len(addresses)

    if POP_SIZE % num_servers != 0:
        raise ValueError(f"POP_SIZE ({POP_SIZE}) must be divisible by num_servers ({num_servers})")

    batch_size = POP_SIZE // num_servers
    channels = [grpc.insecure_channel(addr) for addr in addresses]
    stubs = [tsp_pb2_grpc.TSPSolverStub(channel) for channel in channels]

    CITIES = len(cities)
    population = np.hstack([np.random.permutation(CITIES) for _ in range(POP_SIZE)]).tolist()
    cities_x = [city[0] for city in cities]
    cities_y = [city[1] for city in cities]
    fitness = [0] * POP_SIZE

    best_fitness = float('inf')
    generations_wout_improvement = 0

    for gen in range(GENERATIONS):
        print(f"Generation {gen}:", end=" ")

        with ThreadPoolExecutor(max_workers=num_servers) as executor:
            futures = [
                executor.submit(
                    stub.ProcessPopulation,
                    tsp_pb2.PopulationRequest(
                        global_population=population,
                        global_fitness=fitness,
                        cities_x=cities_x,
                        cities_y=cities_y,
                        batch_size=batch_size,
                        seed=gen * num_servers + i
                    )
                ) for i, stub in enumerate(stubs)
            ]

            responses = [future.result() for future in futures]

        for i, response in enumerate(responses):
            population[i * batch_size * CITIES:(i + 1) * batch_size * CITIES] = response.updated_population
            fitness[i * batch_size:(i + 1) * batch_size] = response.updated_fitness

        gen_best_fitness = np.min(fitness)
        print(f"best fitness = {gen_best_fitness}")

        if gen_best_fitness < best_fitness:
            best_fitness = gen_best_fitness
            generations_wout_improvement = 0
        else:
            generations_wout_improvement += 1
            if generations_wout_improvement >= MAX_GENERATIONS_WITHOUT_IMPROVEMENT:
                break

    print("Best fitness:", best_fitness)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tsp_ga_cuda.py <tsp_data_file>")
        sys.exit(1)

    city_filename = sys.argv[1]
    cities = load_cities(city_filename)

    run_client(cities, ["localhost:50052", "localhost:50053"], 50000, 500, 100)
