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


def run_client(cities, POP_SIZE, GENERATIONS, MAX_GENERATIONS_WITHOUT_IMPROVEMENT):
    channel1 = grpc.insecure_channel('localhost:50052')
    channel2 = grpc.insecure_channel('localhost:50053')

    stub1 = tsp_pb2_grpc.TSPSolverStub(channel1)
    stub2 = tsp_pb2_grpc.TSPSolverStub(channel2)

    CITIES = len(cities)
    population = np.hstack([np.random.permutation(CITIES) for _ in range(POP_SIZE)]).tolist()
    cities_x = [city[0] for city in cities]
    cities_y = [city[1] for city in cities]
    fitness = [0] * POP_SIZE

    best_fitness = float('inf')
    generations_wout_improvement = 0

    for gen in range(GENERATIONS):
        print(f"Generation {gen}:", end=" ")

        half = POP_SIZE // 2

        with ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(
                stub1.ProcessPopulation,
                tsp_pb2.PopulationRequest(
                    global_population=population,
                    global_fitness=fitness,
                    cities_x=cities_x,
                    cities_y=cities_y,
                    batch_size=half,
                    seed=gen * 2
                )
            )

            future2 = executor.submit(
                stub2.ProcessPopulation,
                tsp_pb2.PopulationRequest(
                    global_population=population,
                    global_fitness=fitness,
                    cities_x=cities_x,
                    cities_y=cities_y,
                    batch_size=half,
                    seed=gen * 2 + 1
                )
            )

            response1 = future1.result()
            response2 = future2.result()

        population[:half * CITIES] = response1.updated_population
        population[half * CITIES:] = response2.updated_population

        fitness[:half] = response1.updated_fitness
        fitness[half:] = response2.updated_fitness

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

    run_client(cities, 50000, 500, 100)
