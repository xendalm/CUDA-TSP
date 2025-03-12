import argparse
import logging
import time
from concurrent import futures

import grpc
import numpy as np
from grpc_reflection.v1alpha import reflection
from numba import cuda

from generated.api.tsp import tsp_pb2, tsp_pb2_grpc
from tsp_cuda import tournament_selection_kernel, crossover_kernel, mutate_kernel, evaluate_fitness_kernel

BLOCK_SIZE = 512

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class TSPSolverServicer(tsp_pb2_grpc.TSPSolverServicer):
    def ProcessPopulation(self, request, context):
        CITIES = len(request.cities_x)
        GLOBAL_POP_SIZE = len(request.global_population) // CITIES
        POP_SIZE = request.batch_size

        global_population = np.array(request.global_population, dtype=np.int32).reshape((GLOBAL_POP_SIZE, CITIES))
        global_fitness = np.array(request.global_fitness, dtype=np.float32)
        cities_x = np.array(request.cities_x, dtype=np.float32)
        cities_y = np.array(request.cities_y, dtype=np.float32)
        selected = np.zeros((POP_SIZE, CITIES), dtype=np.int32)
        fitness = np.zeros(POP_SIZE, dtype=np.float32)

        d_global_population = cuda.to_device(global_population)
        d_global_fitness = cuda.to_device(global_fitness)
        d_cities_x = cuda.to_device(cities_x)
        d_cities_y = cuda.to_device(cities_y)
        d_selected = cuda.to_device(selected)
        d_fitness = cuda.to_device(fitness)
        rng_states = cuda.random.create_xoroshiro128p_states(POP_SIZE, seed=int(time.time()) + request.seed)

        threads_per_block = BLOCK_SIZE
        blocks_per_grid = (POP_SIZE + threads_per_block - 1) // threads_per_block

        logger.info(f"Global population {GLOBAL_POP_SIZE}, Local Population {POP_SIZE}, Cities {CITIES}, "
                    f"Blocks per grid: {blocks_per_grid}, Threads per block: {threads_per_block}")

        tournament_selection_kernel[blocks_per_grid, threads_per_block](d_selected, d_global_population,
                                                                        d_global_fitness, rng_states, POP_SIZE, CITIES)
        cuda.synchronize()

        d_offspring1 = cuda.to_device(np.zeros((POP_SIZE // 2, CITIES), dtype=np.int32))
        d_offspring2 = cuda.to_device(np.zeros((POP_SIZE // 2, CITIES), dtype=np.int32))
        crossover_kernel[blocks_per_grid, threads_per_block](d_selected, d_offspring1, d_offspring2, rng_states,
                                                             POP_SIZE, CITIES)
        cuda.synchronize()

        mutate_kernel[blocks_per_grid, threads_per_block](d_selected, rng_states, POP_SIZE, CITIES)
        cuda.synchronize()

        evaluate_fitness_kernel[blocks_per_grid, threads_per_block](d_selected, d_cities_x, d_cities_y, d_fitness,
                                                                    POP_SIZE, CITIES)
        cuda.synchronize()

        h_selected = d_selected.copy_to_host().flatten().tolist()
        h_fitness = d_fitness.copy_to_host().tolist()

        best_fitness = np.min(h_fitness)
        logger.info(f"Generation best fitness: {best_fitness}")

        return tsp_pb2.PopulationResponse(updated_population=h_selected, updated_fitness=h_fitness)


def serve(port):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), interceptors=[])
    tsp_pb2_grpc.add_TSPSolverServicer_to_server(TSPSolverServicer(), server)

    SERVICE_NAMES = (
        tsp_pb2.DESCRIPTOR.services_by_name["TSPSolver"].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)

    server.add_insecure_port(f"[::]:{port}")
    logger.info(f"Server started at port {port}...")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TSP Solver gRPC Server")
    parser.add_argument('--port', type=int, required=True, help='Port to run the server on')
    args = parser.parse_args()

    serve(args.port)
