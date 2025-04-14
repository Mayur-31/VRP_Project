from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import logging
import numpy as np
from typing import Dict, List

logging.basicConfig(level=logging.INFO)

class VRPSolver:
    def __init__(self, distance_matrix: List[List[float]], num_vehicles: int):
        self._validate_matrix(distance_matrix)
        self.distance_matrix = self._sanitize_matrix(distance_matrix)
        self.num_vehicles = num_vehicles
        self.depot_index = 0
        
        try:
            self.manager = pywrapcp.RoutingIndexManager(
                len(self.distance_matrix), self.num_vehicles, self.depot_index
            )
            self.routing = pywrapcp.RoutingModel(self.manager)
        except Exception as e:
            logging.error(f"OR-Tools initialization failed: {e}")
            raise

    def _sanitize_matrix(self, matrix):
        sanitized = []
        for row in matrix:
            sanitized_row = [float(min(max(abs(d), 0.0), 1000.0)) for d in row]
            sanitized.append(sanitized_row)
        return sanitized

    def _validate_matrix(self, matrix):
        if not matrix or len(matrix) != len(matrix[0]):
            raise ValueError("Invalid distance matrix")
        for i, row in enumerate(matrix):
            for j, d in enumerate(row):
                if not isinstance(d, (int, float)) or d < 0:
                    raise ValueError(f"Invalid distance at ({i},{j}): {d}")

    def solve(self) -> Dict[int, List[int]]:
        try:
            transit_callback_index = self.routing.RegisterTransitCallback(self._distance_callback)
            self.routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
            
            demand_callback_index = self.routing.RegisterUnaryTransitCallback(lambda index: 1)
            self.routing.AddDimensionWithVehicleCapacity(
                demand_callback_index, 0, [1000] * self.num_vehicles, True, 'Capacity'
            )
            
            search_parameters = self._configure_search()
            solution = self.routing.SolveWithParameters(search_parameters)
            
            if solution:
                return self._format_solution(solution)
            logging.error("No solution found.")
            return {}
        except Exception as e:
            logging.error(f"Solver error: {e}")
            return {}

    def _distance_callback(self, from_index, to_index):
        from_node = self.manager.IndexToNode(from_index)
        to_node = self.manager.IndexToNode(to_index)
        return self.distance_matrix[from_node][to_node]

    def _configure_search(self):
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH
        )
        search_parameters.time_limit.seconds = 30  # Reduced time
        search_parameters.log_search = False
        return search_parameters

    def _format_solution(self, solution):
        routes = {}
        for vehicle_id in range(self.num_vehicles):
            index = self.routing.Start(vehicle_id)
            route = []
            while not self.routing.IsEnd(index):
                node = self.manager.IndexToNode(index)
                route.append(node)
                index = solution.Value(self.routing.NextVar(index))
            routes[vehicle_id] = route
        return routes