# vrp_solver.py

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import logging
from typing import Tuple, Dict, List

logging.basicConfig(level=logging.INFO)

class VRPSolver:
    def __init__(self, distance_matrix: List[List[float]], num_vehicles: int):
        self.distance_matrix = distance_matrix
        self.num_vehicles = num_vehicles
        self.manager = pywrapcp.RoutingIndexManager(
            len(distance_matrix),
            num_vehicles,
            0  # Depot index
        )
        self.routing = pywrapcp.RoutingModel(self.manager)

    def solve(self) -> Dict[int, List[int]]:
        """Solve VRP and return optimized routes"""
        try:
            transit_callback_index = self.routing.RegisterTransitCallback(
                lambda from_idx, to_idx: self.distance_matrix[
                    self.manager.IndexToNode(from_idx)
                ][self.manager.IndexToNode(to_idx)]
            )

            self.routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
            self._add_constraints(transit_callback_index)

            search_parameters = self._configure_search()
            solution = self.routing.SolveWithParameters(search_parameters)

            return self._format_solution(solution) if solution else {}
        except Exception as e:
            logging.error(f"VRP solve failed: {e}")
            return {}

    def _add_constraints(self, transit_callback_index: int):
        """Add problem constraints"""
        dimension_name = 'Distance'
        self.routing.AddDimension(
            transit_callback_index,
            0,  # Slack
            3000,  # Max route distance (miles)
            True,  # Start cumul to zero
            dimension_name
        )
        distance_dimension = self.routing.GetDimensionOrDie(dimension_name)
        distance_dimension.SetGlobalSpanCostCoefficient(100)

    def _configure_search(self):
        """Configure OR-Tools search parameters"""
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = 30
        return search_parameters

    def _format_solution(self, solution) -> Dict[int, List[int]]:
        """Format OR-Tools solution"""
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