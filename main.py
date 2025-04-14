import logging
from dotenv import load_dotenv
from src.data_processor import DataProcessor
from src.ai_helper import AIHelper
from src.vrp_solver import VRPSolver
from src.report_generator import ReportGenerator
from src.langchain_agent import LangChainAgent
from src.distance_analyzer import DistanceAnalyzer
import pandas as pd
import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

logging.basicConfig(level=logging.INFO)

def load_data():
    """Load job details and distance matrix from CSV files."""
    jobs_df = pd.read_csv("data/enhanced_job_distances.csv")
    distance_df = pd.read_csv("data/new_driving_distance_matrix_miles.csv", index_col=0)
    return jobs_df, distance_df

def create_data_model(distance_matrix, postcodes, depot_index):
    """Creates a data model for a single driver."""
    data = {}
    data["distance_matrix"] = distance_matrix
    data["num_vehicles"] = 1  # One vehicle per driver
    data["depot"] = depot_index
    data["postcodes"] = postcodes
    return data

def print_solution(data, manager, routing, solution, driver_name):
    """Prints the optimized route for a specific driver."""
    print(f"\nDriver: {driver_name} - Objective: {solution.ObjectiveValue()}")
    index = routing.Start(0)  # Single vehicle (vehicle 0)
    plan_output = "Route:\n"
    route_distance = 0
    while not routing.IsEnd(index):
        node = data["postcodes"][manager.IndexToNode(index)]
        plan_output += f" {node} -> "
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    node = data["postcodes"][manager.IndexToNode(index)]
    plan_output += f"{node}\n"
    route_distance_miles = route_distance / 1000.0  # Convert back to miles
    plan_output += f"Distance: {route_distance_miles:.2f} miles\n"
    print(plan_output)

def solve_vrp_for_driver(data, driver_name):
    """Solves the VRP for a single driver with memory-efficient settings."""
    try:
        manager = pywrapcp.RoutingIndexManager(
            len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
        )
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data["distance_matrix"][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        dimension_name = "Distance"
        routing.AddDimension(
            transit_callback_index,
            0,  # No slack
            3000000,  # 3000 miles * 1000 (scaled)
            True,  # Start cumul to zero
            dimension_name,
        )
        distance_dimension = routing.GetDimensionOrDie(dimension_name)
        distance_dimension.SetGlobalSpanCostCoefficient(100)

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT
        )
        search_parameters.time_limit.seconds = 10  # 10 seconds per driver
        search_parameters.log_search = False  # Disable logging to save memory

        solution = routing.SolveWithParameters(search_parameters)
        if solution:
            print_solution(data, manager, routing, solution, driver_name)
        else:
            print(f"No solution found for Driver {driver_name} within 10 seconds.")
    except Exception as e:
        logging.error(f"Error solving VRP for {driver_name}: {e}")

def optimize_routes():
    """Optimize routes for all drivers using the precomputed distance matrix."""
    try:
        jobs_df, distance_df = load_data()
        distance_matrix = (distance_df.values * 1000).round().astype(int).tolist()
        all_postcodes = distance_df.index.tolist()
        drivers = jobs_df["DRIVER NAME"].unique()

        for driver in drivers:
            driver_jobs = jobs_df[jobs_df["DRIVER NAME"] == driver]
            collection_postcodes = driver_jobs["COLLECTION POST CODE"].dropna().unique()
            deliver_postcodes = driver_jobs["DELIVER POST CODE"].dropna().unique()
            driver_postcodes = list(set(collection_postcodes) | set(deliver_postcodes))
            valid_postcodes = [pc for pc in driver_postcodes if pc in all_postcodes]
            
            if len(valid_postcodes) < 2:
                print(f"\nDriver {driver} has insufficient valid postcodes ({len(valid_postcodes)}). Skipping.")
                continue

            postcode_to_index = {pc: idx for idx, pc in enumerate(all_postcodes)}
            indices = [postcode_to_index[pc] for pc in valid_postcodes]
            cluster_matrix = [[distance_matrix[i][j] for j in indices] for i in indices]

            depot_postcode = driver_jobs["COLLECTION POST CODE"].iloc[0]
            depot_index = valid_postcodes.index(depot_postcode) if depot_postcode in valid_postcodes else 0

            data = create_data_model(cluster_matrix, valid_postcodes, depot_index)
            print(f"\nSolving for Driver {driver} with {len(valid_postcodes)} locations...")
            solve_vrp_for_driver(data, driver)
    except Exception as e:
        logging.error(f"Optimization failed: {e}")

def main():
    load_dotenv("Test.env")
    try:
        processor = DataProcessor()
        ai_helper = AIHelper()
        distance_analyzer = DistanceAnalyzer(jobs_path='data/enhanced_job_distances.csv')
        jobs, drivers = processor.load_data()
        postcodes = processor.get_unique_postcodes()
        agent = LangChainAgent(processor.context, distance_analyzer)

        while True:
            command = input("\nAsk a question (or 'optimize/add/remove/exit'): ").strip().lower()
            if command == 'optimize':
                optimize_routes()
                continue
            if command == 'add':
                _add_new_job(distance_analyzer)
                continue
            if command == 'remove':
                _remove_job(distance_analyzer)
                continue
            if command == 'exit':
                return
            print(agent.answer_question(command))
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")

def _add_new_job(analyzer):
    """Interface for adding a new job."""
    try:
        new_job = {
            'JOB NAME': input("Enter job name: ").strip(),
            'DATE': input("Enter date (MM/DD/YYYY): ").strip(),
            'DEPARTURE TIME': input("Enter departure time (HH:MM): ").strip(),
            'COLLECTION POST CODE': input("Enter collection postcode: ").strip(),
            'DELIVER POST CODE': input("Enter delivery postcode: ").strip(),
            'DRIVER NAME': input("Enter driver name: ").strip(),
            'CUSTOMER': input("Enter customer name: ").strip(),
            'JOB TYPE': input("Enter job type: ").strip()
        }
        if analyzer.add_job(new_job):
            print("Job added successfully with optimized sequencing!")
        else:
            print("Failed to add job")
    except Exception as e:
        print(f"Error: {str(e)}")

def _remove_job(analyzer):
    """Interface for removing a job."""
    try:
        job_name = input("Enter job name to remove: ").strip()
        if analyzer.remove_job(job_name):
            print("Job removed successfully")
        else:
            print("Job removal failed")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()