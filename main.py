#main.py

import logging
import src.agent_empty_miles_patch
from dotenv import load_dotenv
from src.data_processor import DataProcessor
from src.ai_helper import AIHelper
from src.vrp_solver import VRPSolver
from src.report_generator import ReportGenerator
from src.langchain_agent import LangChainAgent
from src.distance_analyzer import DistanceAnalyzer

def main():
    load_dotenv("Test.env")
    try:
        # Initialize components
        processor = DataProcessor()
        ai_helper = AIHelper()
        # Use enhanced_jobs.csv for empty/loaded miles and add/remove job operations
        distance_analyzer = DistanceAnalyzer(jobs_path='data/enhanced_job_distances.csv')
        
        # Load data (for context, use original jobs file)
        jobs, drivers = processor.load_data()
        postcodes = processor.get_unique_postcodes()

        # Initialize LangChain agent
        agent = LangChainAgent(processor.context, distance_analyzer)

        # Interactive session for questions and job modifications
        while True:
            command = input("\nAsk a question (or 'optimize/add/remove/exit'): ").strip().lower()
            if command == 'optimize':
                break
            if command == 'add':
                _add_new_job(distance_analyzer)
                continue
            if command == 'remove':
                _remove_job(distance_analyzer)
                continue
            if command == 'exit':
                return
            print(agent.answer_question(command))

        # Continue with optimization flow
        distance_matrix = ai_helper.build_distance_matrix(postcodes)
        solver = VRPSolver(distance_matrix, len(drivers))
        routes = solver.solve()
        ReportGenerator.generate_summary(processor.context, routes, postcodes)
        ReportGenerator.generate_assignments(routes, postcodes, drivers['DRIVER'].tolist())
    except Exception as e:
        logging.error(f"System error: {e}")

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
