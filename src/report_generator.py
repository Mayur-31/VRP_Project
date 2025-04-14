# report_generator.py

import pandas as pd
import logging
from typing import Dict, List
import os
import pathlib

logging.basicConfig(level=logging.INFO)

class ReportGenerator:
    @staticmethod
    def generate_summary(context: Dict, routes: Dict, postcodes: List[str], output_file: str = "reports/summary_metrics.csv"):
        try:
            pathlib.Path(output_file).parent.mkdir(exist_ok=True, parents=True)
            total_loaded = sum(
                sum(routes[vehicle][i+1] - routes[vehicle][i]
                    for i in range(len(routes[vehicle])-1))
                for vehicle in routes
                )
        
            metrics = {
                'Total Jobs': [context['total_jobs']],
                'Total Drivers': [context['total_drivers']],
                'Earliest Departure': [context['departure time_min']],
                'Latest Arrival': [context['arrival time_max']],
                'Total Loaded Miles': [total_loaded],
                'Unique Postcodes': [len(postcodes)]
                }
        
        
        # Rest of your existing code
            pd.DataFrame(metrics).to_csv(output_file, index=False)
        except PermissionError as pe:
            logging.error(f"Permission denied: {pe}. Try running as administrator or choose different output directory.")
        except Exception as e:
            logging.error(f"File save failed: {e}")
    
    
    
    

    @staticmethod
    def generate_assignments(routes: Dict, postcodes: List[str], drivers: List[str], output_file: str = "reports/driver_assignments.csv"):
        pathlib.Path(output_file).parent.mkdir(exist_ok=True, parents=True)
        report_data = []
        for vehicle_id, route in routes.items():
            driver = drivers[vehicle_id % len(drivers)]
            postcode_sequence = [postcodes[i] for i in route]
            report_data.append({
                'Driver': driver,
                'Route': ' -> '.join(postcode_sequence),
                'Stops': len(route),
                'Estimated Miles': sum(postcodes[i] for i in route)
            })
        pd.DataFrame(report_data).to_csv(output_file, index=False)
        logging.info(f"Saved assignments report to {output_file}")