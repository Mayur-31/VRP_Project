# report_generator.py

import pandas as pd
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO)

class ReportGenerator:
    @staticmethod
    def generate_summary(context: Dict, routes: Dict, postcodes: List[str], output_file: str = "summary_metrics.csv"):
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
        pd.DataFrame(metrics).to_csv(output_file, index=False)
        logging.info(f"Saved summary report to {output_file}")

    @staticmethod
    def generate_assignments(routes: Dict, postcodes: List[str], drivers: List[str], output_file: str = "driver_assignments.csv"):
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
