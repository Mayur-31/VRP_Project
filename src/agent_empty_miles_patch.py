# agent_empty_miles_patch.py
import pandas as pd
import logging
from src.langchain_agent import LangChainAgent
import re

def patched_format_driver_jobs(self, driver_name: str) -> str:
    """Enhanced version with reliable datetime parsing"""
    try:
        clean_driver = re.split(r'[\[\(]', driver_name)[0].strip().upper()
        jobs_df = self.distance_analyzer.jobs_df
        
        # Find all driver matches using cleaned names
        mask = jobs_df['DRIVER NAME'].str.split(r'[\[\(]').str[0].str.strip().str.upper() == clean_driver
        driver_jobs = jobs_df[mask].copy()
        
        if driver_jobs.empty:
            return f"No jobs found for driver {clean_driver}"

        # Ensure proper datetime type
        driver_jobs['DEPARTURE_DATETIME'] = pd.to_datetime(
            driver_jobs['DEPARTURE_DATETIME'],
            format='%d/%m/%Y %H:%M',
            errors='coerce'
        )
        
        # Sort chronologically
        driver_jobs = driver_jobs.sort_values('DEPARTURE_DATETIME')
        output_lines = [f"Jobs for {clean_driver}:"]
        prev_delivery = None
        
        for _, row in driver_jobs.iterrows():
            # Format datetime with validation
            if pd.notnull(row['DEPARTURE_DATETIME']):
                dep_time = row['DEPARTURE_DATETIME'].strftime('%d/%m %H:%M')
            else:
                # Fallback to original columns if needed
                date_part = pd.to_datetime(row['DATE'], dayfirst=True).strftime('%d/%m')
                time_part = str(row['DEPARTURE TIME']).split('.')[0]  # Handle float times
                dep_time = f"{date_part} {time_part}" if date_part != 'NaT' else "Time Unknown"

            # Rest of the formatting logic remains the same...
            loaded = row.get('LOADED MILES', 0)
            empty = row.get('EMPTY MILES', 0)
            
            if prev_delivery:
                empty_str = f"{empty:.4f}mi (from {prev_delivery})"
            else:
                empty_str = "First job"
                
            output_lines.append(
                f"- {row['COLLECTION POST CODE']} → {row['DELIVER POST CODE']} "
                f"({dep_time}) | Loaded: {loaded:.1f}mi | Empty: {empty_str}"
            )
            prev_delivery = row['DELIVER POST CODE']
            
        return "\n".join(output_lines)
    except Exception as e:
        logging.error(f"Error formatting jobs: {e}")
        return f"Error showing jobs: {str(e)}"