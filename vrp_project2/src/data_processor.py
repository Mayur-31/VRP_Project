# data_processor.py

import pandas as pd
import logging
import re
from datetime import datetime
from typing import Tuple, Dict, List

logging.basicConfig(level=logging.INFO)

class DataProcessor:
    def __init__(self):
        self.jobs = None
        self.drivers = None
        self.context = {}
        self.distance_matrix = None
        self.context = {'driver_rest': {}}

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and preprocess datasets"""
        try:
            # Load CSVs
            self.jobs = pd.read_csv('data/Planning NEW TEST - Sheet7.csv')
            self.drivers = pd.read_csv('data/Planning NEW TEST - List of current Drivers-2.csv')

            # Clean data
            self._standardize_columns()
            self._clean_postcodes()
            self._convert_dtypes()
            self._calculate_rest_times()

            # Build context
            self._build_context()
            return self.jobs, self.drivers
        except Exception as e:
            logging.error(f"Data loading failed: {e}")
            raise

    def _standardize_columns(self):
        """Fixed driver name standardization"""
        # For jobs
        self.jobs['DRIVER NAME'] = self.jobs['DRIVER NAME'].str.split(r'\[|\(').str[0].str.strip()

        # For drivers
        self.drivers['DRIVER'] = self.drivers['DRIVER'].str.split(r'\[|\(').str[0].str.strip()

    def _clean_postcodes(self):
        uk_postcode_pattern = r'^[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}$'
    
        for col in ['COLLECTION POST CODE', 'DELIVER POST CODE']:
            self.jobs[col] = self.jobs[col].apply(
                lambda x: re.sub(r'\s+', '', str(x).upper())
                if re.match(uk_postcode_pattern, str(x).upper())
                else pd.NA
            )

    def _convert_dtypes(self):
        """Convert date/time columns to datetime objects"""
    # Jobs time columns
        time_format = '%H:%M'
        time_cols = ['ON DOOR TIME', 'DEPARTURE TIME', 'ARRIVAL TIME', 'RUN TIME']

        for col in time_cols:
            if col in self.jobs.columns:
                 self.jobs[col] = pd.to_datetime(
                    self.jobs[col],
                    format=time_format,
                    errors='coerce'
                )

    # Jobs date column (assuming format MM/DD/YYYY)
        self.jobs['DATE'] = pd.to_datetime(
        self.jobs['DATE'],
        dayfirst=True,
        errors='coerce'
    )
        self.jobs['RUN_TIME_DURATION'] = (
        self.jobs['ARRIVAL TIME'] - self.jobs['DEPARTURE TIME']
        ).dt.total_seconds() / 3600

    # Create proper datetime fields using combine
        self.jobs['DEPARTURE_DATETIME'] = self.jobs.apply(
            lambda row: datetime.combine(row['DATE'].date(), row['DEPARTURE TIME'].time())
            if pd.notnull(row['DATE']) and pd.notnull(row['DEPARTURE TIME']) else pd.NaT,
            axis=1
        )
    # Drivers start date (assuming format DD/MM/YYYY)
        self.drivers['STARTDATE'] = pd.to_datetime(
            self.drivers['STARTDATE'],
            format='%d/%m/%Y',
            errors='coerce'
        )

    def _calculate_rest_times(self):
        self.context['driver_rest'] = {}
        sorted_jobs = self.jobs.sort_values(['DRIVER NAME', 'DEPARTURE_DATETIME'])

        for driver, group in sorted_jobs.groupby('DRIVER NAME'):
            rest_periods = []
            prev_end = None

            for _, row in group.iterrows():
                if prev_end and pd.notnull(row['DEPARTURE_DATETIME']):
                    rest_hours = (row['DEPARTURE_DATETIME'] - prev_end).total_seconds() / 3600
                    rest_periods.append(rest_hours)
                
                current_end = row['ARRIVAL TIME'] if pd.notnull(row['ARRIVAL TIME']) else row['DEPARTURE_DATETIME']
                prev_end = current_end if pd.notnull(current_end) else prev_end

            self.context['driver_rest'][driver] = {
                'periods': rest_periods,
                'all_rests': rest_periods
            }
    
    def _build_context(self):
        """Build summary context for LangChain"""
        # Basic metrics
        self.context.update({
            'total_jobs': len(self.jobs),
            'total_drivers': len(self.drivers),
            'job_types': self.jobs['JOB TYPE'].value_counts().to_dict(),
            'driver_levels': self.drivers['LEVEL'].value_counts().to_dict(),
            'customers': self.jobs['CUSTOMER'].value_counts().to_dict(),
            'driver_rest': self.context['driver_rest'],
            'postcodes': {
                'collection': self.jobs['COLLECTION POST CODE'].nunique(),
                'delivery': self.jobs['DELIVER POST CODE'].nunique()
            }
        })
        
        driver_levels = self.drivers.set_index('DRIVER')['LEVEL'].to_dict()
        self.context['driver_levels'] = driver_levels
        self.context['max_level_driver'] = self.drivers.loc[
        self.drivers['LEVEL'].idxmax()
        ]['DRIVER']
        
        # Time metrics
        time_fields = ['DEPARTURE TIME', 'ARRIVAL TIME', 'ON DOOR TIME']
        for field in time_fields:
            valid = self.jobs[field].dropna()
            self.context[f'{field.lower()}_min'] = valid.min().strftime('%Y-%m-%d %H:%M:%S')
            self.context[f'{field.lower()}_max'] = valid.max().strftime('%Y-%m-%d %H:%M:%S')

    def get_unique_postcodes(self) -> List[str]:
        """Get sorted unique postcodes"""
        coll = self.jobs['COLLECTION POST CODE'].dropna().unique().tolist()
        deliver = self.jobs['DELIVER POST CODE'].dropna().unique().tolist()
        return sorted(list(set(coll + deliver)))