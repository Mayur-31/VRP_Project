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
        self.context = {'driver_rest': {}}

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and preprocess datasets."""
        try:
            self.jobs = pd.read_csv('data/Planning NEW TEST - Sheet7.csv')
            self.drivers = pd.read_csv('data/Planning NEW TEST - List of current Drivers-2.csv')

            self._standardize_columns()
            self._clean_postcodes()
            self._convert_dtypes()
            self._calculate_rest_times()
            self._build_context()
            return self.jobs, self.drivers
        except Exception as e:
            logging.error(f"Data loading failed: {e}")
            raise

    def _standardize_columns(self):
        """Standardize driver names."""
        self.jobs['DRIVER NAME'] = self.jobs['DRIVER NAME'].str.split(r'\[|\(').str[0].str.strip()
        self.drivers['DRIVER'] = self.drivers['DRIVER'].str.split(r'\[|\(').str[0].str.strip()

    def _clean_postcodes(self):
        """Clean and validate postcodes, filtering out invalid ones."""
        uk_postcode_pattern = r'^[A-Z]{1,2}\d[A-Z\d]? ?\d[A-Z]{2}$'
        for col in ['COLLECTION POST CODE', 'DELIVER POST CODE']:
            self.jobs[col] = (
                self.jobs[col]
                .astype(str)
                .str.strip()
                .str.upper()
                .str.replace(' ', '')
                .apply(lambda x: f"{x[:-3]} {x[-3:]}" if len(x) > 3 else x)
            )
            valid_mask = self.jobs[col].str.match(uk_postcode_pattern, na=False)
            invalid = self.jobs[col][~valid_mask]
            if not invalid.empty:
                logging.warning(f"Invalid postcodes in {col}: {invalid.unique().tolist()}")
            self.jobs = self.jobs[valid_mask].reset_index(drop=True)

    def _convert_dtypes(self):
        """Convert date/time columns to datetime objects."""
        time_format = '%H:%M'
        time_cols = ['ON DOOR TIME', 'DEPARTURE TIME', 'ARRIVAL TIME', 'RUN TIME']
        for col in time_cols:
            if col in self.jobs.columns:
                self.jobs[col] = pd.to_datetime(self.jobs[col], format=time_format, errors='coerce')
        self.jobs['DATE'] = pd.to_datetime(self.jobs['DATE'], format='%m/%d/%Y', errors='coerce')
        self.jobs['RUN TIME'] = (self.jobs['ARRIVAL TIME'] - self.jobs['DEPARTURE TIME']).dt.total_seconds() / 3600
        self.jobs['DEPARTURE_DATETIME'] = pd.to_datetime(
            self.jobs['DATE'].dt.strftime('%m/%d/%Y') + ' ' + 
            self.jobs['DEPARTURE TIME'].dt.strftime('%H:%M'),
            format='%m/%d/%Y %H:%M',
            errors='coerce'
        )
        self.drivers['STARTDATE'] = pd.to_datetime(self.drivers['STARTDATE'], format='%d/%m/%Y', errors='coerce')

    def _calculate_rest_times(self):
        """Calculate rest times between consecutive jobs per driver."""
        driver_rest = {}
        sorted_jobs = self.jobs.sort_values(['DRIVER NAME', 'DEPARTURE_DATETIME'])
        for driver, group in sorted_jobs.groupby('DRIVER NAME'):
            rest_times = []
            prev_end = None
            for _, row in group.iterrows():
                if prev_end is not None and pd.notnull(row['DEPARTURE_DATETIME']):
                    rest_hours = (row['DEPARTURE_DATETIME'] - prev_end).total_seconds() / 3600
                    rest_times.append(rest_hours)
                current_end = row['ARRIVAL TIME'] if pd.notnull(row['ARRIVAL TIME']) else row['DEPARTURE_DATETIME']
                prev_end = current_end if pd.notnull(current_end) else prev_end
            driver_rest[driver] = rest_times
        self.context['driver_rest'] = driver_rest

    def _build_context(self):
        """Build summary context for LangChain."""
        self.context.update({
            'total_jobs': len(self.jobs),
            'total_drivers': len(self.drivers),
            'job_types': self.jobs['JOB TYPE'].value_counts().to_dict(),
            'driver_levels': self.drivers.set_index('DRIVER')['LEVEL'].to_dict(),
            'customers': self.jobs['CUSTOMER'].value_counts().to_dict(),
            'postcodes': {
                'collection': self.jobs['COLLECTION POST CODE'].nunique(),
                'delivery': self.jobs['DELIVER POST CODE'].nunique()
            }
        })
        time_fields = ['DEPARTURE TIME', 'ARRIVAL TIME', 'ON DOOR TIME']
        for field in time_fields:
            valid = self.jobs[field].dropna()
            self.context[f'{field.lower()}_min'] = valid.min().strftime('%H:%M') if not valid.empty else 'N/A'
            self.context[f'{field.lower()}_max'] = valid.max().strftime('%H:%M') if not valid.empty else 'N/A'

    def get_unique_postcodes(self) -> List[str]:
        """Get sorted unique postcodes."""
        coll = self.jobs['COLLECTION POST CODE'].dropna().unique().tolist()
        deliver = self.jobs['DELIVER POST CODE'].dropna().unique().tolist()
        return sorted(list(set(coll + deliver)))