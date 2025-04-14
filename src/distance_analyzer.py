# distance_analyzer.py
import pandas as pd
import numpy as np
import requests
import logging
import re
from tqdm import tqdm
from typing import Dict, List, Tuple
from math import radians, sin, cos, sqrt, atan2
import time
from geopy.distance import geodesic
from geopy.geocoders import Nominatim

logging.basicConfig(level=logging.INFO)

# Postcode overrides with cleaned keys (no spaces, uppercase)
POSTCODE_OVERRIDES = {
    'BD112BZ': (53.758755, -1.689026),  # Bradford
    'WA119TY': (53.476785, -2.666254)   # Warrington
}

class DistanceAnalyzer:
    def __init__(self, jobs_path='data/enhanced_job_distances.csv'):
        self.jobs_df = pd.read_csv(jobs_path)
        self.coord_cache = POSTCODE_OVERRIDES.copy()  # Initialize with overrides
        self.distance_matrix = None
        self.postcode_index = {}
        self._validate_input_data()
        
        self._clean_postcodes()
        self._process_datetime()
        self._precompute_distances()
        
    def _clean_postcodes(self):
        uk_pattern = r'^[A-Z]{1,2}\d[A-Z\d]? ?\d[A-Z]{2}$'  # Strict full postcode pattern
    
        for col in ['COLLECTION POST CODE', 'DELIVER POST CODE']:
            self.jobs_df[col] = (
                self.jobs_df[col]
                .astype(str)
                .str.strip()
                .str.upper()
                .str.replace(r'\s+', '', regex=True)  # Remove all spaces
                .apply(lambda x: f"{x[:-3]} {x[-3:]}" if len(x) > 3 else x)  # Add space for validation
                .where(lambda x: x.str.match(uk_pattern), np.nan)
            )
            
        # Apply overrides after cleaning
            self.jobs_df[col] = self.jobs_df[col].apply(
                lambda x: x if x not in POSTCODE_OVERRIDES else x
            )
    
    def _validate_input_data(self):
        initial_count = len(self.jobs_df)
        required_columns = ['COLLECTION POST CODE', 'DELIVER POST CODE', 'DEPARTURE_DATETIME']
        missing = [col for col in required_columns if col not in self.jobs_df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    # Validate postcodes using UK government regex pattern
        uk_postcode_pattern = r'^[A-Z]{1,2}\d[A-Z\d]? ?\d[A-Z]{2}$'
        valid_mask = (
            self.jobs_df['COLLECTION POST CODE'].str.match(uk_postcode_pattern, na=False) &
            self.jobs_df['DELIVER POST CODE'].str.match(uk_postcode_pattern, na=False)
        )
    
        self.jobs_df = self.jobs_df[valid_mask].copy()
        removed_count = initial_count - len(self.jobs_df)
    
        if removed_count > 0:
            logging.warning(f"Removed {removed_count} jobs with invalid postcodes")
            logging.info(f"Sample invalid entries:\n{self.jobs_df[~valid_mask].head(2)}")
    def _get_coordinates(self, postcode: str) -> Tuple[float, float]:
        """Get coordinates with override priority"""
        if pd.isna(postcode):
            return (None, None)
            
        # Clean postcode format for matching
        clean_pc = re.sub(r'\s+', '', postcode).upper()
        
        # 1. Check overrides first
        if clean_pc in POSTCODE_OVERRIDES:
            return POSTCODE_OVERRIDES[clean_pc]
            
        # 2. Check cache
        if clean_pc in self.coord_cache:
            return self.coord_cache[clean_pc]
        
        if not re.match(r'^[A-Z]{1,2}\d[A-Z\d]? ?\d[A-Z]{2}$', clean_pc):
            logging.warning(f"Invalid postcode format: {clean_pc}")
            return (None, None)
        # 3. API lookup
        try:
            response = requests.get(
                f"https://api.postcodes.io/postcodes/{clean_pc}",
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                if data['status'] == 200:
                    result = data['result']
                    coords = (result['latitude'], result['longitude'])
                    self.coord_cache[clean_pc] = coords
                    return coords
        except Exception as e:
            logging.error(f"Geocoding failed for {clean_pc}: {e}")
            
        return (None, None)
    
    
    
    
    
    def _process_datetime(self):
        """Handle datetime conversion with efficient vectorization"""
        self.jobs_df['DEPARTURE_DATETIME'] = pd.to_datetime(
            self.jobs_df['DATE'] + ' ' + self.jobs_df['DEPARTURE TIME'],
            dayfirst=True,
            errors='coerce'
        )

    

    def _precompute_distances(self):
        depot_pc = "WA119TY"
        unique_postcodes = [depot_pc] + [
            pc for pc in pd.unique(
                self.jobs_df[['COLLECTION POST CODE', 'DELIVER POST CODE']].values.ravel()
            ) if pc != depot_pc and pd.notna(pc)
        ]
        """Batch compute all distances using OSRM Table API"""
        # Get unique postcodes and their coordinates
        unique_postcodes = pd.unique(
            self.jobs_df[['COLLECTION POST CODE', 'DELIVER POST CODE']].values.ravel()
        )
        
        # Filter valid postcodes and create index mapping
        valid_postcodes = []
        self.postcode_index = {}
        for idx, pc in enumerate(unique_postcodes):
            if pd.notna(pc) and self._get_coordinates(pc) != (None, None):
                valid_postcodes.append(pc)
                self.postcode_index[pc] = idx
                
        # Build distance matrix in batches
        matrix_size = len(valid_postcodes)
        self.distance_matrix = np.full((matrix_size, matrix_size), np.nan)
        
        # OSRM batch processing (100 locations per request)
        batch_size = 100
        for i in tqdm(range(0, matrix_size, batch_size), desc="Batch Processing"):
            batch_indices = range(i, min(i+batch_size, matrix_size))
            coords = [
                self._get_coordinates(valid_postcodes[j])
                for j in batch_indices
            ]
            
            # Convert to OSRM format (lon,lat)
            osrm_coords = [f"{lon},{lat}" for lat, lon in coords]
            
            try:
                response = requests.get(
                    f"http://router.project-osrm.org/table/v1/driving/{';'.join(osrm_coords)}",
                    params={'annotations': 'distance'},
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()
                
                if data['code'] == 'Ok':
                    distances = np.array(data['distances']) / 1609.34  # Convert to miles
                    # Fill matrix slice
                    start = i
                    end = start + len(batch_indices)
                    self.distance_matrix[start:end, start:end] = distances
            except Exception as e:
                logging.error(f"OSRM batch failed: {e}")

        # Fill remaining NaN with haversine fallback
        self._fill_missing_distances(valid_postcodes)

    def _fill_missing_distances(self, postcodes: List[str]):
        """Fill missing distances with haversine calculation"""
        n = len(postcodes)
        for i in tqdm(range(n), desc="Filling missing distances"):
            for j in range(n):
                if np.isnan(self.distance_matrix[i][j]):
                    coord_i = self._get_coordinates(postcodes[i])
                    coord_j = self._get_coordinates(postcodes[j])
                    if None not in coord_i and None not in coord_j:
                        self.distance_matrix[i][j] = self._haversine(*coord_i, *coord_j)
                    else:
                        self.distance_matrix[i][j] = 0.0  # Fallback value

    def _haversine(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great-circle distance between two points"""
        R = 6371.0088  # Earth radius in kilometers
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = (sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2)
        return R * 2 * atan2(sqrt(a), sqrt(1-a)) * 0.621371  # Convert to miles

    def _calculate_loaded_miles(self):
        """Vectorized loaded miles calculation using precomputed matrix"""
        def get_distance(row):
            try:
                src = self.postcode_index.get(row['COLLECTION POST CODE'], -1)
                dst = self.postcode_index.get(row['DELIVER POST CODE'], -1)
                if src != -1 and dst != -1 and src != dst:
                    return self.distance_matrix[src][dst]
                return 0.0
            except:
                return 0.0

        self.jobs_df['LOADED MILES'] = self.jobs_df.apply(get_distance, axis=1)

    def _calculate_empty_miles(self):
        """Optimized empty miles calculation using matrix lookups"""
        driver_groups = self.jobs_df.groupby('DRIVER NAME')
        
        for driver, group in driver_groups:
            sorted_jobs = group.sort_values('DEPARTURE_DATETIME')
            prev_drop = None
            
            for idx, row in sorted_jobs.iterrows():
                if prev_drop:
                    src = self.postcode_index.get(prev_drop, -1)
                    dst = self.postcode_index.get(row['COLLECTION POST CODE'], -1)
                    if src != -1 and dst != -1:
                        self.jobs_df.at[idx, 'EMPTY MILES'] = self.distance_matrix[src][dst]
                
                prev_drop = row['DELIVER POST CODE']
                
        self.jobs_df['EMPTY MILES'].fillna(0.0, inplace=True)
        self.jobs_df['TOTAL MILES'] = self.jobs_df['LOADED MILES'] + self.jobs_df['EMPTY MILES']
        
    def get_distance_between_postcodes(self, origin: str, destination: str) -> float:
        """Get driving distance between any two postcodes"""
        return self._get_driving_distance(
            origin.strip().upper().replace(' ', ''),
            destination.strip().upper().replace(' ', '')
        )


    def _calculate_single_loaded_miles(self, origin: str, destination: str) -> float:
        """Calculate driving distance between two postcodes"""
        return self._get_driving_distance(origin, destination)




    def _calculate_single_empty_miles(self, prev_drop, current_pickup):
        """Calculate empty miles between two postcodes"""
        start = self._get_coordinates(prev_drop)
        end = self._get_coordinates(current_pickup)
        if all(start) and all(end):
            return self._haversine(*start, *end)
        return 0.0

    def get_distance_metrics(self):
        """Return comprehensive distance metrics."""
        return {
            'total_loaded': self.jobs_df['LOADED MILES'].sum(),
            'average_loaded': self.jobs_df['LOADED MILES'].mean(),
            'total_empty': self.jobs_df['EMPTY MILES'].sum(),
            'average_empty': self.jobs_df['EMPTY MILES'].mean(),
            'max_empty': self.jobs_df['EMPTY MILES'].max(),
            'min_empty': self.jobs_df['EMPTY MILES'].min()
        }

    def get_postcode_stats(self):
        return {
            'unique_collection': self.jobs_df['COLLECTION POST CODE'].nunique(),
            'unique_delivery': self.jobs_df['DELIVER POST CODE'].nunique(),
            'most_common_collection': self.jobs_df['COLLECTION POST CODE'].mode()[0],
            'most_common_delivery': self.jobs_df['DELIVER POST CODE'].mode()[0]
        }

    def add_job(self, new_job: dict):
        """Enhanced job addition with sequence optimization"""
        try:
            # Validate required fields
            clean_driver = (
                new_job['DRIVER NAME']
                .split('[')[0]
                .split('(')[0]
                .strip()
                .upper()
            )
            new_job['DRIVER NAME'] = clean_driver
            required_fields = ['DATE', 'DEPARTURE TIME', 'COLLECTION POST CODE',
                              'DELIVER POST CODE', 'DRIVER NAME', 'JOB NAME']
            for field in required_fields:
                if field not in new_job:
                    raise ValueError(f"Missing required field: {field}")

            # Calculate loaded miles
            loaded = self._get_driving_distance(
                new_job['COLLECTION POST CODE'],
                new_job['DELIVER POST CODE']
            )
            new_job['LOADED MILES'] = loaded

            # Process datetime
            new_job['DEPARTURE_DATETIME'] = pd.to_datetime(
                new_job['DATE'] + ' ' + new_job['DEPARTURE TIME'],
                dayfirst=True,
                errors='coerce'
            )

            # Calculate empty miles based on driver's last job
            driver_jobs = self.jobs_df[self.jobs_df['DRIVER NAME'] == new_job['DRIVER NAME']]
            if not driver_jobs.empty:
                last_job = driver_jobs.sort_values('DEPARTURE_DATETIME').iloc[-1]
                new_job['EMPTY MILES'] = self._get_driving_distance(
                    last_job['DELIVER POST CODE'],
                    new_job['COLLECTION POST CODE']
                )

            # Add to DataFrame
            new_df = pd.DataFrame([new_job])
            self.jobs_df = pd.concat([self.jobs_df, new_df], ignore_index=True)

            # Re-sort and optimize
            self._optimize_sequence(new_job['DRIVER NAME'])
            return True
        except Exception as e:
            logging.error(f"Job addition failed: {e}")
            return False

    def remove_job(self, job_name: str):
        """Remove job and update affected sequences"""
        try:
            # Find job and its driver
            job_idx = self.jobs_df[self.jobs_df['JOB NAME'] == job_name].index
            if len(job_idx) == 0:
                raise ValueError("Job not found")

            driver = self.jobs_df.loc[job_idx[0], 'DRIVER NAME']

            # Remove job
            self.jobs_df = self.jobs_df.drop(job_idx)

            # Re-optimize driver's sequence
            self._optimize_sequence(driver)
            return True
        except Exception as e:
            logging.error(f"Job removal failed: {e}")
            return False

    def _optimize_sequence(self, driver_name: str):
        """Re-sort and recalculate empty miles for the given driver."""
        mask = self.jobs_df['DRIVER NAME'] == driver_name
        driver_jobs = self.jobs_df[mask].sort_values('DEPARTURE_DATETIME')
        self.jobs_df = pd.concat([self.jobs_df[~mask], driver_jobs])
        self._calculate_empty_miles(driver_name=driver_name)
        
        
    def _create_haversine_matrix(self, postcodes: List[str]) -> List[List[float]]:
        size = len(postcodes)
        matrix = np.zeros((size, size))
       
    
        for  i in range(size):
            for j in range(size):
                if i != j:
                    coord_i = self._get_coordinates(postcodes[i])
                    coord_j = self._get_coordinates(postcodes[j])
                    matrix[i][j] = self._haversine(*coord_i, *coord_j)
    
        return matrix.tolist()
    
    def _validate_input_data(self):
        initial_count = len(self.jobs_df)
    
    # Check both collection and delivery postcodes
        valid_mask = (
            self.jobs_df['COLLECTION POST CODE'].notna() &
            self.jobs_df['DELIVER POST CODE'].notna()
        )
    
        self.jobs_df = self.jobs_df[valid_mask].copy()
    
        removed_count = initial_count - len(self.jobs_df)
        if removed_count > 0:
            logging.warning(f"Removed {removed_count} jobs with invalid postcodes")