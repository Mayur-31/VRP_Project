#distance_analyzer.py


from math import radians, sin, cos, sqrt, atan2
# distance_analyzer.py (Corrected Version)
import pandas as pd
import requests
import logging
import time
from tqdm import tqdm
from math import radians, sin, cos, sqrt, atan2

# Configuration
OPENCAGE_API_KEY = '2af745c444854cb8ae05e3089905a93a'
POSTCODE_OVERRIDES = {
    'BD112BZ': (53.758755, -1.689026),
    'WA119TY': (53.476785, -2.666254)
}

class DistanceAnalyzer:
    def __init__(self, jobs_path='data/Planning NEW TEST - Sheet7.csv'):
        self.jobs_df = pd.read_csv(jobs_path)
        self._clean_postcodes()
        self._process_datetime()
        self.coord_cache = {}
        self._calculate_loaded_miles()
        # Changed empty miles calculation approach
        self._calculate_empty_miles()  

    def _clean_postcodes(self):
        """Clean postcode columns"""
        for col in ['COLLECTION POST CODE', 'DELIVER POST CODE']:
            self.jobs_df[col] = self.jobs_df[col].astype(str).str.replace(' ', '').str.upper()

    def _process_datetime(self):
        """Process datetime for sorting"""
        self.jobs_df['DEPARTURE_DATETIME'] = pd.to_datetime(
            self.jobs_df['DATE'] + ' ' + self.jobs_df['DEPARTURE TIME'],
            dayfirst=True,
            errors='coerce'
        )

    def _get_coordinates(self, postcode: str) -> tuple:
        """Get coordinates with caching"""
        if postcode in self.coord_cache:
            return self.coord_cache[postcode]
        
        if postcode in POSTCODE_OVERRIDES:
            self.coord_cache[postcode] = POSTCODE_OVERRIDES[postcode]
            return self.coord_cache[postcode]

        try:
            # Try UK postcode first
            response = requests.get(f"https://api.postcodes.io/postcodes/{postcode}", timeout=5)
            if response.status_code == 200:
                data = response.json()
                coords = (data['result']['latitude'], data['result']['longitude'])
                self.coord_cache[postcode] = coords
                return coords
        except:
            pass

        try:
            # Fallback to OpenCage
            params = {'q': postcode, 'key': OPENCAGE_API_KEY, 'limit': 1}
            response = requests.get("https://api.opencagedata.com/geocode/v1/json", params=params, timeout=5)
            if response.status_code == 200 and response.json()['results']:
                result = response.json()['results'][0]['geometry']
                coords = (result['lat'], result['lng'])
                self.coord_cache[postcode] = coords
                return coords
        except Exception as e:
            logging.error(f"Geocoding error: {str(e)}")

        return (None, None)

    def _get_driving_distance(self, origin: str, destination: str) -> float:
        """Improved distance calculation with coordinate validation"""
        if origin == destination:
            return 0.0

        start_coords = self._get_coordinates(origin)
        end_coords = self._get_coordinates(destination)
        
        if None in start_coords + end_coords:
            return 0.0

        for attempt in range(3):
            try:
                url = f"http://router.project-osrm.org/route/v1/driving/" \
                      f"{start_coords[1]},{start_coords[0]};{end_coords[1]},{end_coords[0]}?overview=false"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('code') == 'Ok':
                        return data['routes'][0]['distance'] / 1609.34  # meters to miles
                time.sleep(2 ** attempt)
            except Exception as e:
                logging.error(f"Routing error: {str(e)}")
                time.sleep(2 ** attempt)
        
        return 0.0

    def _calculate_loaded_miles(self):
        """Calculate loaded miles between collection and delivery"""
        self.jobs_df['LOADED MILES'] = [
            self._get_driving_distance(row['COLLECTION POST CODE'], row['DELIVER POST CODE'])
            for _, row in tqdm(self.jobs_df.iterrows(), desc="Calculating Loaded Miles")
        ]

    def _calculate_empty_miles(self, driver_name: str = None):
        self.jobs_df['EMPTY MILES'] = 0.0
        self.jobs_df['TOTAL MILES'] = self.jobs_df['LOADED MILES']

        for driver, group in self.jobs_df.groupby('DRIVER NAME'):
            if driver_name and driver != driver_name:
                continue
            
            sorted_jobs = group.sort_values('DEPARTURE_DATETIME')
            prev_delivery = None
            prev_end_time = None

            for idx, row in sorted_jobs.iterrows():
                if prev_delivery and pd.notnull(row['DEPARTURE_DATETIME']):
                # Calculate actual road distance
                    empty = self._get_driving_distance(
                        prev_delivery,
                        row['COLLECTION POST CODE']
                    )
                    self.jobs_df.at[idx, 'EMPTY MILES'] = empty
                    self.jobs_df.at[idx, 'TOTAL MILES'] = row['LOADED MILES'] + empty

                prev_delivery = row['DELIVER POST CODE']
                prev_end_time = row['ARRIVAL TIME'] if pd.notnull(row['ARRIVAL TIME']) else row['DEPARTURE_DATETIME']
            
           
            
            # Calculate total miles
            self.jobs_df.at[idx, 'TOTAL MILES'] = (
                self.jobs_df.at[idx, 'LOADED MILES'] + 
                self.jobs_df.at[idx, 'EMPTY MILES']
            )

    # Keep other existing methods below...




   
    
    
        
        
    
    

    def _haversine(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two coordinates"""
        R = 3958.8  # Earth radius in miles
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
        return R * 2 * atan2(sqrt(a), sqrt(1-a))

    def _calculate_distances(self):
        """Calculate distances for all jobs"""
        distances = []
        for _, row in self.jobs_df.iterrows():
            start = self._get_coordinates(row['COLLECTION POST CODE'])
            end = self._get_coordinates(row['DELIVER POST CODE'])

            if all(start) and all(end):
                distances.append(self._haversine(*start, *end))
            else:
                distances.append(None)

        self.jobs_df['DISTANCE (MILES)'] = distances

    def get_distance_metrics(self):
        """Return calculated distance metrics"""
        return {
            'total_miles': self.jobs_df['DISTANCE (MILES)'].sum(),
            'average_miles': self.jobs_df['DISTANCE (MILES)'].mean(),
            'max_miles': self.jobs_df['DISTANCE (MILES)'].max(),
            'min_miles': self.jobs_df['DISTANCE (MILES)'].min()
        }
    
    def get_distance_metrics(self):
        """Return comprehensive distance metrics"""
        return {
            'total_loaded': self.jobs_df['LOADED MILES'].sum(),
            'average_loaded': self.jobs_df['LOADED MILES'].mean(),
            'total_empty': self.jobs_df['EMPTY MILES'].sum(),
            'average_empty': self.jobs_df['EMPTY MILES'].mean(),
            'max_empty': self.jobs_df['EMPTY MILES'].max(),
            'min_empty': self.jobs_df['EMPTY MILES'].min()
        }
        
    def _calculate_distances(self):
        """Calculate loaded distances for all jobs"""
        self.jobs_df['LOADED MILES'] = self.jobs_df.apply(
            lambda row: self._calculate_loaded_miles(row), axis=1
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
    
    def get_distance_between_postcodes(self, origin: str, destination: str) -> float:
        """Get driving distance between any two postcodes"""
        return self._get_driving_distance(
            origin.strip().upper().replace(' ', ''),
            destination.strip().upper().replace(' ', '')
        )
    def get_postcode_stats(self):
        """Return postcode-related statistics"""
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
        driver_mask = self.jobs_df['DRIVER NAME'] == driver_name
        driver_jobs = self.jobs_df[driver_mask].sort_values('DEPARTURE_DATETIME')
    
    # Remove and reinsert jobs to maintain sequence
        self.jobs_df = self.jobs_df[~driver_mask]
        self.jobs_df = pd.concat([self.jobs_df, driver_jobs])
    
        self._calculate_empty_miles(driver_name)

    