#ai_helper.py
import os
import requests
import numpy as np
import logging
from dotenv import load_dotenv
from typing import Dict, List
from tqdm import tqdm
import pandas as pd
import re

load_dotenv('Test.env')

# Use the same postcode overrides as in your distance_analyzer.py
POSTCODE_OVERRIDES = {
    'BD112BZ': (53.758755, -1.689026),
    'WA119TY': (53.476785, -2.666254)
}

class AIHelper:
    def __init__(self):
        self.api_key = os.getenv('OPEN_ROUTER_API_KEY')
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.distance_cache = {}
        # For OSRM Table API, cache coordinates for postcodes
        self.coords_cache = {}
        self.osrm_url = "http://router.project-osrm.org/table/v1/driving/"
        self.max_batch_size = 50
    def _get_coords(self, postcode: str) -> tuple:
        """Get coordinates for a postcode, using overrides, cache, and fallback to postcodes.io."""
        postcode = postcode.replace(" ", "").upper()
        # Check for overrides first
        if pd.isna(postcode) or not re.match(r'^[A-Z]{1,2}\d[A-Z\d]? ?\d[A-Z]{2}$', postcode):
            return (None, None)
        if postcode in POSTCODE_OVERRIDES:
            return POSTCODE_OVERRIDES[postcode]
        if postcode in self.coords_cache:
            return self.coords_cache[postcode]
        try:
            response = requests.get(f"https://api.postcodes.io/postcodes/{postcode}", timeout=5)
            response.raise_for_status()
            data = response.json()
            coords = (data['result']['latitude'], data['result']['longitude'])
            self.coords_cache[postcode] = coords
            return coords
        except Exception as e:
            logging.error(f"Error fetching coordinates for {postcode}: {e}")
        return (None, None)

    def get_distance(self, origin: str, destination: str) -> float:
        """Get distance between two postcodes using OpenRouter API (fallback method)."""
        cache_key = (origin, destination)
        if cache_key in self.distance_cache:
            return self.distance_cache[cache_key]
        try:
            prompt = f"Calculate road distance in miles between {origin} and {destination} UK postcodes. Respond only with the number."
            payload = {
                "model": "deepseek/deepseek-r1:free",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1
            }
            response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=10)
            response.raise_for_status()
            json_data = response.json()
            if 'choices' not in json_data:
                raise KeyError(f"Key 'choices' not found in response. Full response: {json_data}")
            distance_str = json_data['choices'][0]['message']['content'].strip()
            distance = float(distance_str)
            self.distance_cache[cache_key] = distance
            return distance
        except Exception as e:
            logging.error(f"Distance API failed: {e}")
            return 10.0  # Fallback value

    def build_distance_matrix_osrm(self, postcodes: List[str]) -> List[List[float]]:
        valid_postcodes = [
            pc for pc in postcodes 
            if re.match(r'^[A-Z]{1,2}\d[A-Z\d]? ?\d[A-Z]{2}$', pc)
        ]
        if len(valid_postcodes) < 2:
            logging.error("Insufficient valid postcodes for routing")
            return []
    
        logging.info(f"Building matrix for {len(valid_postcodes)} postcodes")
    
        if len(valid_postcodes) > 100:
            logging.warning("Large matrix size may cause memory issues. Consider subsetting data.")
        depot = "WA119TY"
        unique_pcs = [depot] + [pc for pc in postcodes if pc != depot]
    
    # Get valid coordinates with proper index mapping
        coords, valid_indices = self._get_valid_coordinates(unique_pcs)
        size = len(unique_pcs)
    
    # Initialize matrix with NaN to detect unprocessed pairs
        matrix = np.full((size, size), np.nan)
        np.fill_diagonal(matrix, 0.0)  # Zero diagonals
    
    # Process in valid coordinate batches
        for batch in self._batch_coordinates(coords, valid_indices):
            self._process_osrm_batch(batch, matrix)
    
    # Fill remaining NaNs with Haversine fallback
        self._fill_matrix_with_haversine(matrix, unique_pcs, valid_indices)
    
        return matrix.tolist()

    def _fill_matrix_with_haversine(self, matrix, postcodes, valid_indices):
    
        n = len(postcodes)
        for i in range(n):
            for j in range(n):
                if np.isnan(matrix[i][j]):
                    coord_i = self._get_valid_coordinates(postcodes[i])
                    coord_j = self._get_valid_coordinates(postcodes[j])
                    if coord_i[0] and coord_i[1] and coord_j[0] and coord_j[1]:
                        matrix[i][j] = self._haversine(*coord_i, *coord_j)
                    else:
                        matrix[i][j] = 100.0  # Only use 100 as true fallback
    # Fallback method: builds matrix using individual API calls.
    def build_distance_matrix(self, postcodes: List[str]) -> List[List[float]]:
        size = len(postcodes)
        matrix = [[0.0 for _ in range(size)] for _ in range(size)]
    
        for i in tqdm(range(size), desc="Building fallback matrix"):
            for j in range(size):
                if i != j:
                    matrix[i][j] = self.get_distance(postcodes[i], postcodes[j])
    
        return matrix
    def _get_valid_coordinates(self, postcodes):
        valid_coords = []
        valid_indices = []
        for idx, pc in enumerate(postcodes):
            coords = self._get_coords(pc)
            if coords[0] and coords[1]:
                valid_coords.append((coords[1], coords[0]))  # (lon, lat)
                valid_indices.append(idx)
        return valid_coords, valid_indices

    def _batch_coordinates(self, coords, indices):
        """Generate coordinate batches"""
        for i in range(0, len(coords), self.max_batch_size):
            yield {
                "coords": coords[i:i+self.max_batch_size],
                "indices": indices[i:i+self.max_batch_size]
            }

    def _process_osrm_batch(self, batch, matrix):
        coord_str = ";".join([f"{lon},{lat}" for (lon, lat) in batch["coords"]])
        try:
            response = requests.get(
                f"{self.osrm_url}{coord_str}?annotations=distance",
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
        
            if data.get("code") == "Ok" and "distances" in data:
                distances = np.array(data["distances"]) / 1609.34  # meters to miles
                batch_size = len(batch["indices"])
            
            # Map OSRM response to correct matrix indices
                for src_idx, src_row in zip(batch["indices"], distances):
                    for dst_idx, distance in zip(batch["indices"], src_row):
                        if not np.isnan(distance):
                            matrix[src_idx][dst_idx] = distance
            else:
                logging.warning(f"OSRM partial failure: {data.get('message')}")
            
        except Exception as e:
            logging.error(f"OSRM batch failed: {str(e)}")