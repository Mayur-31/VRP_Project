# ai_helper.py

import os
import requests
import logging
from dotenv import load_dotenv
from typing import Dict, List

load_dotenv('Test.env')

class AIHelper:
    def __init__(self):
        self.api_key = os.getenv('OPEN_ROUTER_API_KEY')
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.distance_cache = {}

    def get_distance(self, origin: str, destination: str) -> float:
        """Get distance between postcodes using OpenRouter API"""
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

            distance = float(response.json()['choices'][0]['message']['content'].strip())
            self.distance_cache[cache_key] = distance
            return distance
        except Exception as e:
            logging.error(f"Distance API failed: {e}")
            return 10.0  # Fallback value

    def build_distance_matrix(self, postcodes: List[str]) -> List[List[float]]:
        """Optimized matrix using pre-calculated distances"""
        coord_map = {pc: self._get_cached_coords(pc) for pc in postcodes}

        matrix = []
        for i, origin in enumerate(postcodes):
            row = []
            for j, dest in enumerate(postcodes):
                if i == j:
                    row.append(0.0)
                else:
                    row.append(self._fast_distance(coord_map[origin], coord_map[dest]))
            matrix.append(row)
        return matrix