import csv
import os
from datetime import datetime
from typing import List, Dict, Set, Optional
import pandas as pd

class LocationTracker:
    """Track and manage Ofer's visited locations in a CSV file."""
    
    def __init__(self, csv_path: str = "./data/ofer_locations.csv"):
        self.csv_path = csv_path
        self.ensure_csv_exists()
    
    def ensure_csv_exists(self):
        """Create CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.csv_path):
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            
            # Create CSV with headers
            headers = [
                'location_name',
                'country', 
                'city',
                'latitude',
                'longitude',
                'first_visit_date',
                'last_visit_date',
                'visit_count',
                'photo_filenames',
                'source',
                'confidence'
            ]
            
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(headers)
            
            print(f"Created location tracking CSV: {self.csv_path}")
    
    def add_location(self, location_data: Dict, photo_filename: str = "", visit_date: str = ""):
        """Add or update a location in the CSV."""
        if not location_data.get('location_name'):
            return
        
        # Read existing data
        existing_locations = self.read_locations()
        location_name = location_data['location_name']
        
        # Parse location parts
        city, country = self._parse_location_name(location_name)
        
        # Check if location already exists
        existing_location = None
        for loc in existing_locations:
            if loc['location_name'].lower() == location_name.lower():
                existing_location = loc
                break
        
        if existing_location:
            # Update existing location
            existing_location['last_visit_date'] = visit_date or existing_location['last_visit_date']
            existing_location['visit_count'] = str(int(existing_location['visit_count']) + 1)
            
            # Add photo filename if not already included
            photos = existing_location['photo_filenames'].split(';') if existing_location['photo_filenames'] else []
            if photo_filename and photo_filename not in photos:
                photos.append(photo_filename)
                existing_location['photo_filenames'] = ';'.join(photos)
        else:
            # Create new location entry
            new_location = {
                'location_name': location_name,
                'country': country,
                'city': city,
                'latitude': location_data.get('latitude', ''),
                'longitude': location_data.get('longitude', ''),
                'first_visit_date': visit_date,
                'last_visit_date': visit_date,
                'visit_count': '1',
                'photo_filenames': photo_filename,
                'source': location_data.get('source', 'gps'),
                'confidence': location_data.get('confidence', 'high')
            }
            existing_locations.append(new_location)
        
        # Write back to CSV
        self._write_locations(existing_locations)
    
    def _parse_location_name(self, location_name: str) -> tuple:
        """Parse location name into city and country."""
        if ', ' in location_name:
            parts = location_name.split(', ')
            if len(parts) >= 2:
                return parts[0].strip(), parts[1].strip()
        
        # If no comma, assume it's just a country/city
        return "", location_name.strip()
    
    def read_locations(self) -> List[Dict]:
        """Read all locations from CSV."""
        locations = []
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                locations = list(reader)
        except Exception as e:
            print(f"Error reading locations CSV: {e}")
        
        return locations
    
    def _write_locations(self, locations: List[Dict]):
        """Write locations to CSV."""
        if not locations:
            return
        
        headers = list(locations[0].keys())
        
        try:
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=headers)
                writer.writeheader()
                writer.writerows(locations)
        except Exception as e:
            print(f"Error writing locations CSV: {e}")
    
    def get_location_summary(self) -> Dict:
        """Get summary statistics about visited locations."""
        locations = self.read_locations()
        
        if not locations:
            return {
                'total_locations': 0,
                'countries': 0,
                'cities': 0,
                'most_visited': None,
                'recent_visit': None
            }
        
        # Count unique countries and cities
        countries = set()
        cities = set()
        
        for loc in locations:
            if loc['country']:
                countries.add(loc['country'])
            if loc['city']:
                cities.add(loc['city'])
        
        # Find most visited location
        most_visited = max(locations, key=lambda x: int(x['visit_count']) if x['visit_count'] else 0)
        
        # Find most recent visit
        recent_visit = None
        for loc in locations:
            if loc['last_visit_date']:
                if not recent_visit or loc['last_visit_date'] > recent_visit['last_visit_date']:
                    recent_visit = loc
        
        return {
            'total_locations': len(locations),
            'countries': len(countries),
            'cities': len(cities),
            'most_visited': most_visited,
            'recent_visit': recent_visit
        }
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """Export locations to pandas DataFrame for analysis."""
        locations = self.read_locations()
        return pd.DataFrame(locations)
    
    def get_locations_by_country(self, country: str) -> List[Dict]:
        """Get all locations visited in a specific country."""
        locations = self.read_locations()
        return [loc for loc in locations if loc['country'].lower() == country.lower()]
    
    def search_locations(self, query: str) -> List[Dict]:
        """Search locations by name, city, or country."""
        locations = self.read_locations()
        query_lower = query.lower()
        
        matches = []
        for loc in locations:
            if (query_lower in loc['location_name'].lower() or 
                query_lower in loc['city'].lower() or 
                query_lower in loc['country'].lower()):
                matches.append(loc)
        
        return matches
