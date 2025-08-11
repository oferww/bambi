import os
import exifread
from PIL import Image
from datetime import datetime
from typing import Dict, Optional, Tuple
import json
from geopy.geocoders import Nominatim
from googletrans import Translator
from google.cloud import vision
import piexif
import piexif.helper
from .exif_utils import write_gps
import subprocess
# import easyocr          # Commented out - too heavy for Docker build
# import cv2              # Commented out - too heavy for Docker build  
# import numpy as np      # Commented out - too heavy for Docker build

class PhotoProcessor:
    """Process photos and extract metadata for RAG system."""
    
    def __init__(self, photos_dir: str = "./data/uploads/photos"):
        self.photos_dir = photos_dir
        os.makedirs(photos_dir, exist_ok=True)
        self.geolocator = Nominatim(user_agent="ofergpt_chatbot")
        self.translator = Translator()
        # Initialize OCR reader (lazy loading to avoid startup delay) - COMMENTED OUT
        # self._ocr_reader = None
    
    def extract_metadata(self, image_path: str) -> Dict:
        """Extract metadata from a photo including EXIF data."""
        metadata = {
            "filename": os.path.basename(image_path),
            "file_path": image_path,
            "file_size": os.path.getsize(image_path),
            "date_taken": None,
            "location": None,
            "camera_info": {},
            "description": ""
        }
        
        try:
            # Open image with PIL for basic info
            with Image.open(image_path) as img:
                metadata["dimensions"] = img.size
                metadata["format"] = img.format
                metadata["mode"] = img.mode
            
            # Extract EXIF data
            with open(image_path, 'rb') as f:
                tags = exifread.process_file(f)
            
            # Extract date taken
            if 'EXIF DateTimeOriginal' in tags:
                date_str = str(tags['EXIF DateTimeOriginal'])
                try:
                    metadata["date_taken"] = datetime.strptime(date_str, '%Y:%m:%d %H:%M:%S').isoformat()
                except ValueError:
                    pass
            
            # Extract GPS coordinates
            if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
                print(f"[DEBUG] GPS data found in {os.path.basename(image_path)}", flush=True)
                lat = self._convert_to_degrees(tags['GPS GPSLatitude'].values)
                lon = self._convert_to_degrees(tags['GPS GPSLongitude'].values)
                
                if tags['GPS GPSLatitudeRef'].values == 'S':
                    lat = -lat
                if tags['GPS GPSLongitudeRef'].values == 'W':
                    lon = -lon
                
                print(f"[DEBUG] Converted GPS coordinates: lat={lat}, lon={lon}", flush=True)
                
                location_name = self._get_location_name(lat, lon)
                metadata["location"] = {
                    "latitude": lat,
                    "longitude": lon,
                    "coordinates": f"{lat}, {lon}",
                    "location_name": location_name
                }
                print(f"[DEBUG] Location set to: {location_name}", flush=True)
            else:
                print(f"[DEBUG] No GPS data found in {os.path.basename(image_path)}", flush=True)
                print(f"[DEBUG] Available EXIF tags: {list(tags.keys())}", flush=True)

                # Try Google Vision landmark detection
                lat, lon, loc_name = self._enrich_location_with_vision(image_path)
                if loc_name:
                    metadata["location"] = {
                        "latitude": lat,
                        "longitude": lon,
                        "coordinates": f"{lat}, {lon}",
                        "location_name": loc_name,
                        "source": "google_vision"
                    }
                    print(f"[DEBUG] Google Vision found location: {loc_name} (lat={lat}, lon={lon})", flush=True)
                else:
                    # Vision failed – signal caller to skip this photo
                    print(f"[WARN] Vision could not determine location for {os.path.basename(image_path)}. Rejecting photo.", flush=True)
                    return None
            
            # Extract camera info
            if 'Image Model' in tags:
                metadata["camera_info"]["model"] = str(tags['Image Model'])
            if 'Image Make' in tags:
                metadata["camera_info"]["make"] = str(tags['Image Make'])
            if 'EXIF ExposureTime' in tags:
                metadata["camera_info"]["exposure"] = str(tags['EXIF ExposureTime'])
            if 'EXIF FNumber' in tags:
                metadata["camera_info"]["f_number"] = str(tags['EXIF FNumber'])
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
        
        # Image analysis disabled for lighter Docker build - requires opencv-python and easyocr
        # if not metadata.get("location"):
        #     print(f"No GPS data found for {os.path.basename(image_path)}, analyzing image content...")
        #     try:
        #         # Extract text from image
        #         extracted_text = self._extract_text_from_image(image_path)
        #         if extracted_text:
        #             print(f"Extracted text: {extracted_text[:100]}...")  # Show first 100 chars
        #             metadata["extracted_text"] = extracted_text
        #             
        #             # Analyze for location
        #             detected_location = self._analyze_image_for_location(image_path, extracted_text)
        #             if detected_location:
        #                 print(f"Detected location from image: {detected_location}")
        #                 metadata["location"] = {
        #                     "location_name": detected_location,
        #                     "source": "image_analysis",
        #                     "confidence": "medium"
        #                 }
        #         else:
        #             print("No text detected in image")
        #     except Exception as e:
        #         print(f"Error analyzing image content: {e}")
        
        return metadata
    
    def _convert_to_degrees(self, values) -> float:
        """Convert GPS coordinates to decimal degrees."""
        d = float(values[0].num) / float(values[0].den)
        m = float(values[1].num) / float(values[1].den)
        s = float(values[2].num) / float(values[2].den)
        return d + (m / 60.0) + (s / 3600.0)
    
    def _get_location_name(self, latitude: float, longitude: float) -> str:
        """Convert coordinates to city, country name in English."""
        try:
            print(f"[DEBUG] _get_location_name called with lat={latitude}, lon={longitude}", flush=True)
            # Force English language for location names
            location = self.geolocator.reverse(f"{latitude}, {longitude}", language='en')
            if location:
                print(f"[DEBUG] geolocator.reverse returned: {location}", flush=True)
                print(f"[DEBUG] location.raw: {getattr(location, 'raw', None)}", flush=True)
                address = location.raw.get('address', {})
                print(f"[DEBUG] address dict: {address}", flush=True)
                
                # Try multiple address fields for city name
                city = (address.get('city') or 
                       address.get('town') or 
                       address.get('village') or 
                       address.get('suburb') or
                       address.get('municipality') or
                       address.get('county') or
                       address.get('state'))
                
                country = address.get('country')
                
                print(f"[DEBUG] City: {city}", flush=True)
                print(f"[DEBUG] Country: {country}", flush=True)
                
                if city and country:
                    # Create location name and always attempt translation
                    location_name = f"{city}, {country}"
                    translated_name = self._translate_to_english(location_name)
                    print(f"  ✅ Using: {translated_name}", flush=True)
                    return translated_name
                elif country:
                    # If only country is available, translate it
                    translated_country = self._translate_to_english(country)
                    print(f"  ✅ Using country only: {translated_country}", flush=True)
                    return translated_country
                else:
                    print(f"  ⚠️ No city/country found, using coordinates", flush=True)
                    return f"{latitude:.4f}, {longitude:.4f}"
            else:
                print(f"  ❌ No geocoding result for {latitude}, {longitude}, using coordinates", flush=True)
                return f"{latitude:.4f}, {longitude:.4f}"
        except Exception as e:
            print(f"[ERROR] Exception in _get_location_name for {latitude}, {longitude}: {e}", flush=True)
            return f"{latitude:.4f}, {longitude:.4f}"
    

    
    def _translate_to_english(self, location_name: str) -> str:
        """Translate location names to English using Google Translate service."""
        try:
            # Skip translation if already in English or contains coordinates
            if self._is_english_text(location_name) or ',' in location_name and any(c.isdigit() for c in location_name):
                return location_name
            
            print(f"[TRANSLATE] Translating: {location_name}", flush=True)
            
            # Use Google Translate to translate to English
            translation = self.translator.translate(location_name, dest='en')
            translated_text = translation.text
            
            print(f"[TRANSLATE] Result: {location_name} -> {translated_text}", flush=True)
            return translated_text
            
        except Exception as e:
            print(f"[TRANSLATE] Error translating '{location_name}': {e}", flush=True)
            # Try fallback translation method
            return self._fallback_translate(location_name)
    
    def _fallback_translate(self, location_name: str) -> str:
        """Fallback translation method using simple heuristics."""
        try:
            # Common location name patterns and their English equivalents
            # This is a minimal fallback for when Google Translate fails
            fallback_translations = {
                # Vietnamese
                'Đà Nẵng': 'Da Nang',
                'Hà Nội': 'Hanoi', 
                'Hồ Chí Minh': 'Ho Chi Minh City',
                'Việt Nam': 'Vietnam',
                # Chinese
                '北京': 'Beijing',
                '上海': 'Shanghai',
                '中国': 'China',
                # Japanese
                '東京': 'Tokyo',
                '日本': 'Japan',
                # Korean
                '서울': 'Seoul',
                '한국': 'South Korea',
                # Thai
                'กรุงเทพฯ': 'Bangkok',
                'ประเทศไทย': 'Thailand',
                # Arabic
                'القاهرة': 'Cairo',
                'مصر': 'Egypt',
                # Russian
                'Москва': 'Moscow',
                'Россия': 'Russia',
            }
            
            # Check for exact matches
            if location_name in fallback_translations:
                return fallback_translations[location_name]
            
            # Check for partial matches
            for original, english in fallback_translations.items():
                if original in location_name:
                    return location_name.replace(original, english)
            
            # If no fallback translation found, return original
            return location_name
            
        except Exception as e:
            print(f"[FALLBACK] Error in fallback translation: {e}", flush=True)
            return location_name
    
    def _is_english_text(self, text: str) -> bool:
        """Check if text appears to be in English."""
        try:
            # Simple heuristic: check if text contains mostly ASCII characters
            # and common English words/patterns
            english_indicators = [
                'the', 'and', 'of', 'in', 'to', 'a', 'is', 'that', 'it', 'with',
                'for', 'as', 'was', 'his', 'they', 'at', 'be', 'this', 'have',
                'from', 'or', 'one', 'had', 'by', 'word', 'but', 'not', 'what',
                'all', 'were', 'we', 'when', 'your', 'can', 'said', 'there',
                'use', 'an', 'each', 'which', 'she', 'do', 'how', 'their', 'if'
            ]
            
            # Check if text contains mostly ASCII characters
            ascii_ratio = sum(1 for c in text if ord(c) < 128) / len(text) if text else 0
            
            # Check for common English words
            text_lower = text.lower()
            english_word_count = sum(1 for word in english_indicators if word in text_lower)
            
            # Consider it English if mostly ASCII and contains some English words
            return ascii_ratio > 0.8 and english_word_count > 0
            
        except Exception:
            return False
    
    # Vision methods commented out for lighter Docker build
    # @property
    # def ocr_reader(self):
    #     """Lazy initialization of OCR reader to avoid startup delays."""
    #     if self._ocr_reader is None:
    #         print("Initializing OCR reader...")
    #         self._ocr_reader = easyocr.Reader(['en'])
    #     return self._ocr_reader
    
    # def _extract_text_from_image(self, image_path: str) -> str:
    #     """Extract text from image using OCR."""
    #     try:
    #         # Read the image
    #         img = cv2.imread(image_path)
    #         if img is None:
    #             return ""
    #         
    #         # Use EasyOCR to extract text
    #         results = self.ocr_reader.readtext(img)
    #         
    #         # Combine all detected text
    #         extracted_text = []
    #         for (bbox, text, confidence) in results:
    #             # Only include text with reasonable confidence
    #             if confidence > 0.3:
    #                 extracted_text.append(text)
    #         
    #         return " ".join(extracted_text)
    #     except Exception as e:
    #         print(f"Error extracting text from {image_path}: {e}")
    #         return ""
    # 
    # def _analyze_image_for_location(self, image_path: str, extracted_text: str) -> Optional[str]:
    #     """Analyze image content to guess location."""
    #     try:
    #         # Look for location-related keywords in extracted text
    #         location_keywords = [
    #             # Famous landmarks
    #             'eiffel tower', 'statue of liberty', 'big ben', 'colosseum', 'pyramids',
    #             'times square', 'golden gate', 'sydney opera', 'taj mahal', 'mount fuji',
    #             'empire state', 'london bridge', 'brooklyn bridge', 'hollywood sign',
    #             'christ redeemer', 'machu picchu', 'stonehenge', 'acropolis',
    #             
    #             # Cities
    #             'new york', 'london', 'paris', 'tokyo', 'sydney', 'rome', 'berlin',
    #             'madrid', 'barcelona', 'amsterdam', 'vienna', 'prague', 'budapest',
    #             'florence', 'venice', 'milan', 'munich', 'dublin', 'edinburgh',
    #             'copenhagen', 'stockholm', 'oslo', 'helsinki', 'warsaw', 'moscow',
    #             'beijing', 'shanghai', 'hong kong', 'singapore', 'bangkok', 'manila',
    #             'jakarta', 'kuala lumpur', 'delhi', 'mumbai', 'bangalore', 'chennai',
    #             'tel aviv', 'jerusalem', 'haifa', 'eilat', 'jerusalem', 'acre',
    #             
    #             # Countries
    #             'usa', 'united states', 'uk', 'england', 'france', 'germany', 'italy',
    #             'spain', 'japan', 'china', 'australia', 'canada', 'israel', 'thailand',
    #             'singapore', 'malaysia', 'indonesia', 'philippines', 'india',
    #             
    #             # Common place indicators
    #             'airport', 'station', 'university', 'museum', 'hotel', 'restaurant',
    #             'cafe', 'park', 'beach', 'mountain', 'lake', 'river', 'bridge',
    #             'street', 'avenue', 'boulevard', 'plaza', 'square', 'center'
    #         ]
    #         
    #         # Convert extracted text to lowercase for comparison
    #         text_lower = extracted_text.lower()
    #         
    #         # Find location matches
    #         detected_locations = []
    #         for keyword in location_keywords:
    #             if keyword in text_lower:
    #                 detected_locations.append(keyword.title())
    #         
    #         if detected_locations:
    #             # Return the first detected location
    #             return detected_locations[0]
    #         
    #         # Look for patterns that might indicate locations
    #         # e.g., signs, street names, etc.
    #         words = text_lower.split()
    #         for i, word in enumerate(words):
    #             # Look for words followed by common location suffixes
    #             if word in ['st.', 'street', 'ave.', 'avenue', 'rd.', 'road', 'blvd.', 'boulevard']:
    #                 if i > 0:
    #                     return f"{words[i-1].title()} {word.title()}"
    #             
    #             # Look for airport codes (3 uppercase letters)
    #             if len(word) == 3 and word.isalpha():
    #                 return f"{word.upper()} Airport"
    #         
    #         return None
    #         
    #     except Exception as e:
    #         print(f"Error analyzing image for location: {e}")
    #         return None
    
    def _enrich_location_with_vision(self, image_path: str):
        """Use Google Vision Landmark detection to derive location.

        Returns (lat, lon, location_name) or (None, None, "") if not found.
        """
        try:
            client = vision.ImageAnnotatorClient()
            with open(image_path, "rb") as f:
                img = vision.Image(content=f.read())
            response = client.landmark_detection(image=img)
            anns = response.landmark_annotations
            if not anns:
                return None, None, ""
            best = max(anns, key=lambda a: a.score)
            if not best.locations:
                return None, None, ""
            lat_lng = best.locations[0].lat_lng
            lat, lon = lat_lng.latitude, lat_lng.longitude
            # Derive city, country via reverse geocoding for user-friendly location name
            loc_name = self._get_location_name(lat, lon)
            # write to EXIF for permanence
            try:
                write_gps(image_path, lat, lon)
            except Exception as ex:
                print(f"[WARN] Could not write GPS to EXIF: {ex}", flush=True)
            return lat, lon, loc_name
        except Exception as e:
            print(f"[ERROR] Vision API failed: {e}", flush=True)
            return None, None, ""

    def process_photos_directory(self) -> Dict[str, Dict]:
        """Process all photos in the directory and return metadata."""
        photos_metadata = {}
        
        print(f"[DEBUG] Processing photos in directory: {self.photos_dir}", flush=True)
        for filename in os.listdir(self.photos_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                print(f"[DEBUG] Processing photo: {filename}", flush=True)
                file_path = os.path.join(self.photos_dir, filename)
                metadata = self.extract_metadata(file_path)
                photos_metadata[filename] = metadata
                print(f"[DEBUG] Finished processing: {filename}", flush=True)
        
        print(f"[DEBUG] Total photos processed: {len(photos_metadata)}", flush=True)
        return photos_metadata
    
    def create_photo_descriptions(self, photos_metadata: Dict[str, Dict]) -> list:
        """Create natural language descriptions for each photo, omitting technical details and using friendly dates."""
        descriptions = []
        for filename, metadata in photos_metadata.items():
            description = "Ofer took a photo"

            # Format date in a human-friendly way
            date_taken = metadata.get("date_taken", "")
            if date_taken:
                try:
                    dt = datetime.fromisoformat(date_taken)
                    friendly_date = dt.strftime("%B %d, %Y")  # e.g., May 13, 2010
                    description += f" on {friendly_date}"
                except Exception:
                    description += f" on {date_taken}"

            # Add location if available
            if metadata.get("location"):
                loc = metadata["location"]
                if loc.get("location_name"):
                    description += f" in {loc['location_name']}"
                elif loc.get("coordinates"):
                    description += f" at coordinates {loc['coordinates']}"

            # Complete the sentence
            description += "."

            # Add any additional context without technical details
            if metadata.get("dimensions"):
                width, height = metadata['dimensions']
                if width > 3000 or height > 3000:
                    description += " This was a high-resolution photo."

            descriptions.append({
                "content": description.strip(),
                "metadata": {
                    "filename": filename,  # Include filename for identification
                    "date_taken": metadata.get("date_taken"),
                    "location": metadata.get("location"),
                    "type": "photo"  # Add type field for proper categorization
                }
            })
        return descriptions
