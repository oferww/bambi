"""Utility functions for reading/writing EXIF GPS data using piexif.
"""
from __future__ import annotations
import piexif
from pathlib import Path

def write_gps(image_path: str | Path, lat: float, lon: float) -> None:
    """Write GPSLatitude and GPSLongitude tags (in EXIF rational format) to an image.

    The file is modified in place (piexif inserts a new EXIF blob). If the image
    already contains EXIF data, it is preserved and merged.
    """
    image_path = Path(image_path)
    exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
    if image_path.exists():
        try:
            existing = piexif.load(str(image_path))
            exif_dict.update(existing)
        except Exception:
            # No EXIF present; start fresh
            pass

    def _deg_to_dms_rational(deg: float):
        deg_abs = abs(deg)
        minutes, seconds = divmod(deg_abs * 3600, 60)
        degrees, minutes = divmod(minutes, 60)
        return (
            (int(degrees), 1),
            (int(minutes), 1),
            (int(seconds * 100), 100),  # keep 2 decimal points
        )

    lat_ref = "N" if lat >= 0 else "S"
    lon_ref = "E" if lon >= 0 else "W"

    exif_dict["GPS"][piexif.GPSIFD.GPSLatitudeRef] = lat_ref.encode()
    exif_dict["GPS"][piexif.GPSIFD.GPSLatitude] = _deg_to_dms_rational(lat)
    exif_dict["GPS"][piexif.GPSIFD.GPSLongitudeRef] = lon_ref.encode()
    exif_dict["GPS"][piexif.GPSIFD.GPSLongitude] = _deg_to_dms_rational(lon)

    piexif.insert(piexif.dump(exif_dict), str(image_path))
