# Ingestion package
from .dispatcher import (
    ingest_files,
    ingest_scan_uploads,
    ingest_pdfs_in_uploads,
    ingest_csvs_in_uploads,
    ingest_instagram_jsons_in_uploads,
    ingest_photos_from_photos_dir,
)
