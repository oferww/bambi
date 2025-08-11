import os
import sys
import hashlib
from typing import Optional


def _md5_of_file(path: str) -> Optional[str]:
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except FileNotFoundError:
        return None


def sync_s3_prefix_to_dir(
    bucket: str,
    prefix: str,
    local_dir: str,
    region: Optional[str] = None,
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    session_token: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    overwrite: bool = False,
) -> int:
    """
    Download all S3 objects under bucket/prefix into local_dir, preserving hierarchy.

    - If overwrite=False, skip files that already exist with matching size/etag-like md5.
    - Returns number of files downloaded/updated.
    """
    try:
        import boto3  # type: ignore
    except Exception as e:
        print(f"[S3] boto3 not installed: {e}", file=sys.stderr, flush=True)
        return 0

    session_kwargs = {}
    if access_key and secret_key:
        session_kwargs.update(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            aws_session_token=session_token,
        )
    if region:
        session_kwargs["region_name"] = region

    session = boto3.session.Session(**session_kwargs)
    s3 = session.client("s3", endpoint_url=endpoint_url)

    # Normalize prefix (allow empty or trailing slash)
    norm_prefix = prefix.strip()
    if norm_prefix and not norm_prefix.endswith("/"):
        norm_prefix += "/"

    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=norm_prefix)

    downloaded = 0
    for page in pages:
        contents = page.get("Contents", [])
        for obj in contents:
            key = obj["Key"]
            # Skip 'directory placeholder' keys
            if key.endswith("/"):
                continue

            # Compute local path
            rel_key = key[len(norm_prefix):] if norm_prefix and key.startswith(norm_prefix) else key
            local_path = os.path.join(local_dir, rel_key)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            # Decide to download
            head_size = obj.get("Size")
            etag = obj.get("ETag", "").strip('"')

            need_download = True
            if not overwrite and os.path.exists(local_path):
                try:
                    # Quick check: size matches
                    local_size = os.path.getsize(local_path)
                    if local_size == head_size:
                        # Optional md5 check for simple (non-multipart) uploads
                        local_md5 = _md5_of_file(local_path)
                        if local_md5 and len(etag) == 32 and etag == local_md5:
                            need_download = False
                except Exception:
                    pass

            if need_download:
                try:
                    s3.download_file(bucket, key, local_path)
                    downloaded += 1
                    print(f"[S3] Downloaded s3://{bucket}/{key} -> {local_path}", flush=True)
                except Exception as e:
                    print(f"[S3] Failed to download {key}: {e}", file=sys.stderr, flush=True)

    print(f"[S3] Sync complete. Files downloaded/updated: {downloaded}", flush=True)
    return downloaded


def sync_dir_to_s3_prefix(
    bucket: str,
    prefix: str,
    local_dir: str,
    region: Optional[str] = None,
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    session_token: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    overwrite: bool = False,
) -> int:
    """
    Upload all files under local_dir to bucket/prefix, preserving hierarchy.

    - If overwrite=False, skip files that already exist with matching size/etag-like md5.
    - Returns number of files uploaded/updated.
    """
    try:
        import boto3  # type: ignore
        from botocore.exceptions import ClientError  # type: ignore
    except Exception as e:
        print(f"[S3] boto3 not installed: {e}", file=sys.stderr, flush=True)
        return 0

    session_kwargs = {}
    if access_key and secret_key:
        session_kwargs.update(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            aws_session_token=session_token,
        )
    if region:
        session_kwargs["region_name"] = region

    session = boto3.session.Session(**session_kwargs)
    s3 = session.client("s3", endpoint_url=endpoint_url)

    norm_prefix = prefix.strip()
    if norm_prefix and not norm_prefix.endswith("/"):
        norm_prefix += "/"

    uploaded = 0
    for root, _, files in os.walk(local_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            rel_path = os.path.relpath(local_path, start=local_dir).replace("\\", "/")
            key = f"{norm_prefix}{rel_path}" if norm_prefix else rel_path

            # Unless overwriting, check remote head to possibly skip
            if not overwrite:
                try:
                    head = s3.head_object(Bucket=bucket, Key=key)
                    remote_size = head.get("ContentLength")
                    etag = (head.get("ETag") or "").strip('"')
                    local_size = os.path.getsize(local_path)
                    if local_size == remote_size:
                        local_md5 = _md5_of_file(local_path)
                        if local_md5 and len(etag) == 32 and etag == local_md5:
                            # up-to-date
                            continue
                except ClientError as ce:
                    # Not found or inaccessible -> proceed to upload
                    pass
                except Exception:
                    pass

            try:
                s3.upload_file(local_path, bucket, key)
                uploaded += 1
                print(f"[S3] Uploaded {local_path} -> s3://{bucket}/{key}", flush=True)
            except Exception as e:
                print(f"[S3] Failed to upload {local_path}: {e}", file=sys.stderr, flush=True)

    print(f"[S3] Upload sync complete. Files uploaded/updated: {uploaded}", flush=True)
    return uploaded
