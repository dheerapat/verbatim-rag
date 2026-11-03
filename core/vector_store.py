"""Shared LanceDB connection configuration for local and S3-backed tables."""

import os
from pathlib import Path

import lancedb

DEFAULT_URI = str(Path(__file__).resolve().parents[1] / "lance-data")


def connect_database():
    """Connect to the configured LanceDB URI.

    Local paths work out of the box. For S3 URIs, optional endpoint and AWS
    credentials configure S3-compatible services such as RustFS.
    """
    uri = os.environ.get("LANCEDB_URI", DEFAULT_URI)
    storage_options = {}

    if uri.startswith("s3://"):
        endpoint = os.environ.get("LANCEDB_S3_ENDPOINT") or os.environ.get(
            "AWS_ENDPOINT_URL"
        )
        if endpoint:
            storage_options["aws_endpoint"] = endpoint
            storage_options["aws_allow_http"] = str(
                endpoint.lower().startswith("http://")
            ).lower()
            # S3-compatible local endpoints such as RustFS generally require
            # path-style addressing (http://host/bucket rather than
            # http://bucket.host).
            storage_options["aws_virtual_hosted_style_request"] = "false"

        credentials = {
            "aws_access_key_id": os.environ.get("LANCEDB_S3_ACCESS_KEY_ID")
            or os.environ.get("RUSTFS_ACCESS_KEY")
            or os.environ.get("AWS_ACCESS_KEY_ID"),
            "aws_secret_access_key": os.environ.get("LANCEDB_S3_SECRET_ACCESS_KEY")
            or os.environ.get("RUSTFS_SECRET_KEY")
            or os.environ.get("AWS_SECRET_ACCESS_KEY"),
            "aws_region": os.environ.get("AWS_REGION", "us-east-1"),
        }
        storage_options.update(
            {key: value for key, value in credentials.items() if value is not None}
        )

    return lancedb.connect(uri, storage_options=storage_options or None)
