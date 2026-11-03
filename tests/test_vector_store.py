import os
import unittest
from unittest.mock import patch

from core.vector_store import DEFAULT_URI, connect_database


class ConnectDatabaseTests(unittest.TestCase):
    @patch("core.vector_store.lancedb.connect")
    def test_defaults_to_local_lance_directory(self, connect):
        with patch.dict(os.environ, {}, clear=True):
            connect_database()
        connect.assert_called_once_with(DEFAULT_URI, storage_options=None)

    @patch("core.vector_store.lancedb.connect")
    def test_configures_rustfs_s3_storage(self, connect):
        environment = {
            "LANCEDB_URI": "s3://verbatim-rag/lance-data",
            "LANCEDB_S3_ENDPOINT": "http://localhost:9000",
            "RUSTFS_ACCESS_KEY": "access",
            "RUSTFS_SECRET_KEY": "secret",
            "AWS_REGION": "us-east-1",
        }
        with patch.dict(os.environ, environment, clear=True):
            connect_database()

        connect.assert_called_once_with(
            "s3://verbatim-rag/lance-data",
            storage_options={
                "aws_endpoint": "http://localhost:9000",
                "aws_allow_http": "true",
                "aws_virtual_hosted_style_request": "false",
                "aws_access_key_id": "access",
                "aws_secret_access_key": "secret",
                "aws_region": "us-east-1",
            },
        )

    @patch("core.vector_store.lancedb.connect")
    def test_rustfs_credentials_override_ambient_aws_credentials(self, connect):
        environment = {
            "LANCEDB_URI": "s3://verbatim-rag/lance-data",
            "LANCEDB_S3_ENDPOINT": "http://localhost:9000",
            "AWS_ACCESS_KEY_ID": "ambient-aws-access",
            "AWS_SECRET_ACCESS_KEY": "ambient-aws-secret",
            "RUSTFS_ACCESS_KEY": "rustfs-access",
            "RUSTFS_SECRET_KEY": "rustfs-secret",
        }
        with patch.dict(os.environ, environment, clear=True):
            connect_database()

        options = connect.call_args.kwargs["storage_options"]
        self.assertEqual(options["aws_access_key_id"], "rustfs-access")
        self.assertEqual(options["aws_secret_access_key"], "rustfs-secret")

    @patch("core.vector_store.lancedb.connect")
    def test_https_s3_endpoint_disables_plain_http(self, connect):
        environment = {
            "LANCEDB_URI": "s3://bucket/table",
            "AWS_ENDPOINT_URL": "https://s3.example.test",
            "AWS_ACCESS_KEY_ID": "access",
            "AWS_SECRET_ACCESS_KEY": "secret",
        }
        with patch.dict(os.environ, environment, clear=True):
            connect_database()

        options = connect.call_args.kwargs["storage_options"]
        self.assertEqual(options["aws_allow_http"], "false")
        self.assertEqual(options["aws_endpoint"], "https://s3.example.test")
        self.assertEqual(options["aws_access_key_id"], "access")


if __name__ == "__main__":
    unittest.main()
