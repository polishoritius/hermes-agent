from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import agent.model_metadata as model_metadata


class ModelMetadataReadOnlyTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.cache_path = self.root / "cache" / "openrouter_model_metadata.json"
        self.patchers = [
            patch.object(model_metadata, "_model_metadata_cache", {}),
            patch.object(model_metadata, "_model_metadata_cache_time", 0.0),
            patch.object(
                model_metadata,
                "_get_model_metadata_cache_path",
                return_value=self.cache_path,
            ),
        ]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def _write_cache(self, age_seconds: float = 0.0):
        self.cache_path.parent.mkdir(parents=True)
        self.cache_path.write_text(
            json.dumps({"test/model": {"pricing": {"prompt": "0"}}}),
            encoding="utf-8",
        )
        timestamp = time.time() - age_seconds
        os.utime(self.cache_path, (timestamp, timestamp))

    def _assert_read_only_cache(self, age_seconds: float):
        self._write_cache(age_seconds)
        before = (self.cache_path.stat().st_mtime_ns, self.cache_path.read_bytes())
        with patch.object(
            model_metadata.requests,
            "get",
            side_effect=AssertionError("read-only metadata performed network I/O"),
        ), patch.object(
            model_metadata,
            "_save_model_metadata_disk_cache",
            side_effect=AssertionError("read-only metadata wrote disk cache"),
        ):
            result = model_metadata.fetch_model_metadata(read_only=True)
        self.assertIn("test/model", result)
        self.assertEqual(
            (self.cache_path.stat().st_mtime_ns, self.cache_path.read_bytes()),
            before,
        )

    def test_read_only_uses_fresh_disk_cache_without_network_or_write(self):
        self._assert_read_only_cache(0.0)

    def test_read_only_uses_stale_disk_cache_without_network_or_write(self):
        self._assert_read_only_cache(model_metadata._MODEL_CACHE_TTL + 60)

    def test_read_only_missing_cache_creates_nothing(self):
        with patch.object(
            model_metadata.requests,
            "get",
            side_effect=AssertionError("read-only metadata performed network I/O"),
        ):
            self.assertEqual(model_metadata.fetch_model_metadata(read_only=True), {})
        self.assertFalse(self.cache_path.exists())
        self.assertFalse(self.cache_path.parent.exists())

    def test_normal_mode_retains_refresh_and_write(self):
        writes: list[dict] = []

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "data": [
                        {"id": "test/model", "pricing": {"prompt": "0"}}
                    ]
                }

        with patch.object(
            model_metadata.requests, "get", return_value=FakeResponse()
        ), patch.object(
            model_metadata,
            "_save_model_metadata_disk_cache",
            side_effect=lambda data: writes.append(data),
        ):
            result = model_metadata.fetch_model_metadata()
        self.assertIn("test/model", result)
        self.assertEqual(len(writes), 1)


if __name__ == "__main__":
    unittest.main()
