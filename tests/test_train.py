"""Preflight checks that keep mock or stale data out of real LoRA runs."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cse472_project2 import train


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class RealTrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.patch_root = patch.object(train, "PROJECT_ROOT", self.root)
        self.patch_root.start()
        self.addCleanup(self.patch_root.stop)

        self.original = self.root / "data/original"
        self.original_prepared = self.original / "prepared/train.jsonl"
        self.original_prepared.parent.mkdir(parents=True)
        self.original_prepared.write_text('{"prompt":"Question? Answer:","target":"A"}\n', encoding="utf-8")
        self.original_manifest = {
            "dataset_id": "allenai/ai2_arc",
            "subset": "ARC-Challenge",
            "resolved_revision": "a-real-dataset-commit",
            "files": {
                "prepared/train.jsonl": {"sha256": sha256(self.original_prepared.read_bytes())},
                "raw/train.jsonl": {"rows": 1},
                "raw/validation.jsonl": {"sha256": "clean-validation"},
                "raw/test.jsonl": {"sha256": "clean-test"},
            },
        }
        self._write_manifest(self.original, self.original_manifest)

    @staticmethod
    def _write_manifest(directory, manifest):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_accepts_official_original_data(self):
        provenance = train.validate_real_training_data(self.original_prepared, "original")
        self.assertEqual(provenance["dataset_revision"], "a-real-dataset-commit")

    def test_rejects_mock_original_data(self):
        self.original_manifest["resolved_revision"] = "mock_revision"
        self._write_manifest(self.original, self.original_manifest)
        with self.assertRaisesRegex(ValueError, "not mock data"):
            train.validate_real_training_data(self.original_prepared, "original")

    def test_rejects_protected_data_from_different_source_split(self):
        protected = self.root / "data/protected"
        prepared = protected / "prepared/train.jsonl"
        prepared.parent.mkdir(parents=True)
        prepared.write_text('{"prompt":"Question? [!] Answer:","target":"A"}\n', encoding="utf-8")
        manifest = {
            "protection_method": "label_hint",
            "files": {
                "prepared/train.jsonl": {"sha256": sha256(prepared.read_bytes())},
                "raw/train.jsonl": {"rows": 1},
                "raw/validation.jsonl": {"sha256": "clean-validation"},
                "raw/test.jsonl": {"sha256": "stale-mock-test"},
            },
        }
        self._write_manifest(protected, manifest)
        with self.assertRaisesRegex(ValueError, "clean test split"):
            train.validate_real_training_data(prepared, "protected")


if __name__ == "__main__":
    unittest.main()
