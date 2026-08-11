import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from bioverse.utilities.precomputed import (
    Precomputed,
    _resolve_shard_root,
    fetch_precomputed,
    install_precomputed,
    parse_version_key,
)


class TestPrecomputedConfig:
    def test_parse_version_key(self):
        assert parse_version_key(1) == 1
        assert parse_version_key("2") == 2
        assert parse_version_key("v3") == 3

    def test_invalid_version_key(self):
        with pytest.raises(ValueError):
            parse_version_key("latest")

    def test_version_urls(self):
        config = Precomputed(
            {
                "v1": "https://example.com/v1.tar.gz",
                2: {"url": "https://example.com/v2.tar.gz"},
            }
        )
        assert config.latest_version == 2
        assert config.url_for_version(1) == "https://example.com/v1.tar.gz"
        assert config.url_for_version(2) == "https://example.com/v2.tar.gz"

    def test_transform_urls(self):
        config = Precomputed(
            {
                1: {
                    "url": "https://example.com/base.tar.gz",
                    "transforms": {
                        "abc123": "https://example.com/abc123.tar.gz",
                    },
                },
                "transforms": {
                    "def456": "https://example.com/def456.tar.gz",
                },
            }
        )
        assert (
            config.url_for_transform(1, "abc123")
            == "https://example.com/abc123.tar.gz"
        )
        assert (
            config.url_for_transform(1, "def456")
            == "https://example.com/def456.tar.gz"
        )
        assert (
            config.url_for_transform(2, "def456")
            == "https://example.com/def456.tar.gz"
        )
        assert config.url_for_transform(2, "missing") is None


class TestArchiveResolution:
    def test_resolve_flat_archive(self, tmp_path):
        shard_root = tmp_path / "flat"
        shard_root.mkdir()
        (shard_root / "num_shards.json").write_text("1")
        assert _resolve_shard_root(tmp_path) == shard_root

    def test_resolve_single_subdir(self, tmp_path):
        shard_root = tmp_path / "hashdir"
        shard_root.mkdir()
        (shard_root / "num_shards.json").write_text("1")
        assert _resolve_shard_root(tmp_path) == shard_root

    def test_resolve_version_layout(self, tmp_path):
        shard_root = tmp_path / "v1" / "abc123"
        shard_root.mkdir(parents=True)
        (shard_root / "num_shards.json").write_text("1")
        assert _resolve_shard_root(tmp_path) == shard_root


class TestInstallPrecomputed:
    def _make_archive(self, archive_path: Path, shard_root_name: str = "hash") -> None:
        source = archive_path.parent / "source" / shard_root_name
        source.mkdir(parents=True)
        (source / "num_shards.json").write_text("1")
        (source / "1.ak").write_bytes(b"test")
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(source, arcname=shard_root_name)

    def test_install_precomputed(self, tmp_path):
        archive = tmp_path / "dataset.tar.gz"
        dest = tmp_path / "installed"
        self._make_archive(archive)
        install_precomputed(f"file://{archive}", dest)
        assert (dest / "num_shards.json").is_file()
        assert (dest / "1.ak").is_file()

    def test_fetch_precomputed(self, tmp_path):
        archive = tmp_path / "dataset.tar.gz"
        dest = tmp_path / "v1" / "hash"
        self._make_archive(archive)
        config = Precomputed({1: {"url": f"file://{archive}"}})
        assert fetch_precomputed(config, 1, dest, "hash") is True
        assert (dest / "num_shards.json").is_file()

    def test_fetch_precomputed_missing_url(self, tmp_path):
        dest = tmp_path / "v1" / "hash"
        assert fetch_precomputed(Precomputed({}), 1, dest, "hash") is False
