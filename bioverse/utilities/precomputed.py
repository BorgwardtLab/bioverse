from __future__ import annotations

import re
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path

import requests

from .io import note, progressbar, warn

_VERSION_KEY = re.compile(r"^v?(?P<version>\d+)$")


def parse_version_key(key: str | int) -> int:
    if isinstance(key, int):
        return key
    match = _VERSION_KEY.match(str(key))
    if match is None:
        raise ValueError(
            f"Invalid precomputed version key {key!r}. Expected an integer or vN."
        )
    return int(match.group("version"))


class Precomputed:
    """Precomputed dataset archives keyed by version and transform hash."""

    def __init__(self, config: dict | None = None) -> None:
        self.versions: dict[int, dict] = {}
        self.transforms: dict[str, str] = {}
        if not config:
            return
        for key, value in config.items():
            if key == "transforms":
                if not isinstance(value, dict):
                    raise ValueError("precomputed.transforms must be a mapping of hash to url.")
                self.transforms = {str(hash_key): str(url) for hash_key, url in value.items()}
                continue
            version = parse_version_key(key)
            if isinstance(value, str):
                self.versions[version] = {"url": value}
            elif isinstance(value, dict):
                entry = dict(value)
                transforms = entry.pop("transforms", {})
                if transforms:
                    entry["transforms"] = {
                        str(hash_key): str(url) for hash_key, url in transforms.items()
                    }
                self.versions[version] = entry
            else:
                raise ValueError(
                    f"Invalid precomputed entry for version {key!r}. Expected url or mapping."
                )

    @property
    def latest_version(self) -> int | None:
        return max(self.versions) if self.versions else None

    def url_for_version(self, version: int) -> str | None:
        entry = self.versions.get(version)
        if entry is None:
            return None
        url = entry.get("url")
        return str(url) if url else None

    def url_for_transform(self, version: int, transform_hash: str) -> str | None:
        entry = self.versions.get(version, {})
        transforms = entry.get("transforms", {})
        url = transforms.get(transform_hash)
        if url:
            return str(url)
        return self.transforms.get(transform_hash)


def _download_file(url: str, out_path: Path, chunk_size: int = 1024 * 1024) -> Path:
    url_path = Path(url.split("?")[0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.suffix:
        out_path = out_path.with_suffix("".join(url_path.suffixes))
    if out_path.exists():
        note(f"{out_path.name} already downloaded.")
        return out_path
    if url.startswith("file://"):
        source = Path(url.removeprefix("file://"))
        shutil.copy(source, out_path)
        return out_path
    response = requests.get(url, stream=True, headers={"User-Agent": "XY"})
    response.raise_for_status()
    try:
        with open(out_path, "wb") as file:
            for data in progressbar(
                response.iter_content(chunk_size=chunk_size),
                description=f"Downloading {url_path.name}",
                total=int(response.headers.get("content-length", 0)) // chunk_size,
            ):
                file.write(data)
    except Exception:
        if out_path.exists():
            out_path.unlink()
        raise
    return out_path


def _decompress(path: Path) -> Path:
    if path.suffix == ".gz":
        target = path.with_suffix("")
        if target.exists():
            return target
        import gzip

        with gzip.open(path, "rb") as source, open(target, "wb") as dest:
            shutil.copyfileobj(source, dest)
        return target
    if path.suffix == ".bz2":
        target = path.with_suffix("")
        if target.exists():
            return target
        import bz2

        with bz2.open(path, "rb") as source, open(target, "wb") as dest:
            shutil.copyfileobj(source, dest)
        return target
    if path.suffix == ".xz":
        target = path.with_suffix("")
        if target.exists():
            return target
        import lzma

        with lzma.open(path, "rb") as source, open(target, "wb") as dest:
            shutil.copyfileobj(source, dest)
        return target
    return path


def _extract_archive(archive: Path, out_path: Path) -> None:
    out_path.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive, "r") as archive_file:
            archive_file.extractall(out_path)
        return
    if archive.suffix == ".tar" or archive.name.endswith(".tar"):
        with tarfile.open(archive, "r:*") as archive_file:
            archive_file.extractall(out_path)
        return
    raise ValueError(f"Unsupported archive format: {archive.name}")


def _resolve_shard_root(root: Path) -> Path:
    if (root / "num_shards.json").is_file():
        return root
    children = sorted(path for path in root.iterdir() if path.is_dir())
    if len(children) == 1 and (children[0] / "num_shards.json").is_file():
        return children[0]
    for child in children:
        if child.name.startswith("v") and child.name[1:].isdigit():
            for hash_dir in child.iterdir():
                if hash_dir.is_dir() and (hash_dir / "num_shards.json").is_file():
                    return hash_dir
    for child in children:
        if child.is_dir() and (child / "num_shards.json").is_file():
            return child
    raise ValueError(f"Could not locate dataset shards in archive at {root}")


def install_precomputed(url: str, dest_dir: Path) -> None:
    """Download and install a precomputed dataset archive."""
    dest_dir = Path(dest_dir)
    if (dest_dir / "num_shards.json").is_file():
        note(f"Precomputed dataset already installed at {dest_dir}.")
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        archive = _download_file(url, tmp_path / "archive")
        while archive.suffix in {".gz", ".bz2", ".xz"}:
            archive = _decompress(archive)
        extract_root = tmp_path / "extracted"
        _extract_archive(archive, extract_root)
        source = _resolve_shard_root(extract_root)
        dest_dir.parent.mkdir(parents=True, exist_ok=True)
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(source, dest_dir)


def fetch_precomputed(
    precomputed: Precomputed | None,
    version: int,
    dest_dir: Path,
    transform_hash: str | None = None,
) -> bool:
    if precomputed is None:
        return False
    transform_hash = transform_hash or dest_dir.name
    url = precomputed.url_for_transform(version, transform_hash)
    if url is None and dest_dir.name == transform_hash:
        url = precomputed.url_for_version(version)
    if url is None:
        return False
    try:
        install_precomputed(url, dest_dir)
    except Exception as exc:
        warn(f"Failed to download precomputed dataset from {url}: {exc}")
        return False
    return (dest_dir / "num_shards.json").is_file()
