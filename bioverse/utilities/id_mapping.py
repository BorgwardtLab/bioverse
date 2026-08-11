from __future__ import annotations

import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

from . import config
from .io import download, progressbar


ALPHAFOLD_FILES = "https://alphafold.ebi.ac.uk/files"
ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api/prediction/{uniprot}"
_thread_local = threading.local()


def parse_invbench_name(name: str) -> tuple[str, str]:
    pdb_id, chain = name.split(".", 1)
    return pdb_id.upper(), chain


def uniprot_from_alphafold_path(path: Path | str) -> str | None:
    name = Path(path).name
    if name.startswith("AF-") and "-F" in name:
        return name.split("-")[1]
    return None


def uniprot_from_foldseek_target(target: str) -> str | None:
    name = target.split()[0] if target else target
    if name.startswith("AF-"):
        return uniprot_from_alphafold_path(name)
    return None


def _get_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        session = requests.Session()
        session.headers.update({"User-Agent": "XY"})
        _thread_local.session = session
    return _thread_local.session


def _cached_structure(uniprot: str, cache_dir: Path) -> Path | None:
    cached = sorted(cache_dir.glob(f"AF-{uniprot}-*.pdb*"))
    return cached[0] if cached else None


def _direct_alphafold_urls(uniprot: str, fragment: str = "F1") -> list[str]:
    return [
        f"{ALPHAFOLD_FILES}/AF-{uniprot}-{fragment}-model_{version}.pdb"
        for version in ("v6", "v4", "v5")
    ]


def _write_pdb_response(response: requests.Response, dest: Path) -> bool:
    if response.status_code != 200 or not response.content:
        return False
    dest.write_bytes(response.content)
    return dest.exists()


def load_sifts_mapping(path: Path | None = None) -> dict[tuple[str, str], list[str]]:
    path = path or config.raw_path / "SIFTS" / "pdb_chain_uniprot.csv"
    if not path.exists():
        download(
            "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/pdb_chain_uniprot.csv.gz",
            path,
        )
    df = pd.read_csv(path, comment="#", low_memory=False)
    mapping: dict[tuple[str, str], list[str]] = defaultdict(list)
    for pdb_id, chain, uniprot in df[["PDB", "CHAIN", "SP_PRIMARY"]].itertuples(
        index=False
    ):
        key = (str(pdb_id).upper(), str(chain))
        uniprot = str(uniprot)
        if uniprot not in mapping[key]:
            mapping[key].append(uniprot)
    return mapping


def build_alphafold_index(
    path: Path, version: str = "v4", fragment: str = "F1"
) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for suffix in (
        f"-{fragment}-model_{version}.pdb",
        f"-{fragment}-model_{version}.pdb.gz",
    ):
        for pdb_path in path.glob(f"AF-*{suffix}"):
            accession = pdb_path.name.split("-")[1]
            index.setdefault(accession, pdb_path)
    return index


def alphafold_candidates(
    pdb_id: str,
    chain: str,
    sifts_mapping: dict[tuple[str, str], list[str]],
    alphafold_index: dict[str, Path],
) -> list[tuple[str, Path]]:
    return [
        (uniprot, alphafold_index[uniprot])
        for uniprot in sifts_mapping.get((pdb_id.upper(), chain), [])
        if uniprot in alphafold_index
    ]


def fetch_alphafold_structure(uniprot: str, cache_dir: Path) -> Path | None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = _cached_structure(uniprot, cache_dir)
    if cached is not None:
        return cached

    session = _get_session()
    for url in _direct_alphafold_urls(uniprot):
        dest = cache_dir / Path(urlparse(url).path).name
        if dest.exists():
            return dest
        if _write_pdb_response(session.get(url, timeout=60), dest):
            return dest

    response = session.get(ALPHAFOLD_API.format(uniprot=uniprot), timeout=30)
    if response.status_code != 200:
        return None

    entries = response.json()
    if not entries:
        return None

    pdb_url = entries[0].get("pdbUrl")
    if not pdb_url:
        return None

    dest = cache_dir / Path(urlparse(pdb_url).path).name
    if dest.exists():
        return dest
    if _write_pdb_response(session.get(pdb_url, timeout=60), dest):
        return dest
    return None


def fetch_alphafold_structures(
    uniprots: list[str],
    cache_dir: Path,
    workers: int | None = None,
) -> dict[str, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    workers = workers or max(config.workers, 32)
    unique = sorted(set(uniprots))
    index = {
        uniprot: path
        for uniprot in unique
        if (path := _cached_structure(uniprot, cache_dir)) is not None
    }
    to_fetch = [uniprot for uniprot in unique if uniprot not in index]
    if not to_fetch:
        return index

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fetch_alphafold_structure, uniprot, cache_dir): uniprot
            for uniprot in to_fetch
        }
        for future in progressbar(
            as_completed(futures),
            total=len(futures),
            description="Downloading AlphaFold structures",
        ):
            uniprot, path = futures[future], future.result()
            if path is not None:
                index[uniprot] = path
    return index


def resolve_alphafold_structure(
    uniprot: str,
    local_index: dict[str, Path],
    api_cache_dir: Path,
    use_api: bool = True,
    api_index: dict[str, Path] | None = None,
) -> tuple[Path, str] | None:
    if uniprot in local_index:
        return local_index[uniprot], "local"
    if api_index is not None and uniprot in api_index:
        return api_index[uniprot], "api"
    if not use_api:
        return None
    path = fetch_alphafold_structure(uniprot, api_cache_dir)
    if path is None:
        return None
    return path, "api"
