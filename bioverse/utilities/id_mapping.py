from __future__ import annotations

import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, NamedTuple
from urllib.parse import urlparse

import pandas as pd
import requests

from . import config
from .io import download, load, progressbar

ALPHAFOLD_FILES = "https://alphafold.ebi.ac.uk/files"
ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api/prediction/{uniprot}"
INVBENCH_URL = (
    "https://github.com/A4Bio/ProteinInvBench/releases/download/"
    "dataset_release/data.tar.gz"
)
SIFTS_UNIPROT_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/"
    "pdb_chain_uniprot.csv.gz"
)
CATH_BOUNDARIES_URL = (
    "https://download.cathdb.info/cath/releases/all-releases/v4_2_0/"
    "cath-classification-data/cath-domain-boundaries-seqreschopping-v4_2_0.txt"
)
_thread_local = threading.local()


class SiftsSegment(NamedTuple):
    uniprot: str
    res_beg: int
    res_end: int
    sp_beg: int
    sp_end: int


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


def _as_int(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() in ("", "None", "nan"):
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return int(value)


def ensure_protein_invbench() -> Path:
    path = config.raw_path / "ProteinInvBench"
    split_path = path / "data" / "cath4.2" / "chain_set_splits.json"
    if not split_path.exists():
        download(INVBENCH_URL, path)
    return path


def load_sifts_segments(
    path: Path | None = None,
) -> dict[tuple[str, str], list[SiftsSegment]]:
    path = path or config.raw_path / "SIFTS" / "pdb_chain_uniprot.csv"
    if not path.exists():
        download(SIFTS_UNIPROT_URL, path)
    df = pd.read_csv(path, comment="#", low_memory=False)
    has_ranges = {"RES_BEG", "RES_END", "SP_BEG", "SP_END"}.issubset(df.columns)
    mapping: dict[tuple[str, str], list[SiftsSegment]] = defaultdict(list)
    for row in df.itertuples(index=False):
        pdb_id = str(getattr(row, "PDB")).upper()
        chain = str(getattr(row, "CHAIN"))
        uniprot = str(getattr(row, "SP_PRIMARY"))
        key = (pdb_id, chain)
        if has_ranges:
            res_beg = _as_int(getattr(row, "RES_BEG"))
            res_end = _as_int(getattr(row, "RES_END"))
            sp_beg = _as_int(getattr(row, "SP_BEG"))
            sp_end = _as_int(getattr(row, "SP_END"))
            if None in (res_beg, res_end, sp_beg, sp_end):
                continue
            mapping[key].append(
                SiftsSegment(uniprot, res_beg, res_end, sp_beg, sp_end)
            )
        else:
            mapping[key].append(SiftsSegment(uniprot, 1, 10**9, 1, 10**9))
    return mapping


def load_sifts_mapping(path: Path | None = None) -> dict[tuple[str, str], list[str]]:
    mapping: dict[tuple[str, str], list[str]] = defaultdict(list)
    for key, segments in load_sifts_segments(path).items():
        for segment in segments:
            if segment.uniprot not in mapping[key]:
                mapping[key].append(segment.uniprot)
    return mapping


def parse_cath_boundary_line(line: str) -> tuple[str, list[tuple[int, int]]] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    domain_id, _, ranges = line.partition("\t")
    if not ranges:
        parts = line.split()
        if len(parts) < 2:
            return None
        domain_id, ranges = parts[0], parts[1]
    segments = []
    for part in ranges.split(","):
        start, _, end = part.partition("-")
        if not start or not end:
            continue
        segments.append((int(start), int(end)))
    if not segments:
        return None
    return domain_id, segments


def load_cath_domain_boundaries(
    path: Path | None = None,
) -> dict[str, list[tuple[int, int]]]:
    path = (
        path
        or config.raw_path
        / "CATH"
        / "v4_2_0"
        / "cath-domain-boundaries-seqreschopping-v4_2_0.txt"
    )
    if not path.exists():
        download(CATH_BOUNDARIES_URL, path)
    boundaries: dict[str, list[tuple[int, int]]] = {}
    with open(path) as handle:
        for line in handle:
            parsed = parse_cath_boundary_line(line)
            if parsed is not None:
                boundaries[parsed[0]] = parsed[1]
    return boundaries


def cath_domains_for_chain(
    pdb_id: str, chain: str, boundaries: dict[str, list[tuple[int, int]]]
) -> list[str]:
    pdb_id = pdb_id.lower()
    chain_char = chain[0] if chain else "0"
    return sorted(
        domain_id
        for domain_id in boundaries
        if len(domain_id) >= 7
        and domain_id[:4] == pdb_id
        and domain_id[4] == chain_char
    )


def domain_length(segments: list[tuple[int, int]]) -> int:
    return sum(end - start + 1 for start, end in segments)


def pick_cath_domain(
    domain_ids: list[str],
    boundaries: dict[str, list[tuple[int, int]]],
    seq: str | None = None,
) -> str | None:
    if not domain_ids:
        return None
    if len(domain_ids) == 1 or seq is None:
        return domain_ids[0]
    target = len(seq)
    return min(
        domain_ids,
        key=lambda domain_id: abs(domain_length(boundaries[domain_id]) - target),
    )


def merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ranges:
        return []
    ordered = sorted(ranges)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def map_seqres_range(
    start: int, end: int, segments: list[SiftsSegment]
) -> list[tuple[str, int, int]]:
    mapped = []
    for segment in segments:
        lo = max(start, segment.res_beg)
        hi = min(end, segment.res_end)
        if lo > hi:
            continue
        sp_lo = segment.sp_beg + (lo - segment.res_beg)
        sp_hi = segment.sp_beg + (hi - segment.res_beg)
        mapped.append((segment.uniprot, sp_lo, sp_hi))
    return mapped


def uniprot_ranges_for_segments(
    seqres_ranges: list[tuple[int, int]], segments: list[SiftsSegment]
) -> list[tuple[str, list[tuple[int, int]]]]:
    by_uniprot: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for start, end in seqres_ranges:
        for uniprot, sp_start, sp_end in map_seqres_range(start, end, segments):
            by_uniprot[uniprot].append((sp_start, sp_end))
    return [
        (uniprot, merge_ranges(ranges)) for uniprot, ranges in by_uniprot.items()
    ]


def resolve_invbench_domain_mapping(
    name: str,
    sifts_segments: dict[tuple[str, str], list[SiftsSegment]],
    cath_boundaries: dict[str, list[tuple[int, int]]],
    seq: str | None = None,
    all_chain_domains: bool = False,
) -> list[tuple[str, list[tuple[int, int]]]]:
    """Map a ProteinInvBench CATH domain to UniProt residue ranges.

    ProteinInvBench names are PDB chains, but the structures are CATH domains.
    CATH 4.2 SEQRES boundaries are projected through SIFTS onto UniProt
    numbering (AlphaFold residue numbers).
    """
    pdb_id, chain = parse_invbench_name(name)
    segments = sifts_segments.get((pdb_id, chain), [])
    domain_ids = cath_domains_for_chain(pdb_id, chain, cath_boundaries)
    if not domain_ids:
        return []
    if not all_chain_domains:
        chosen = pick_cath_domain(domain_ids, cath_boundaries, seq)
        domain_ids = [chosen] if chosen is not None else []
    mapped: list[tuple[str, list[tuple[int, int]]]] = []
    seen: set[tuple[str, tuple[tuple[int, int], ...]]] = set()
    for domain_id in domain_ids:
        for uniprot, ranges in uniprot_ranges_for_segments(
            cath_boundaries[domain_id], segments
        ):
            key = (uniprot, tuple(ranges))
            if key in seen:
                continue
            seen.add(key)
            mapped.append((uniprot, ranges))
    return mapped


def invbench_domain_uniprots(
    names: Iterable[str],
    sifts_segments: dict[tuple[str, str], list[SiftsSegment]] | None = None,
    cath_boundaries: dict[str, list[tuple[int, int]]] | None = None,
) -> set[str]:
    sifts_segments = sifts_segments if sifts_segments is not None else load_sifts_segments()
    cath_boundaries = (
        cath_boundaries if cath_boundaries is not None else load_cath_domain_boundaries()
    )
    uniprots: set[str] = set()
    for name in names:
        mapped = resolve_invbench_domain_mapping(
            name, sifts_segments, cath_boundaries, all_chain_domains=True
        )
        if mapped:
            uniprots.update(uniprot for uniprot, _ in mapped)
            continue
        pdb_id, chain = parse_invbench_name(name)
        uniprots.update(
            segment.uniprot for segment in sifts_segments.get((pdb_id, chain), [])
        )
    return uniprots


def load_invbench_test_domain_uniprots() -> set[str]:
    invbench_path = ensure_protein_invbench()
    split = load(invbench_path / "data" / "cath4.2" / "chain_set_splits.json")
    return invbench_domain_uniprots(split["test"])


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
