from pathlib import Path

from ..adapter import Adapter
from ..data import Assets, Split
from ..processors import PdbProcessor
from ..utilities import batched, config, download, glob_delete, load


EXCLUSIONS_PATH = Path(__file__).parent / "data" / "foldseek_exclusions.json"
EXCLUDED_UNIPROTS = frozenset(load(EXCLUSIONS_PATH)["excluded_uniprots"])


class AlphaFoldExclusionAdapter(Adapter):
    """Download AlphaFold structures with Foldseek-based homology exclusions for inverse folding benchmarks."""

    @classmethod
    def download(cls, name: str = "swissprot_pdb", version: str = "v4"):
        path = config.raw_path / "AlphaFoldDB" / version / name
        base_url = "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/"
        download(f"{base_url}/{name}_{version}.tar", path)
        glob_delete(str(path / "*.cif.gz"))
        batches = batched(
            PdbProcessor.process(path, exclude=EXCLUDED_UNIPROTS),
        )
        assets = Assets(
            {
                "exclusions": {
                    "path": str(EXCLUSIONS_PATH),
                    "count": len(EXCLUDED_UNIPROTS),
                }
            }
        )
        return batches, Split(), assets
