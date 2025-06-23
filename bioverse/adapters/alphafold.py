from ..adapter import Adapter
from ..data import Assets, Split
from ..processors import PdbProcessor
from ..utilities import batched, config, download, glob_delete


class AlphaFoldAdapter(Adapter):
    """Adapter for AlphaFoldDB."""

    @classmethod
    def download(cls, name="swissprot_pdb", version="v4"):
        path = config.raw_path / "AlphaFoldDB" / version / name
        base_url = "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/"
        download(f"{base_url}/{name}_{version}.tar", path)
        glob_delete(str(path / "*.cif.gz"))  # remove double files, keep .pdb
        return batched(PdbProcessor.process(path)), Split([]), Assets({})
