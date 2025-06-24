from bioverse.processors import *
from bioverse.utilities import config

config.workers = 1


def test_pdb_processor():
    processor = PdbProcessor()
    item = processor.process_file("testing/dummy/dummy.pdb")


def test_cif_processor():
    processor = CifProcessor()
    item = processor.process_file("testing/dummy/dummy.cif")
