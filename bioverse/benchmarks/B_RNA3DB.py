from ..benchmark import Benchmark
from ..datasets import D_RNA3DB
from ..metrics import TMScoreMetric
from ..samplers import MoleculeSampler
from ..tasks import StructurePredictionTask


class B_RNA3DB(Benchmark):
    dataset = D_RNA3DB
    sampler = MoleculeSampler
    task = StructurePredictionTask
    metric = TMScoreMetric, dict(type="RNA")
