from ..benchmark import Benchmark
from ..datasets import D_ARES00
from ..metrics import MeanAbsoluteErrorMetric
from ..samplers import MoleculeSampler
from ..tasks import PropertyPredictionTask


class B_RNKRNA(Benchmark):
    dataset = D_ARES00
    sampler = MoleculeSampler
    task = PropertyPredictionTask, dict(property="rms")
    metric = MeanAbsoluteErrorMetric
