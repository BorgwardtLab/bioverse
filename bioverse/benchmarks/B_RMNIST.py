from ..benchmark import Benchmark
from ..datasets import D_RMNIST
from ..metrics import ErrorRateMetric
from ..samplers import MoleculeSampler
from ..tasks import PropertyPredictionTask


class B_RMNIST(Benchmark):
    dataset = D_RMNIST
    sampler = MoleculeSampler
    task = PropertyPredictionTask, dict(property="label")
    metric = ErrorRateMetric
