from ..benchmark import Benchmark
from ..datasets import D_HIV2D0
from ..metrics import AurocMetric
from ..samplers import MoleculeSampler
from ..tasks import PropertyPredictionTask


class B_HIV2D0(Benchmark):
    dataset = D_HIV2D0
    sampler = MoleculeSampler
    task = PropertyPredictionTask
    metric = AurocMetric, dict(on=0)
