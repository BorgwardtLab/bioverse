from ..benchmark import Benchmark
from ..datasets import D_QNTMA9
from ..metrics import MeanAbsoluteErrorMetric
from ..samplers import MoleculeSampler
from ..tasks import PropertyPredictionTask


class B_QM9APH(Benchmark):
    dataset = D_QNTMA9
    sampler = MoleculeSampler
    task = PropertyPredictionTask, dict(property="alpha")
    metric = MeanAbsoluteErrorMetric
