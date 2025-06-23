from ..benchmark import Benchmark
from ..datasets import D_INVC42
from ..metrics import RecoveryMetric
from ..samplers import MoleculeSampler
from ..tasks import InverseFoldingTask


class B_INVC42(Benchmark):
    dataset = D_INVC42
    sampler = MoleculeSampler
    task = InverseFoldingTask
    metric = RecoveryMetric
