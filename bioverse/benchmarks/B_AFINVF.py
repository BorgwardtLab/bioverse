from ..benchmark import Benchmark
from ..datasets import D_AFSP00
from ..metrics import RecoveryMetric
from ..samplers import MoleculeSampler
from ..tasks import InverseFoldingTask


class B_AFINVF(Benchmark):
    dataset = D_AFSP00
    sampler = MoleculeSampler
    task = InverseFoldingTask
    metric = RecoveryMetric
