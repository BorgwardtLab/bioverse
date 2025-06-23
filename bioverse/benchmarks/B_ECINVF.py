from ..benchmark import Benchmark
from ..datasets import D_AFEC00
from ..metrics import RecoveryMetric
from ..samplers import MoleculeSampler
from ..tasks import InverseFoldingTask


class B_ECINVF(Benchmark):
    dataset = D_AFEC00
    sampler = MoleculeSampler
    task = InverseFoldingTask
    metric = RecoveryMetric
