from functools import partial


class Trainer:
    def __init__(self, benchmark, model, criterion, optimizer, batch_size, num_workers):
        self.benchmark = benchmark
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __getattr__(self, name):
        partitions = self.benchmark.dataset.split.attrs["names"]
        if name in partitions:
            trainable = self.benchmark.dataset.split.attrs["trainable"][name]
            return partial(self.loop, partition=name, trainable=trainable)
        return getattr(self.benchmark, name)

    def loop(self, epochs, partition, trainable, **kwargs):
        for epoch in range(epochs):
            for batch in getattr(self.benchmark, f"{partition}_loader")(**kwargs):
                self.step(batch, trainable)
