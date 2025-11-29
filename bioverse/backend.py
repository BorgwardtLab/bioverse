from abc import ABC, abstractmethod


class Backend(ABC):
    @abstractmethod
    def __init__(self, trainer):
        self.trainer

    @abstractmethod
    def train_step(self, batch):
        pass

    @abstractmethod
    def eval_step(self, batch):
        pass

    @abstractmethod
    def save_checkpoint(self, path):
        pass

    @abstractmethod
    def load_checkpoint(self, path):
        pass
