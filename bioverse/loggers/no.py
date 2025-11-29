from ..logger import Logger


class NoLogger(Logger):

    def __init__(self, trainer):
        super().__init__(trainer)
        self.trainer = trainer

    def log_loss(self, data, mode="train"):
        pass

    def log_dict(self, data, mode="train"):
        pass
