from ..logger import Logger


class NoLogger(Logger):

    """No-op logger that discards all log calls."""

    def __init__(self, trainer):
        super().__init__(trainer)
        self.trainer = trainer

    def log_loss(self, data, mode="train"):
        pass

    def log_dict(self, data, mode="train"):
        pass
