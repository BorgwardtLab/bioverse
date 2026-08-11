from abc import ABC, abstractmethod


class Backend(ABC):
    """Framework-specific training loop implementation.

    Backends encapsulate device placement, forward/backward passes, optimizer
    steps, and checkpoint I/O for a given deep-learning library. The
    :class:`~bioverse.trainer.Trainer` delegates all framework interaction to
    its backend.

    Examples
    --------
    .. code-block:: python

       from bioverse import Trainer

       trainer = Trainer(model, benchmark, backend="TorchBackend", accelerator="gpu")
       trainer.train()
    """

    @abstractmethod
    def __init__(self, trainer):
        self.trainer

    @abstractmethod
    def train_step(self, batch):
        """Run one training step and return the loss."""
        pass

    @abstractmethod
    def eval_step(self, batch):
        """Run one evaluation step and update the benchmark metric."""
        pass

    @abstractmethod
    def save_checkpoint(self, path):
        """Save model and optimizer state to ``path``."""
        pass

    @abstractmethod
    def load_checkpoint(self, path):
        """Restore model and optimizer state from ``path``."""
        pass
