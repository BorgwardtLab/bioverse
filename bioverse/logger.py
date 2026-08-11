class Logger:
    """Training logger interface for metrics, tensors, and artifacts.

    Loggers receive the owning :class:`~bioverse.trainer.Trainer` and write
    to disk, Comet ML, or nowhere (:class:`~bioverse.loggers.no.NoLogger`).
    The backend calls :meth:`log_loss` and related methods during training.

    Examples
    --------
    .. code-block:: python

       from bioverse import Trainer

       trainer = Trainer(model, benchmark, logger="DiskLogger", root="results/run1")
    """

    def __init__(self, trainer):
        self.trainer = trainer

    @property
    def root(self):
        return self.trainer.root

    def log_loss(self, data, name=None):
        pass

    def log_dict(self, data):
        pass

    def log_tensor(self, data):
        pass

    def log_image(self, data):
        pass

    def log_text(self, data):
        pass
