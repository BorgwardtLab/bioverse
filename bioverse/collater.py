from abc import abstractmethod


class Collater:
    """Batch variable-length Awkward Arrays into framework-ready tensors.

    Collaters pad, stack, or graph-construct features and targets produced by
    a :class:`~bioverse.task.Task` so they can be consumed by a model. The
    trainer calls ``collater(X, y, attr=...)`` inside the benchmark loader.

    Examples
    --------
    .. code-block:: python

       from bioverse.collaters import LongCollater

       collater = LongCollater()
       batch = collater(X, y, attr=["residue_pos", "residue_features"])
    """

    def __call__(self, *args, **kwargs):
        return self.collate(*args, **kwargs)

    @abstractmethod
    def collate(self, *args, **kwargs):
        """Combine one or more samples into a single batch."""
        raise NotImplementedError
