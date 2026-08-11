Sampler
=======

A **sampler** decides which elements of a dataset appear in each batch during
training or evaluation. Given the dataset table-of-contents (``toc``) and an
active split partition, :meth:`~bioverse.sampler.Sampler.index` returns Awkward
Array indices at the scene, frame, molecule, residue, or graph level.

:meth:`~bioverse.sampler.Sampler.sample` then groups those indices into batches
according to ``batch_size``, ``batch_on`` (e.g. ``"molecules"`` or ``"scenes"``),
and distributed-training settings (``world_size``, ``rank``). Different samplers
implement different sampling strategies — uniform over molecules, frame-based
windows, mutation-aware grouping, etc.

See :doc:`../implementations/samplers`.

.. automodule:: bioverse.sampler
   :members: Sampler
