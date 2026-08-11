Task
====

A **task** defines the prediction problem: given a batch index, it loads the
relevant shard through a :class:`~bioverse.virtual.VirtualBatch`, selects input
features and supervision targets, and returns a ``(X, y)`` pair as Awkward Arrays.

Tasks encode what the model should predict — property values, structural classes,
binding affinities, inverse-folding sequences, etc. The trainer and collater
consume the task output; the metric evaluates predictions against ``y``.

Each benchmark declares exactly one task. Override it in ``B_*.yaml`` when the
same dataset supports multiple prediction targets. See
:doc:`../implementations/tasks`.

.. automodule:: bioverse.task
   :members: Task
