User Guide
==========

Overview
--------

Bioverse lets you run machine-learning experiments on biomolecular benchmarks
without implementing data loading, splitting, or evaluation yourself. You
assemble experiments from **existing components** — benchmarks, transforms, models,
and trainer settings — using YAML configuration files and the ``bioverse`` CLI.

This guide covers setup, configuration, and execution. To extend Bioverse with new
components, see the :doc:`Developer Guide <developer>`.

Concepts at a glance
--------------------

An experiment wires together:

.. code-block:: text

   Benchmark
     ├── Dataset      (raw data + offline transforms)
     ├── Sampler      (what to iterate over)
     ├── Task         (features and targets per batch)
     └── Metric       (evaluation)

   Trainer
     ├── Model        (your architecture)
     ├── Backend      (PyTorch / TensorFlow execution)
     ├── Collater     (batch tensor layout)
     └── Logger       (metrics and checkpoints)

Built-in benchmarks are defined as ``B_*.yaml`` files. Each references a dataset
(``D_*.yaml``), sampler, task, and metric by class name. Browse all built-ins
under :doc:`../implementations/benchmarks`.

Environment and paths
---------------------

Bioverse stores artifacts under ``~/.bioverse`` by default. Relevant paths are
configured in :mod:`bioverse.utilities.config`:

- ``BIOVERSE_ROOT`` — default ``~/.bioverse``
- ``BIOVERSE_DATASET_ROOT`` — dataset shards (default under ``BIOVERSE_ROOT``)
- ``BIOVERSE_RAWDATA_ROOT`` — downloaded raw files
- ``BIOVERSE_BENCHMARKS_ROOT`` — benchmark run artifacts
- ``BIOVERSE_SCRATCH_ROOT`` — temporary loader scratch space (default ``/tmp``)

Set ``globals`` in your experiment config to override runtime options such as
``workers`` and ``seed`` on :mod:`bioverse.utilities.config`.

Choosing a benchmark
--------------------

Each built-in benchmark is identified by a short name such as ``B_AFCATH`` or
``B_PROGYM``. Names map to YAML files shipped with the package under
``bioverse/benchmarks/``.

.. code-block:: yaml

   dataset: D_AFCATH
   sampler: MoleculeSampler
   task: InverseFoldingTask
   metric: RecoveryMetric

Use the :doc:`../implementations/benchmarks` and :doc:`../implementations/datasets`
pages to inspect available configs. Pick a benchmark whose task and metric match
your model's objective.

Experiment configuration
------------------------

The CLI expects an OmegaConf YAML file with four **required** top-level keys:

.. list-table::
   :header-rows: 1

   * - Key
     - Description
   * - ``benchmark``
     - Built-in name (``B_*``), path to a YAML file, or inline dict
   * - ``transforms``
     - Offline transforms applied when building the dataset (list)
   * - ``model``
     - Import path to your model class, optionally with kwargs
   * - ``trainer``
     - :class:`~bioverse.trainer.Trainer` keyword arguments

Optional keys:

- ``live_transforms`` — transforms applied on-the-fly inside the data loader
- ``globals`` — key/value pairs written to :mod:`bioverse.utilities.config`

Config composition
------------------

Pass multiple config files and CLI overrides; they are merged in order:

.. code-block:: bash

   bioverse train base.yaml model.yaml trainer.batch_size=64

Files are loaded with :func:`omegaconf.OmegaConf.load`. Dotlist overrides such as
``trainer.epochs=10`` are parsed with :func:`omegaconf.OmegaConf.from_dotlist`.
Interpolations are resolved before the experiment starts.

Specifying the model
--------------------

The ``model`` key accepts either a string import path or a single-key dict with
constructor kwargs:

.. code-block:: yaml

   model: .model.MyModel

   model:
     .model.MyModel:
       hidden_dim: 128
       dropout: 0.1

Paths starting with ``.`` are resolved relative to the **current working directory**
(the directory from which you invoke ``bioverse``). Your model must expose:

- ``train_step(Xy, data)`` — return ``(loss, output)``
- ``eval_step(Xy, data)`` — return ``(loss, output)``
- ``optimizer`` — used by :class:`~bioverse.backends.torch.TorchBackend`

Transforms
----------

**Offline transforms** (``transforms``) are applied when a dataset is released or
re-built. They modify stored shards, splits, and assets.

**Live transforms** (``live_transforms``) run inside the loader on each access and
are useful for augmentation or compute-heavy features.

Reference transforms by class name:

.. code-block:: yaml

   transforms:
     - FilterSequenceLength:
         max_length: 1024
     - SceneSplit:
         test_size: 100
         val_size: 100

   live_transforms:
     - NormalizeVector:
         field: atom_force

See :doc:`../implementations/transforms` for the full list.

Trainer options
---------------

Common ``trainer`` fields:

.. list-table::
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``backend``
     - ``TorchBackend``
     - Execution backend (``TorchBackend``, ``TensorflowBackend``)
   * - ``collater``
     - ``LongCollater``
     - Batching layout (``LongCollater``, ``WideCollater``, …)
   * - ``logger``
     - ``DiskLogger``
     - Metric logger (``NoLogger`` to disable)
   * - ``root``
     - ``results``
     - Output directory for checkpoints and metrics
   * - ``epochs``
     - ``1``
     - Training epochs
   * - ``batch_size``
     - ``32``
     - Batch size
   * - ``batch_on``
     - ``molecules``
     - Unit to batch on (``scenes``, ``molecules``, ``residues``, …)
   * - ``accelerator``
     - ``cpu``
     - Device for ``TorchBackend`` (``cpu``, ``gpu``, ``cuda``, …)
   * - ``validate_every``
     - ``1.0``
     - Run validation every N epochs
   * - ``checkpoint_every``
     - ``1.0``
     - Save checkpoint every N epochs
   * - ``restore_from``
     - ``checkpoint``
     - Checkpoint filename for test evaluation

Running experiments
-------------------

.. code-block:: bash

   bioverse train experiment.yaml
   bioverse val experiment.yaml
   bioverse test experiment.yaml

Commands map to :meth:`~bioverse.trainer.Trainer.run`:

- ``train`` — fit for ``trainer.epochs``; optionally validate each epoch
- ``val`` — evaluate on the validation partition; log metrics
- ``test`` — load ``restore_from`` checkpoint, then evaluate on the test partition

Inspecting data and results
---------------------------

Load a benchmark in Python without training:

.. code-block:: python

   from bioverse.factory import BenchmarkFactory

   benchmark = BenchmarkFactory("B_AFCATH")
   loader = benchmark.loader(
       partition="train",
       batch_size=4,
       batch_on="molecules",
       progress=True,
   )
   for (X, y), data in loader:
       print(y)
       break

After ``bioverse val`` or ``bioverse test``, metrics are printed to the console
and saved under ``trainer.root`` as ``val_results.yaml`` or ``test_results.yaml``.

Common workflows
----------------

**Train from scratch**

.. code-block:: bash

   bioverse train configs/experiment.yaml

**Short dry run** — limit batches while debugging:

.. code-block:: yaml

   trainer:
     limit_train_batches: 5
     epochs: 1

**Evaluate only** — reuse a trained checkpoint:

.. code-block:: bash

   bioverse test experiment.yaml

Ensure ``trainer.restore_from`` points to the saved checkpoint in ``trainer.root``.

**Swap benchmark, keep model** — use separate YAML files merged at the CLI:

.. code-block:: bash

   bioverse train shared/model.yaml benchmarks/pcasso.yaml

Troubleshooting
---------------

**Import errors for ``model``**
  Run ``bioverse`` from the directory containing your model module, or use an
  absolute import path.

**Missing dataset shards**
  The first access triggers download and release. Check ``BIOVERSE_DATASET_ROOT``
  permissions and network access for adapters that fetch remote data.

**Config assertion failures**
  The CLI requires ``trainer``, ``model``, ``benchmark``, and ``transforms`` keys.
  Use an empty list (``transforms: []``) when no offline transforms are needed.

**Out-of-memory errors**
  Reduce ``trainer.batch_size``, use ``batch_on: molecules`` instead of ``residues``,
  or choose a benchmark with smaller structures.
