Quickstart
==========

What is Bioverse?
-----------------

Bioverse is a framework for running standardized machine-learning experiments on
biomolecular data. You pick a **benchmark** (dataset, sampling strategy, task,
and metric), attach **transforms** and a **model**, and run training or evaluation
through a single configuration file and CLI.

Installation
------------

Install Bioverse from PyPI:

.. code-block:: bash

   pip install bioverse-ml

Training requires PyTorch and Lightning Fabric, which are not installed by default:

.. code-block:: bash

   pip install torch lightning

Before you run
--------------

By default, Bioverse stores data under ``~/.bioverse``. You can override paths
with environment variables such as ``BIOVERSE_ROOT``, ``BIOVERSE_DATASET_ROOT``,
and ``BIOVERSE_BENCHMARKS_ROOT``. See the :doc:`User Guide <guides/user>` for
details.

Minimal experiment
------------------

The example below loads ``B_AFCATH``, an inverse-folding benchmark (see
:doc:`../implementations/benchmarks`), and inspects one training batch. Full
training requires a model whose ``train_step`` / ``eval_step`` match the task.

.. code-block:: python

   from bioverse.factory import BenchmarkFactory

   benchmark = BenchmarkFactory("B_AFCATH")
   loader = benchmark.loader(
       partition="train",
       batch_size=2,
       batch_on="molecules",
       progress=True,
   )
   (X, y), data = next(iter(loader))
   print(X.resolution, y.fields)

To run training from the CLI, create an ``experiment.yaml`` with a task-compatible
model and trainer settings:

.. code-block:: yaml

   globals:
     workers: 1

   benchmark: B_AFCATH
   transforms: []

   model: .model.MyInverseFolder

   trainer:
     backend: TorchBackend
     collater: LongCollater
     logger: NoLogger
     root: results/afcath
     model_name: MyInverseFolder
     epochs: 3
     batch_size: 4
     batch_on: molecules
     accelerator: cpu
     log_every: 1
     limit_train_batches: 10

Run training from the directory that contains your model module:

.. code-block:: bash

   bioverse train experiment.yaml

Evaluate on the validation split:

.. code-block:: bash

   bioverse val experiment.yaml

What happened?
--------------

The CLI merged your YAML config, loaded the built-in ``B_AFCATH`` benchmark,
instantiated your model from the local module, and handed both to
:class:`~bioverse.trainer.Trainer`. The four required top-level keys are:

- ``benchmark`` — a built-in name (``B_*``) or path to a benchmark YAML file
- ``transforms`` — list of offline transforms applied when the dataset is built
- ``model`` — import path to your model class (``.`` prefix searches the current directory)
- ``trainer`` — backend, logging, batching, and training hyperparameters

Next steps
----------

- :doc:`User Guide <guides/user>` — full config reference and CLI workflows
- :doc:`Implementations <implementations/benchmarks>` — browse built-in benchmarks
- :doc:`Developer Guide <guides/developer>` — architecture and extending Bioverse
