.. image:: ../../assets/bioverse_logo.png
   :alt: Bioverse
   :align: left
   :width: 360px

|

*Machine learning on biomolecules*

Bioverse is a standardized framework for machine-learning experiments on
biomolecules — proteins, RNA, small molecules, and related structures. It
combines reusable **benchmarks**, **transforms**, and **evaluation components**
with a config-driven CLI so you can train and evaluate models on common tasks
without reimplementing data loading, splitting, or metrics.

What you can do
---------------

- Run an experiment from a YAML config and the ``bioverse`` CLI
- Browse built-in datasets, benchmarks, metrics, tasks, and transforms
- Extend the library with new components and contribute them upstream

Get started
-----------

.. code-block:: bash

   pip install bioverse-ml
   pip install torch lightning   # required for training
   bioverse train experiment.yaml

See :doc:`quickstart` for a complete minimal example (model + config + command).

Choose your path
----------------

**Run experiments with existing components**
  Start with :doc:`quickstart`, then read the :doc:`guides/user` for configuration,
  CLI workflows, and troubleshooting.

**Understand the architecture and add components**
   Read the :doc:`guides/developer`, then follow the **How to** tutorials for
   datasets, benchmarks, metrics, and other component types.

**Contribute to the repository**
  See the :doc:`guides/contributor` for naming conventions, tests, and pull-request
  expectations.

How it works
------------

.. code-block:: text

   Adapter  →  Dataset  →  Benchmark  →  Trainer + Model
                          (sampler,
                           task,
                           metric)

1. An **adapter** downloads or generates raw data.
2. A **dataset** stores versioned shards and splits (optionally via transforms).
3. A **benchmark** wires the dataset to a sampler, task, and metric.
4. The **trainer** runs your model and logs results.

Details: :doc:`guides/developer` and :doc:`api_reference/code_structure`.

Built-in library
----------------

Bioverse ships with a growing catalog of ready-to-use components. Browse the
**Implementations** section for auto-generated reference pages with docstrings
and configs.

Featured benchmarks:

- :ref:`B_AFCATH` — inverse folding on CATH domains with recovery and BLOSUM metrics
- :ref:`B_PROGYM` — protein mutational effect prediction (ProteinGym)
- :ref:`B_INVC42` — inverse folding on CATH structures

See :doc:`implementations/benchmarks` and :doc:`implementations/datasets` for the
full lists.

Project links
-------------

- `Source code <https://github.com/BorgwardtLab/bioverse>`_
- `Documentation <https://borgwardtlab.github.io/bioverse/>`_
- `PyPI package <https://pypi.org/project/bioverse-ml/>`_
- `Issue tracker <https://github.com/BorgwardtLab/bioverse/issues>`_
- :doc:`citation` — how to cite Bioverse
- `License <https://github.com/BorgwardtLab/bioverse/blob/main/LICENSE>`_ (BSD-3-Clause)

Citation
--------

.. code-block:: bibtex

   @software{bioverse2026,
     author = {Kucera, Tim and Bioverse Contributors},
     title = {Bioverse: A standardized framework for machine learning on biomolecules},
     year = {2026},
     url = {https://github.com/BorgwardtLab/bioverse}
   }

See :doc:`citation` for the full reference and author contact details.

----

.. toctree::
   :maxdepth: 1
   :hidden:

   quickstart
   citation

.. toctree::
   :maxdepth: 1
   :caption: Guides

   guides/user
   guides/developer
   guides/contributor

.. toctree::
   :maxdepth: 1
   :caption: How to

   Create a dataset <tutorials/make_your_own_dataset>
   Create a benchmark <tutorials/make_your_own_benchmark>
   Create a metric <tutorials/make_your_own_metric>
   Create a sampler <tutorials/make_your_own_sampler>
   Create a processor <tutorials/make_your_own_processor>
   Create a task <tutorials/make_your_own_task>
   Create a transform <tutorials/make_your_own_transform>

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api_reference/code_structure
   api_reference/adapter
   api_reference/processor
   api_reference/data
   api_reference/dataset
   api_reference/transform
   api_reference/benchmark
   api_reference/sampler
   api_reference/task
   api_reference/metric
   api_reference/utilities

.. toctree::
   :maxdepth: 1
   :caption: Implementations

   implementations/adapters
   implementations/processors
   implementations/datasets
   implementations/transforms
   implementations/benchmarks
   implementations/samplers
   implementations/tasks
   implementations/metrics
