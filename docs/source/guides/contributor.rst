Contributor Guide
=================

Overview
--------

Thank you for contributing to Bioverse. This guide describes the conventions and
checklist for submitting **new components** — adapters, datasets, benchmarks,
transforms, tasks, samplers, metrics, or processors — to the repository.

For local prototyping without upstreaming, the :doc:`Developer Guide <developer>`
is sufficient.

Before you contribute
---------------------

- Open an issue or discussion for large additions (new benchmark suites, heavy
  dependencies, or breaking API changes)
- Prefer reusable components over one-off experiment scripts
- Do not commit raw data, credentials, or large binary artifacts

Repository conventions
----------------------

Naming
~~~~~~

.. list-table::
   :header-rows: 1

   * - Artifact
     - Pattern
     - Example
   * - Dataset config
     - ``D_<SHORT>.yaml``
     - ``D_AFCATH.yaml``
   * - Benchmark config
     - ``B_<SHORT>.yaml``
     - ``B_AFCATH.yaml``
   * - Adapter
     - ``<Name>Adapter``
     - ``AlphaFoldInvBenchAdapter``
   * - Metric
     - ``<Name>Metric``
     - ``BinaryAccuracyMetric``
   * - Task
     - ``<Name>Task``
     - ``PropertyPredictionTask``
   * - Sampler
     - ``<Name>Sampler``
     - ``MoleculeSampler``
   * - Processor
     - ``<Format>Processor``
     - ``PdbProcessor``
   * - Transform
     - PascalCase, no suffix required
     - ``FilterSequenceLength``

File placement
~~~~~~~~~~~~~~

- Python implementations: ``bioverse/<subpackage>/<snake_case>.py``
- Dataset YAML: ``bioverse/datasets/``
- Benchmark YAML: ``bioverse/benchmarks/``
- Tests: ``tests/test_<topic>.py``

Code style
----------

The project uses:

- **Black** (line length 88)
- **isort** (profile ``black``)
- **flake8** for linting
- **mypy** with strict settings (see ``pyproject.toml``)

Format before submitting:

.. code-block:: bash

   pip install -e ".[dev]"
   black bioverse tests
   isort bioverse tests

Component requirements
----------------------

Every new component should:

- Subclass the correct ABC (where applicable) and implement required methods
- Include a class docstring describing purpose and data expectations
- Avoid hard-coded absolute paths; use :mod:`bioverse.utilities.config`
- Be discoverable by the lazy-import pattern (module filename ↔ class name)
- Ship with tests that exercise the public contract

Additional expectations by type:

**Adapter**
  ``download()`` returns ``(batches, split, assets)``; idempotent or versioned downloads

**Dataset YAML**
  Exactly one of ``adapter`` or ``parent``; optional ``transforms`` list

**Benchmark YAML**
  Valid ``dataset``, ``sampler``, ``task``, and ``metric`` references

**Metric**
  Implement ``compute()``; set ``better`` to ``"higher"`` or ``"lower"``

**Transform**
  Implement ``transform_batch()`` or ``transform()``; document ``filter`` behaviour if used

YAML configs
------------

Dataset example:

.. code-block:: yaml

   description: CATH domain structures for inverse folding evaluation.
   adapter:
     AlphaFoldInvBenchAdapter:
       af_name: swissprot_pdb
       af_version: v4

Benchmark example:

.. code-block:: yaml

   description: Inverse folding on CATH domains.
   dataset: D_AFCATH
   sampler: MoleculeSampler
   task: InverseFoldingTask
   metric: RecoveryMetric

Pass kwargs to components with a single-key dict:

.. code-block:: yaml

   task:
     PropertyPredictionTask:
       property: mu

YAML files are packaged with the library (see ``pyproject.toml`` ``package-data``).

Tests
-----

- Place tests in ``tests/``
- Use pytest fixtures for expensive setup (see ``tests/test_progym.py``)
- Prefer built-in benchmarks where possible; mock or subset large downloads
- Mark slow or integration tests with ``@pytest.mark.slow`` or
  ``@pytest.mark.integration``

Run the test suite:

.. code-block:: bash

   pytest tests/

Documentation
-------------

- Add docstrings to new classes; Napoleon renders them on implementation pages
- For new component **types** or non-obvious workflows, extend the matching
  :doc:`../tutorials/make_your_own_dataset` tutorial page
- Verify docs build: ``cd docs && make html``

Pull request checklist
----------------------

- [ ] Code formatted with Black and isort
- [ ] Tests added or updated; ``pytest`` passes
- [ ] YAML configs named ``D_*`` / ``B_*`` and referenced correctly
- [ ] No secrets, large data, or unrelated changes
- [ ] Docstrings present on new public classes
- [ ] Docs build without errors

Review criteria
---------------

Reviewers will look for:

- **Reusability** — useful beyond a single paper or experiment
- **Consistency** — naming, patterns, and config style match existing components
- **Correctness** — tests cover splits, shapes, and metric behaviour
- **Dependencies** — new packages justified and added to ``pyproject.toml`` if required
- **Backward compatibility** — existing benchmarks and configs keep working

Release and packaging
---------------------

Version numbers are managed with **setuptools-scm** from git tags. YAML configs
and package data ship automatically with ``bioverse-ml`` on PyPI. Maintainers cut
releases via the GitHub publish workflow when tagged versions are pushed.
