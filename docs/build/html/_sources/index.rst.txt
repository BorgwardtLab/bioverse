.. Bioverse documentation master file, created by
   sphinx-quickstart on Tue Apr  1 19:56:29 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Bioverse documentation
======================

Add your content using ``reStructuredText`` syntax. See the
`reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
documentation for details.


.. toctree::
   :glob:
   :maxdepth: 1

   quickstart

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Guides

   guides/user
   guides/developer
   guides/contributor

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: How to...

   tutorials/make_your_own_dataset

API Reference
-------------

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
   api_reference/framework
   api_reference/utilities

Implementations
---------------

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
   implementations/frameworks
