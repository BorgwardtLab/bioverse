Metrics
=======

Metrics score model predictions against ground truth inside a benchmark.
Summaries below are taken from each class docstring.

.. currentmodule:: bioverse.metrics

{% for cls in bioverse.metrics.classes %}
{% if cls.summary %}
{{ cls.summary }}

{% endif %}
.. autoclass:: {{ cls.qualname }}
   :members:
   :show-inheritance:

{% endfor %}
