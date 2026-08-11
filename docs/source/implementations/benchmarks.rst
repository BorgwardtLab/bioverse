Benchmarks
==========

Benchmark configs (``B_*.yaml``) wire a dataset to a sampler, task, and metric.
Each config may include ``description`` and ``citation`` keys for documentation
(ignored at runtime). ``citation`` may be a string or a list of strings when
multiple references apply.

{% for item in bioverse.benchmarks.configs %}
.. _{{ item.name }}:

{{ item.name }}
{{ "~" * item.name|length }}

{% if item.description %}{{ item.description }}

{% endif %}{% if item.citations %}
.. admonition:: Please cite
   :class: tip

{% for citation in item.citations %}{{ citation | indent(3, first=True) }}

{% endfor %}{% endif %}{% if item.config_preview %}
.. code-block:: yaml

{{ item.config_preview | indent(3, first=True) }}
{% endif %}
{% endfor %}
