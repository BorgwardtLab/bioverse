Adapters
========

Adapters download or generate raw biomolecular data and produce the initial
:class:`~bioverse.data.Batch` stream, :class:`~bioverse.data.Split`, and
:class:`~bioverse.data.Assets` objects consumed by datasets. Summaries below are
taken from each class docstring.

.. currentmodule:: bioverse.adapters

{% for cls in bioverse.adapters.classes %}
{% if cls.summary %}
{{ cls.summary }}

{% endif %}
.. autoclass:: {{ cls.qualname }}
   :members:
   :show-inheritance:

{% endfor %}
