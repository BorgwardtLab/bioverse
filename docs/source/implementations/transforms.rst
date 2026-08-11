Transforms
==========

Transforms modify dataset shards, splits, and assets offline or at load time.
Summaries below are taken from each class docstring.

.. currentmodule:: bioverse.transforms

{% for cls in bioverse.transforms.classes %}
{% if cls.summary %}
{{ cls.summary }}

{% endif %}
.. autoclass:: {{ cls.qualname }}
   :members:
   :show-inheritance:

{% endfor %}
