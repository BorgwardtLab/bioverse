Tasks
=====

Tasks map sampler indices to model inputs and supervision targets. Summaries
below are taken from each class docstring.

.. currentmodule:: bioverse.tasks

{% for cls in bioverse.tasks.classes %}
{% if cls.summary %}
{{ cls.summary }}

{% endif %}
.. autoclass:: {{ cls.qualname }}
   :members:
   :show-inheritance:

{% endfor %}
