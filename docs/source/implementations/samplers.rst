Samplers
========

Samplers decide which scenes, frames, molecules, or residues appear in each
batch. Summaries below are taken from each class docstring.

.. currentmodule:: bioverse.samplers

{% for cls in bioverse.samplers.classes %}
{% if cls.summary %}
{{ cls.summary }}

{% endif %}
.. autoclass:: {{ cls.qualname }}
   :members:
   :show-inheritance:

{% endfor %}
