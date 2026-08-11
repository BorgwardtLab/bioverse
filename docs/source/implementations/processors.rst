Processors
==========

Processors parse structure files on disk (PDB, mmCIF, SDF, …) into Awkward
records during adapter ingestion. Summaries below are taken from each class
docstring.

.. currentmodule:: bioverse.processors

{% for cls in bioverse.processors.classes %}
{% if cls.summary %}
{{ cls.summary }}

{% endif %}
.. autoclass:: {{ cls.qualname }}
   :members:
   :show-inheritance:

{% endfor %}
