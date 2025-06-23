Adapter
=======

Adapters provide an interface to raw data repositories. They provide the download logic to place structure files and meta data to a local directory (:py:data:`~bioverse.utilities.config.raw_path`). After download, they construct the initial :py:class:`~bioverse.data.Data`-iterator, as well as :py:class:`~bioverse.data.Split` and :py:class:`~bioverse.data.Assets` objects. Usually, Adapters make use of a :py:class:`~bioverse.processor.Processor` to convert raw structure files to :py:class:`~bioverse.data.Data` objects.

.. automodule:: bioverse.adapter
   :members: Adapter
