import importlib
import os
import sys

__all__ = [
    fname[:-3]
    for fname in os.listdir(os.path.dirname(__file__))
    if fname.endswith(".py") and fname not in ("__init__.py",)
]  # type: ignore


def __getattr__(name):
    if name in __all__:
        mod = importlib.import_module(f".{name}", __name__)
        try:
            cls = getattr(mod, name)
            setattr(sys.modules[__name__], name, cls)
            return cls
        except AttributeError as e:
            raise ImportError(
                f"Cannot import name '{name}' from '{mod.__name__}': {e}"
            ) from e
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
