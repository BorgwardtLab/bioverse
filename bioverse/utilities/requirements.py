import importlib
import shutil


def requires(python_packages=None, system_packages=None):
    """
    Decorator to check if the required packages are installed.

    Args:
        python_packages: List of required Python package names
        system_packages: List of required system package names

    Raises:
        ImportError: If a required Python package is not installed
        RuntimeError: If a required system package is not found
    """
    if python_packages is None:
        python_packages = []
    if system_packages is None:
        system_packages = []

    def decorator(cls):
        # Check Python packages
        for package in python_packages:
            try:
                importlib.import_module(package)
            except ImportError:
                raise ImportError(
                    f"Required Python package '{package}' is not installed."
                )

        # Check system packages
        for package in system_packages:
            if shutil.which(package) is None:
                raise RuntimeError(
                    f"Required system package '{package}' was not found."
                )

        return cls

    return decorator
