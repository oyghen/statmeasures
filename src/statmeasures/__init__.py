__all__ = ["__version__", "center", "spread", "utils"]

from importlib import metadata

from statmeasures import center, spread, utils

__version__ = metadata.version(__name__)
