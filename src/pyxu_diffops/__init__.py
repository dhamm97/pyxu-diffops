import importlib.metadata

try:
    __version__ = importlib.metadata.version("pyxu_diffops")
except ImportError:
    __version__ = "unknown"


from .operator import Flip, NullFunc


__all__ = (
    "Flip",
    "NullFunc",
)
