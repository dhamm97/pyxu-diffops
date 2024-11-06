import importlib.metadata

try:
    __version__ = importlib.metadata.version("pyxu_diffops")
except ImportError:
    __version__ = "unknown"
