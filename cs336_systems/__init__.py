import importlib.metadata

try:
    __version__ = importlib.metadata.version("cs336_systems")
except importlib.metadata.PackageNotFoundError:
    __version__ = "dev"
