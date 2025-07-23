

__version__ = "0.1.2"
__author__ = "ZMLiang <lzzzmmgpt@gmail.com>"

from .main import main
from .trainer import run 
from .model import Model, Encoder


# from cellniche import *
__all__ = ["main", "run", "Model", "Encoder"]
