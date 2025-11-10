"""Training loops and procedures"""

from .training import train_model
from .testing import testing, evaluation
from .training_definitions import *

__all__ = ['train_model', 'testing', 'evaluation']
