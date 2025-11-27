
__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from src.preprocessing import AudioPreprocessor, CLASS_NAMES
from src.model import SoundClassifier, train_model
from src.prediction import SoundPredictor

__all__ = [
    'AudioPreprocessor',
    'SoundClassifier',
    'SoundPredictor',
    'train_model',
    'CLASS_NAMES'
]