"""Initialize main package."""
import pkg_resources
from speech_datasets.dataloader import SpeechDataLoader

try:
    __version__ = pkg_resources.get_distribution("speech_datasets").version
except Exception:
    __version__ = "(Not installed from setup.py)"
del pkg_resources
