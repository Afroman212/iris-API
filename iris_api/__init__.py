import os
import pkg_resources

MODULE_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH = os.path.join(MODULE_PATH, 'model_store')

VERSION = pkg_resources.get_distribution(__name__).version