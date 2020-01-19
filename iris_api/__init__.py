from os import path
import pkg_resources

MODULE_PATH = path.dirname(path.realpath(__file__))
MODEL_PATH = path.join(MODULE_PATH, 'model_store')

VERSION = pkg_resources.get_distribution(__name__).version