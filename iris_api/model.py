import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


class IrisPipeline(BaseEstimator, ClassifierMixin):
    """

    """
    def __init__(self):
