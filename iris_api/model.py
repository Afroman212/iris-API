import pandas as pd
from os import path
from os import mkdir
import joblib
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import requests


class IrisPipeline(BaseEstimator, ClassifierMixin):
    """

    """
    def __init__(self):
        self.model = None
        self.le = LabelEncoder()

    @staticmethod
    def validate_input(x_var):
        """

        :param x_var:
        :return:
        """
        expected_cols = {'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'}
        missing_cols = expected_cols.difference(x_var.keys())
        if missing_cols:
            raise requests.exceptions.RequestException(f'Missing inputs: {missing_cols}')

    def fit(self, x_var, y_var):
        """

        :param x_var:
        :param y_var:
        :return:
        """
        self.validate_input(x_var)
        self.le.fit(y_var)
        y_var_adjusted = self.le.transform(y_var)
        self.model = Pipeline([
            (),
            (),
            ()
        ])

        self.model.fit(x_var, y_var_adjusted)

    def predict(self, x_var):
        """

        :param x_var:
        :return:
        """
        self.validate_input(x_var)
        pred = self.model.predict(x_var)
        prediction = self.le.inverse_transform(pred)
        return prediction, ['Iris']

    def save(self, directory):
        """

        :param directory:
        :return:
        """
        if not path.exists(directory):
            mkdir(directory)
        joblib.dump(self.model, path.join(directory, 'IrisModel.pkl'))

    def load(self, directory):
        """

        :param directory:
        :return:
        """
        self.model = joblib.load(path.join(directory, 'IrisModel.pkl'))
