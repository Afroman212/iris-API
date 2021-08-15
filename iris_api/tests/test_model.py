import pytest
import os
from shutil import rmtree
import pandas as pd
import numpy as np
import pickle
from sklearn import datasets
from iris_api.model import IrisPipeline
from iris_api import MODULE_PATH


def teardown():
    try:
        rmtree('temp')
    except FileNotFoundError:
        pass


@pytest.fixture
def test_data():
    df = pd.read_csv(MODULE_PATH + '/tests/test_data.csv')
    return df


@pytest.fixture(scope="session")
def test_pickle():
    os.mkdir('temp')
    read_data = pickle.dump(np.zeros(10), open('./temp/IrisModel.pkl', 'wb'))
    return read_data


def test_class_init():
    model = IrisPipeline()
    assert model.model == None


def test_validate_pass():
    model = IrisPipeline()
    vars_t = [
        'sepal length (cm)',
        'sepal width (cm)',
        'petal length (cm)',
        'petal width (cm)'
    ]

    data_t = np.zeros((4, 4))
    df_t = pd.DataFrame(data=data_t, columns=vars_t)

    assert model.validate_input(df_t) == None


def test_validate_fail():
    with pytest.raises(Exception) as error:
        model = IrisPipeline()
        vars_f = [
            'sepal length (cm)',
            'sepal width (cm)',
            'petal length (cm)',
        ]

        data_f = np.zeros((4, 3))
        df_f = pd.DataFrame(data=data_f, columns=vars_f)

        assert model.validate_input(df_f)


def test_fit(test_data):
    model = IrisPipeline()
    X = test_data.drop(['target'], axis=1)
    y = test_data['target']
    model.fit(X, y)


def test_load(test_pickle):
    model = IrisPipeline()
    model.load('./')


def test_save(test_data, test_pickle):
    model = IrisPipeline()
    X = test_data.drop(['target'], axis=1)
    y = test_data['target']
    model.fit(X, y)
    model.save('./temp/')