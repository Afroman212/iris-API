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
def test_pickle_mod():
    try:
        os.mkdir('temp')
    except FileExistsError:
        pass
    read_data = pickle.dump(np.zeros(10), open('./temp/IrisModel.pkl', 'wb'))
    return read_data


@pytest.fixture(scope="session")
def test_pickle_le():
    try:
        os.mkdir('temp')
    except FileExistsError:
        pass
    read_data = pickle.dump(np.zeros(10), open('./temp/Le.pkl', 'wb'))
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


def test_load(test_pickle_mod, test_pickle_le):
    model = IrisPipeline()
    model.load('./temp/')


def test_save(test_data, test_pickle_mod):
    model = IrisPipeline()
    X = test_data.drop(['target'], axis=1)
    y = test_data['target']
    model.fit(X, y)
    model.save('./temp/')


def test_predict(test_data):
    model = IrisPipeline()
    model.load(directory=MODEL_PATH)
    X = test_data.drop(['target'], axis=1)
    y_pred = test_data['target'].values
    prediction, label = model.predict(X)
    
    np.testing.assert_array_equal(prediction, y_pred)
    assert label[0] == 'Iris'
