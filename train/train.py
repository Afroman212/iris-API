from iris_api import MODEL_PATH
import pandas as pd
from sklearn import datasets
from iris_api.model import IrisPipeline
import numpy as np

iris = datasets.load_iris()
df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                  columns=iris['feature_names'] + ['target'])

converter = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
df.replace({'target': converter}, inplace=True)

model = IrisPipeline()
X = df.drop(['target'], axis=1)
y = df['target']
model.fit(X, y)
model.save(directory=MODEL_PATH)
