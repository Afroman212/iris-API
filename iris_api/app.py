# Dependencies
from flask import Flask, request, jsonify, json
import requests as req
from flask_restplus import Api
import pandas as pd
import numpy as np
from iris_api.model import IrisPipeline
from iris_api import MODEL_PATH, VERSION


model = IrisPipeline()
model.load(directory=MODEL_PATH)
flask_app = Flask(__name__)
app = Api(app=flask_app)


@app.route('/health')
class Health:
    """

    """
    def get(self):
        """

        :return:
        """
        return jsonify({
            "status": {
                "code": 200,
                "status": "SUCCESS"
            }
        })


@app.route('/predict')
class Predict:
    """

    """
    def post(self):
        """

        :return:
        """
        if request.form:
            body = json.loads(request.form.get('json'))
        elif request.is_json:
            body = request.get_json(silent=False)
        else:
            body = None
            raise req.exceptions.RequestException('Missing or invalid request body')

        names = np.array(body['data']['names'])
        data = np.array(body['data']['ndarray'])
        query_df = pd.DataFrame(data=data, columns=names)

        prediction, labels = model.predict(query_df)
        pred_pkgd = [[p] for p in prediction]

        return jsonify({
            "status": {
                "code": 200,
                "status": "SUCCESS"
            },
            "meta": {
                "tags": {
                    "model_version": VERSION
                }
            },
            "data": {
                "names": labels,
                "ndarray": pred_pkgd
            }
        })


if __name__ == '__main__':
    port = 5000
    app.run(port=port, debug=True)
