from flask import Flask, request, jsonify
import numpy as np
from tensorflow import keras
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the model from the saved files
model = keras.models.model_from_json(open('./models/diabetes-prediction-model.json').read())
model.load_weights('./models/diabetes-prediction-model.hdf5')


# Define a REST API endpoint for making predictions
@app.route('/api/v1/predict', methods=['POST'])
def predict():
    data = request.get_json()
    data_num = map(float, list(data.values()))
    X = np.array([*data_num])
    y_pred = model.predict(X.reshape(1, -1)).flatten().tolist()
    return jsonify({'prediction': y_pred})


if __name__ == '__main__':
    app.run(debug=True)
