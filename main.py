from flask import Flask, request, jsonify
import joblib
import numpy as np

# Inisialisasi Flask app
app = Flask(__name__)

# Load model
random_forest_model = joblib.load('random_forest_model.pkl')
gbm_model = joblib.load('gbm_model.pkl')
svm_model = joblib.load('svm_model.pkl')

@app.route('/predict/rf', methods=['POST'])
def predict_rf():
    data = request.json  # Mengambil data JSON dari request
    features = np.array(data['features']).reshape(1, -1)  # Mengubah data menjadi array numpy
    prediction = random_forest_model.predict(features)
    return jsonify({'prediction': prediction[0]})

@app.route('/predict/gbm', methods=['POST'])
def predict_gbm():
    data = request.json
    features = np.array(data['features']).reshape(1, -1)
    prediction = gbm_model.predict(features)
    return jsonify({'prediction': prediction[0]})

@app.route('/predict/svm', methods=['POST'])
def predict_svm():
    data = request.json
    features = np.array(data['features']).reshape(1, -1)
    prediction = svm_model.predict(features)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
