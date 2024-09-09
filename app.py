import numpy as np
import joblib
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the model and scaler
model = load_model('best_model.keras')
scaler = joblib.load('scaler.pkl')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [
        float(request.form['goldDiff']),
        float(request.form['xpDiff']),
        float(request.form['dragonDiff']),
        float(request.form['heraldDiff']),
        float(request.form['towerDiff']),
        float(request.form['inhibitorDiff']),
        float(request.form['turretPlateDiff']),
        float(request.form['minionDiff']),
        float(request.form['jungleMinionDiff']),
        float(request.form['damageDiff']),
        float(request.form['firstBlood'])
    ]
    
    final_features = np.array([features])

    # Scale the features using the loaded scaler
    final_features_scaled = scaler.transform(final_features)

    # Make prediction
    prediction_prob = model.predict(final_features_scaled)[0][0]
    prediction = int(prediction_prob > 0.5)

    output = 'Blue Team Win' if prediction == 1 else 'Red Team Win'

    return render_template('result.html', prediction=output, probability=f'{prediction_prob:.2f}')

if __name__ == "__main__":
    app.run(debug=True)
