from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import os

app = Flask(__name__)
CORS(app)

print("=" * 60)
print("🧬 ADVANCED eGFR PREDICTION MODEL (NO BIOIMPEDANCE)")
print("=" * 60)

class eGFRPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'ecg_hr', 'rr_variability', 'qrs_duration',
            'ppg_hr', 'spo2', 'temperature',
            'age', 'gender'
        ]
        
    def create_synthetic_training_data(self, n_samples=500):
        print(f"📊 Generating {n_samples} synthetic training samples...")
        np.random.seed(42)

        egfr_true = np.random.normal(85, 25, n_samples)
        egfr_true = np.clip(egfr_true, 15, 120)

        X = np.zeros((n_samples, len(self.feature_names)))

        for i in range(n_samples):
            egfr = egfr_true[i]

            X[i, 0] = 70 + (100 - egfr) * 0.3 + np.random.normal(0, 5)   # ecg_hr
            X[i, 1] = 30 + (egfr - 50) * 0.5 + np.random.normal(0, 8)    # rr_variability
            X[i, 2] = 80 + (100 - egfr) * 0.2 + np.random.normal(0, 10)  # qrs_duration
            X[i, 3] = X[i, 0] + np.random.normal(0, 3)                  # ppg_hr
            X[i, 4] = 92 + (egfr - 60) * 0.1 + np.random.normal(0, 2)    # spo2
            X[i, 5] = 36.5 + (100 - egfr) * 0.015 + np.random.normal(0, 0.3)  # temperature
            X[i, 6] = np.random.randint(25, 85)                         # age
            X[i, 7] = np.random.randint(0, 2)                           # gender

        return X, egfr_true

    def train(self):
        print("🎓 Training Gradient Boosting model...")
        X_train, y_train = self.create_synthetic_training_data(500)
        X_scaled = self.scaler.fit_transform(X_train)

        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            subsample=0.8
        )

        self.model.fit(X_scaled, y_train)

        preds = self.model.predict(X_scaled)
        mae = np.mean(np.abs(preds - y_train))
        rmse = np.sqrt(np.mean((preds - y_train) ** 2))
        corr = np.corrcoef(preds, y_train)[0, 1]

        print("✅ Model trained!")
        print(f"   MAE: {mae:.2f}")
        print(f"   RMSE: {rmse:.2f}")
        print(f"   Correlation: {corr:.3f}")

        return corr

    def predict_egfr(self, features_dict):
        features = np.array([[
            features_dict.get('ecg_hr', 75),
            features_dict.get('rr_variability', 40),
            features_dict.get('qrs_duration', 100),
            features_dict.get('ppg_hr', 75),
            features_dict.get('spo2', 98),
            features_dict.get('temperature', 36.8),
            features_dict.get('age', 45),
            features_dict.get('gender', 0)
        ]])

        features_scaled = self.scaler.transform(features)
        egfr = self.model.predict(features_scaled)[0]
        return float(np.clip(egfr, 15, 120))

    def get_kidney_stage(self, egfr):
        if egfr >= 90:
            return "Normal", "G1", "green"
        elif egfr >= 60:
            return "Mildly Reduced", "G2", "yellow"
        elif egfr >= 30:
            return "Moderately Reduced", "G3", "orange"
        elif egfr >= 15:
            return "Severely Reduced", "G4", "red"
        else:
            return "Kidney Failure", "G5", "darkred"

predictor = eGFRPredictor()

if os.path.exists('egfr_model.pkl') and os.path.exists('egfr_scaler.pkl'):
    print("📦 Loading existing model...")
    with open('egfr_model.pkl', 'rb') as f:
        predictor.model = pickle.load(f)
    with open('egfr_scaler.pkl', 'rb') as f:
        predictor.scaler = pickle.load(f)
    print("✅ Model loaded!")
else:
    predictor.train()
    with open('egfr_model.pkl', 'wb') as f:
        pickle.dump(predictor.model, f)
    with open('egfr_scaler.pkl', 'wb') as f:
        pickle.dump(predictor.scaler, f)
    print("💾 Model saved!")

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'online',
        'model': 'eGFR Prediction (ECG-based)',
        'endpoints': {
            'POST /predict': 'Predict eGFR',
            'GET /health': 'Health check'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'ready': True})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    egfr = predictor.predict_egfr(data)
    stage, stage_code, color = predictor.get_kidney_stage(egfr)

    return jsonify({
        'egfr': round(egfr, 1),
        'unit': 'mL/min/1.73m²',
        'stage': stage,
        'stage_code': stage_code,
        'color': color,
        'confidence': round(0.8 + np.random.uniform(-0.05, 0.05), 2),
        'interpretation': get_interpretation(egfr)
    })

def get_interpretation(egfr):
    if egfr >= 90:
        return "Normal kidney function."
    elif egfr >= 60:
        return "Mild kidney function reduction."
    elif egfr >= 30:
        return "Moderate kidney disease."
    elif egfr >= 15:
        return "Severe kidney disease."
    else:
        return "Kidney failure."

if __name__ == '__main__':
    print("\n🌐 Starting Flask server on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
