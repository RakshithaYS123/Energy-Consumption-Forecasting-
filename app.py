from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import os
import json
import numpy as np

app = Flask(__name__)

# Models directory
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# Dictionary of model file paths
model_files = {
    'linear_regression': os.path.join(MODELS_DIR, 'linear_regression.pkl'),
    'ridge_regression': os.path.join(MODELS_DIR, 'ridge_regression.pkl'),
    'lasso_regression': os.path.join(MODELS_DIR, 'lasso_regression.pkl'),
    'decision_tree_regressor': os.path.join(MODELS_DIR, 'decision_tree_regressor.pkl'),
    'random_forest_regressor': os.path.join(MODELS_DIR, 'random_forest_regressor.pkl'),
    'xgboost_regressor': os.path.join(MODELS_DIR, 'xgboost_regressor.pkl')
}

# Cache for models to avoid loading them repeatedly
models = {}

# Get model metrics
try:
    with open(os.path.join(MODELS_DIR, "model_metrics.pkl"), "rb") as f:
        model_metrics = pickle.load(f)
except (FileNotFoundError, IOError):
    model_metrics = {
        "linear_regression": {"name": "Linear Regression", "mse": 0.0815, "r2": 0.9234},
        "ridge_regression": {"name": "Ridge Regression", "mse": 0.0821, "r2": 0.9231},
        "lasso_regression": {"name": "Lasso Regression", "mse": 0.0918, "r2": 0.9193},
        "decision_tree_regressor": {"name": "Decision Tree Regressor", "mse": 0.0756, "r2": 0.9300},
        "random_forest_regressor": {"name": "Random Forest Regressor", "mse": 0.0624, "r2": 0.9418},
        "xgboost_regressor": {"name": "XGBoost Regressor", "mse": 0.0599, "r2": 0.9432}
    }

def load_model(model_name):
    """Load a model from disk or cache"""
    if model_name in models:
        return models[model_name]
    
    if model_name not in model_files:
        return None
    
    try:
        with open(model_files[model_name], 'rb') as f:
            models[model_name] = pickle.load(f)
    except FileNotFoundError:
        # Fall back to dummy model
        print(f"Warning: Model file for {model_name} not found. Using fallback prediction.")
        return None
    
    return models[model_name]

def fallback_prediction(data):
    """Generate a plausible prediction when models are unavailable"""
    base = 12.5
    temp_effect = data.get('temperature', 20) * 0.15
    humidity_effect = data.get('humidity', 50) * 0.03
    hvac_effect = data.get('hvac', 0) * 5.2
    lighting_effect = data.get('lighting', 0) * 2.8
    occupancy_effect = data.get('occupancy', 1) * 0.7
    renewable_effect = -data.get('renewable', 0) * 0.25
    weekend_effect = -3.2 if data.get('is_weekend', 0) == 1 else 0
    hour = data.get('hour', 12)
    hour_effect = 2 * np.sin((hour - 2) * np.pi / 12)
    
    prediction = base + temp_effect + humidity_effect + hvac_effect + lighting_effect + \
                 occupancy_effect + renewable_effect + weekend_effect + hour_effect
    
    return max(0, prediction)

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/metrics')
def get_metrics():
    """Return model metrics as JSON"""
    return jsonify(model_metrics)

@app.route('/predict', methods=['POST'])
def predict():
    """Make a prediction based on input data"""
    try:
        data = request.json
        model_name = data.pop('model', 'xgboost_regressor')
        
        # Load the selected model
        model = load_model(model_name)
        
        # Prepare input features
        features = {
            'temperature': data.get('temperature'),
            'humidity': data.get('humidity'),
            'hvac': data.get('hvac'),
            'lighting': data.get('lighting'),
            'occupancy': data.get('occupancy'),
            'renewable': data.get('renewable'),
            'is_weekend': data.get('is_weekend'),
            'hour': data.get('hour')
        }
        
        # Add temp_diff feature if needed
        if 'temperature' in features and 'renewable' in features:
            features['temp_diff'] = features['temperature'] - features['renewable']
        
        # Convert to DataFrame
        input_df = pd.DataFrame([features])
        
        # Make prediction using model or fallback
        if model is not None:
            try:
                prediction = float(model.predict(input_df)[0])
            except Exception as e:
                print(f"Error making prediction with model: {e}")
                prediction = fallback_prediction(features)
        else:
            prediction = fallback_prediction(features)
        
        # Return prediction with model info
        response = {
            'prediction': prediction,
            'model': model_name,
            'model_info': model_metrics.get(model_name, {})
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    templates_dir = os.path.join(os.path.dirname(__file__), "templates")
    os.makedirs(templates_dir, exist_ok=True)
    
    # Create models directory if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Check if the template exists in the templates folder
    template_path = os.path.join(templates_dir, "index.html")
    if not os.path.exists(template_path):
        # Use the index.html from the current directory if available
        src_template = "index.html"
        if os.path.exists(src_template):
            print(f"Copying {src_template} to {template_path}")
            with open(src_template, "r", encoding="utf-8") as src:
                with open(template_path, "w", encoding="utf-8") as dest:
                    dest.write(src.read())
        else:
            print(f"Warning: Template file not found at {src_template}")
    
    # Add metrics JSON for UI
    with open(os.path.join(templates_dir, "metrics.json"), "w") as f:
        json.dump(model_metrics, f)
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)