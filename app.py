from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
import os

# Initialize the Flask application
app = Flask(__name__)

# --- Get the directory where this script (app.py) is located ---
script_dir = os.path.dirname(os.path.abspath(__file__))

# --- Construct the full paths to the .pkl files ---
model_path = os.path.join(script_dir, 'model.pkl')
scaler_path = os.path.join(script_dir, 'scaler.pkl')

# Load the trained model and scaler using the constructed paths
try:
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    print(f"SUCCESS: Loaded model from {model_path}")
    print(f"SUCCESS: Loaded scaler from {scaler_path}")

except FileNotFoundError:
    print(f"Error: model.pkl or scaler.pkl not found.")
    print(f"Attempted to load model from: {model_path}")
    print(f"Attempted to load scaler from: {scaler_path}")
    model = None
    scaler = None
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

EXPECTED_FEATURE_ORDER = ['LT', 'LB', 'JKT', 'JKM', 'GRS']

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = None
    calculation_details = None # Initialize calculation_details

    if request.method == 'POST':
        if model is None or scaler is None:
            prediction_text = "Error: Model or scaler not loaded. Please check server logs for details."
            return render_template('index.html', prediction_text=prediction_text, calculation_details=calculation_details)
        try:
            # Get data from the form
            lt = float(request.form['LT'])
            lb = float(request.form['LB'])
            jkt = int(request.form['JKT'])
            jkm = int(request.form['JKM'])
            grs = int(request.form['GRS'])

            # Create a DataFrame for the input features IN THE CORRECT ORDER
            input_features_df = pd.DataFrame([[lt, lb, jkt, jkm, grs]], columns=EXPECTED_FEATURE_ORDER)
            scaled_features = scaler.transform(input_features_df)
            predicted_price = model.predict(scaled_features)
            prediction_text = f"Prediksi Harga Rumah: Rp {predicted_price[0]:,.2f}"

            # --- Prepare calculation details ---
            intercept = model.intercept_
            coefficients = model.coef_
            
            calculation_details = {
                'intercept': round(intercept, 2),
                'features': [],
                'calculated_prediction': 0 # To sum up and verify
            }
            
            current_calculation_sum = intercept

            for i, feature_name in enumerate(EXPECTED_FEATURE_ORDER):
                raw_value = input_features_df.iloc[0, i]
                scaled_value = scaled_features[0, i]
                coefficient = coefficients[i]
                contribution = scaled_value * coefficient
                current_calculation_sum += contribution
                
                calculation_details['features'].append({
                    'name': feature_name,
                    'raw_value': raw_value,
                    'scaled_value': round(scaled_value, 4),
                    'coefficient': round(coefficient, 4),
                    'contribution': round(contribution, 2)
                })
            
            calculation_details['calculated_prediction'] = round(current_calculation_sum, 2)
            # Ensure the manually calculated prediction matches the model's output (for debugging)
            # print(f"Model prediction: {predicted_price[0]}, Manual calculation: {current_calculation_sum}")

        except ValueError:
            prediction_text = "Error: Please enter valid numbers for all features."
        except Exception as e:
            prediction_text = f"An error occurred during prediction: {e}"
            # Ensure calculation_details is reset or not shown if an error occurs before it's populated
            calculation_details = None


    return render_template('index.html', prediction_text=prediction_text, calculation_details=calculation_details)

if __name__ == '__main__':
    print(f"Flask app starting. Expecting model at: {model_path}")
    print(f"Expecting scaler at: {scaler_path}")
    app.run(debug=True)
