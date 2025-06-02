from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
import os # Ensures os module is imported

# Initialize the Flask application
app = Flask(__name__)

# --- Get the directory where this script (app.py) is located ---
# This makes the paths to model.pkl and scaler.pkl relative to app.py's location
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
    # Optional: print a success message to the console during development
    print(f"SUCCESS: Loaded model from {model_path}")
    print(f"SUCCESS: Loaded scaler from {scaler_path}")

except FileNotFoundError:
    # Updated error message to show the paths that were checked
    print(f"Error: model.pkl or scaler.pkl not found.")
    print(f"Attempted to load model from: {model_path}")
    print(f"Attempted to load scaler from: {scaler_path}")
    print(f"Please ensure 'model.pkl' and 'scaler.pkl' are in the same directory as 'app.py': {script_dir}")
    model = None
    scaler = None
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

# Define the order of features your model expects
# This MUST match the order used during training in your Colab notebook
# Based on your script, it's likely: ['LT', 'LB', 'JKT', 'JKM', 'GRS']
EXPECTED_FEATURE_ORDER = ['LT', 'LB', 'JKT', 'JKM', 'GRS']

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = None
    if request.method == 'POST':
        if model is None or scaler is None:
            # This message is shown on the webpage
            prediction_text = "Error: Model or scaler not loaded. Please check server logs for details."
            return render_template('index.html', prediction_text=prediction_text)
        try:
            # Get data from the form
            lt = float(request.form['LT'])
            lb = float(request.form['LB'])
            jkt = int(request.form['JKT'])
            jkm = int(request.form['JKM'])
            grs = int(request.form['GRS']) # GRS is '1' for Ada, '0' for Tidak Ada

            # Create a DataFrame for the input features IN THE CORRECT ORDER
            input_features_df = pd.DataFrame([[lt, lb, jkt, jkm, grs]], columns=EXPECTED_FEATURE_ORDER)

            # Scale the input features
            # The scaler expects a 2D array, which a DataFrame provides
            scaled_features = scaler.transform(input_features_df)

            # Make a prediction
            # The model also expects a 2D array
            predicted_price = model.predict(scaled_features)

            # Format the prediction for display
            # The output 'predicted_price[0]' is a NumPy array with one element
            prediction_text = f"Prediksi Harga Rumah: Rp {predicted_price[0]:,.2f}"

        except ValueError:
            prediction_text = "Error: Please enter valid numbers for all features."
        except Exception as e:
            prediction_text = f"An error occurred during prediction: {e}"

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    # Optional: Print a startup message indicating paths being used (can be removed for production)
    print(f"Flask app starting. Expecting model at: {model_path}")
    print(f"Expecting scaler at: {scaler_path}")
    app.run(debug=True) # debug=True is for development, set to False for production