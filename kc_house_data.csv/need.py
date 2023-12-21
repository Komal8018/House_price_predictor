import os
import glob

# Print the current working directory
print("Current Working Directory:", os.getcwd())

# Search for the file in the current working directory and its subdirectories
model_files = glob.glob(os.path.join(os.getcwd(), '**/housing_price_model.pkl'), recursive=True)

if model_files:
    model_file_path = model_files[0]
    print("Found the model file at:", model_file_path)
else:
    print("Error: Model file not found.")
import pickle

# Load the model
model_file_path = r'C:\Users\komal khatri\Downloads\kc_house_data.csv  final\housing_price_model.pkl'

try:
    with open(model_file_path, 'rb') as model_file:
        # Your code to load the model
        # Example: model = pickle.load(model_file)
        pass  # Replace this with your actual code to load the model
except FileNotFoundError:
    print(f"Error: File '{model_file_path}' not found.")
except Exception as e:
    print(f"Error loading the model: {e}")
import pickle

# Load the model
model_file_path = r'C:\Users\komal khatri\Downloads\kc_house_data.csv  final\housing_price_model.pkl'

try:
    with open(model_file_path, 'rb') as model_file:
        model = pickle.load(model_file)
        # Now 'model' contains your trained machine learning model
except FileNotFoundError:
    print(f"Error: File '{model_file_path}' not found.")
except Exception as e:
    print(f"Error loading the model: {e}")
import joblib
import numpy as np

# Load the model
model_file_path = r'C:\Users\komal khatri\Downloads\kc_house_data.csv  final\housing_price_model.pkl'

try:
    model = joblib.load(model_file_path)
except FileNotFoundError:
    print(f"Error: File '{model_file_path}' not found.")
except Exception as e:
    print(f"Error loading the model: {e}")
    exit()

# Sample new data
new_data = np.array([[3, 2, 1800, 5000, 2, 0, 1, 3, 8, 1600, 200, 1990, 0, 98001, 47.123, -122.345]])

# Use the loaded model to make predictions
predictions = model.predict(new_data)

print("Predictions:", predictions)
import pandas as pd

# Assuming 'your_dataset.csv' is your dataset file
file_path = r'C:\Users\komal khatri\Downloads\kc_house_data.csv  final\kc_house_data.csv\kc_house_data.csv'

# Read the CSV file
data = pd.read_csv(file_path)


# Replace 'column_name' with the actual name of the column you want to display
single_column = data['price']

print(single_column)
