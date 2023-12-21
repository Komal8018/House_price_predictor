import pandas as pd
import joblib

# Load the model
model_file_path = r'C:\Users\komal khatri\Downloads\kc_house_data.csv  final\housing_price_model.pkl'

try:
    model = joblib.load(model_file_path)
except FileNotFoundError:
    print(f"Error: File '{model_file_path}' not found.")
except Exception as e:
    print(f"Error loading the model: {e}")
    exit()

# Sample new data with placeholder values for missing columns
new_data = {
    'bedrooms': 3,
    'bathrooms': 2,
    'sqft_living': 1800,
    'sqft_lot': 5000,
    'floors': 2,
    'waterfront': 0,
    'view': 1,
    'condition': 3,
    'grade': 8,
    'sqft_above': 1600,
    'sqft_basement': 200,
    'yr_built': 1990,
    'yr_renovated': 0,
    'zipcode': 98001,
    'lat': 47.123,
    'long': -122.345,
    'id': 0,  # Placeholder for missing 'id' column
    'sqft_living15': 0,  # Placeholder for missing 'sqft_living15' column
    'sqft_lot15': 0,  # Placeholder for missing 'sqft_lot15' column
}

# Convert the new data into a DataFrame
new_data_df = pd.DataFrame([new_data])

# Use the loaded model to make predictions
predictions = model.predict(new_data_df)

print("Predictions:", predictions)
