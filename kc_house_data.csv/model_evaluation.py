import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

file_path = r'C:\Users\komal khatri\Downloads\kc_house_data.csv  final\kc_house_data.csv\kc_house_data.csv'

data = pd.read_csv(file_path)

numerical_features = ['sqft_living', 'bedrooms', 'bathrooms']
categorical_features = ['zipcode']
target = 'price'

X = data[numerical_features + categorical_features]
y = data[target]

numerical_imputer = SimpleImputer(strategy='mean')
X.loc[:, numerical_features] = numerical_imputer.fit_transform(X.loc[:, numerical_features])

# Using handle_unknown='ignore' in OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

X_processed = preprocessor.fit_transform(X)

print(X_processed[:5])

X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Print metrics
print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')

# Assuming you have actual prices for the new data
# Assuming you have actual prices for the new data
actual_prices = [
    221900.0,
    538000.0,
    180000.0,
    604000.0,
    510000.0
    # ... add more values as needed
]

# Define new_data with actual input data
new_data = pd.DataFrame({
    'sqft_living': [actual_prices[0]],
    'bedrooms': [actual_prices[1]],
    'bathrooms': [actual_prices[2]],
    'zipcode': [actual_prices[3]]
})

# Duplicate rows to match the length of actual_prices
new_data = pd.concat([new_data] * len(actual_prices), ignore_index=True)

# Ensure the length of actual_prices matches the number of rows in new_data
assert len(actual_prices) == len(new_data)

# Continue with the rest of the code...


# Predict on the new data
new_data_processed = preprocessor.transform(new_data)

new_predictions = model.predict(new_data_processed)

# Calculate metrics for new data
new_mae = mean_absolute_error(actual_prices, new_predictions)
new_mse = mean_squared_error(actual_prices, new_predictions)
new_rmse = np.sqrt(new_mse)


# Print metrics for new data
print(f'New Data - Mean Absolute Error: {new_mae}')
print(f'New Data - Mean Squared Error: {new_mse}')
print(f'New Data - Root Mean Squared Error: {new_rmse}')
