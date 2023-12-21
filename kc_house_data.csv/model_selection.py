import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

file_path = r'C:\Users\komal khatri\Downloads\kc_house_data.csv  final\kc_house_data.csv\kc_house_data.csv'

data = pd.read_csv(file_path)

numerical_features = ['sqft_living', 'bedrooms', 'bathrooms']
categorical_features = ['zipcode']  
target = 'price'

X = data[numerical_features + categorical_features]
y = data[target]

numerical_imputer = SimpleImputer(strategy='mean')
X.loc[:, numerical_features] = numerical_imputer.fit_transform(X.loc[:, numerical_features])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

X_processed = preprocessor.fit_transform(X)

print(X_processed[:5])  
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Assuming you have actual prices for the new data
actual_price_for_new_data = [
    221900.0,
    538000.0,
    180000.0,
    604000.0,
    510000.0,
    # ... add more values as needed
    325000.0
]
actual_prices = [actual_price_for_new_data]
# Assuming you have a line like this earlier in your script
predictions = model.predict(input_data)  # Replace 'input_data' with your actual input data

# Now, you can use predictions in the rest of your script
mae = mean_absolute_error(actual_prices, predictions)

# Calculate metrics
mae = mean_absolute_error(actual_prices, predictions)
mse = mean_squared_error(actual_prices, predictions)
rmse = np.sqrt(mse)

# Print metrics
print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
