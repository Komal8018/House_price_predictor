import pandas as pd

# Load the CSV file and display the column names
file_path = r'C:\Users\komal khatri\Downloads\kc_house_data.csv  final\kc_house_data.csv\kc_house_data.csv'

# Read the CSV file
data = pd.read_csv(file_path)
print("Column names in the CSV file:")
print(data.columns)

# Modify the below lists based on the output above
# Separate features and target variable
numerical_features = ['sqft_living', 'bedrooms', 'bathrooms']  # Update with the appropriate numerical column names
categorical_features = []  # Update with the appropriate categorical column names

# Separate features and target variable
X = data[numerical_features + categorical_features]
y = data['price']


