import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = r'C:\Users\komal khatri\Downloads\kc_house_data.csv  final\kc_house_data.csv\kc_house_data.csv'

# Read the CSV file
data = pd.read_csv(file_path)


# Update column names based on assumptions
numerical_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'sqft_living15', 'sqft_lot15']
categorical_features = ['id', 'date', 'zipcode']

# Separate features and target variable
X = data[numerical_features + categorical_features]
y = data['price']

# Visualize the relationship between square footage and price
plt.scatter(X['sqft_living'], y, color='blue')
plt.xlabel('Square Footage')
plt.ylabel('Price')
plt.title('Square Footage vs Price')
plt.show()

# Assuming X_processed contains the preprocessed features
plt.scatter(X['bedrooms'], y, color='green')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Price')
plt.title('Number of Bedrooms vs Price')
plt.show()
# Visualize the relationship between number of bedrooms and price
# Modify this as per your preprocessed data
# plt.scatter(X_processed[:, 1], y, color='green')

# You can create similar visualizations for other features as well
