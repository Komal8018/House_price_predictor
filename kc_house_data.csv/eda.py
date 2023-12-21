import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = r'C:\Users\komal khatri\Downloads\kc_house_data.csv  final\kc_house_data.csv\kc_house_data.csv'

# Read the CSV file
data = pd.read_csv(file_path)

# Visualize the relationship between number of bedrooms, floors, view, and waterfront with price

# Number of Bedrooms vs Price
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.scatter(data['bedrooms'], data['price'], color='green')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Price')
plt.title('Number of Bedrooms vs Price')

# Number of Floors vs Price
plt.subplot(2, 2, 2)
plt.scatter(data['floors'], data['price'], color='orange')
plt.xlabel('Number of Floors')
plt.ylabel('Price')
plt.title('Number of Floors vs Price')

# View vs Price
plt.subplot(2, 2, 3)
plt.scatter(data['view'], data['price'], color='magenta')
plt.xlabel('View')
plt.ylabel('Price')
plt.title('View vs Price')

# Waterfront vs Price
plt.subplot(2, 2, 4)
plt.scatter(data['waterfront'], data['price'], color='cyan')
plt.xlabel('Waterfront (1: Yes, 0: No)')
plt.ylabel('Price')
plt.title('Waterfront vs Price')

plt.tight_layout()
plt.show()

# Visualize the relationship between view and price
X_view = data['view']
y_view = data['price']

plt.scatter(X_view, y_view, color='magenta')
plt.xlabel('View')
plt.ylabel('Price')
plt.title('View vs Price')
plt.show()

# Visualize the relationship between waterfront and price
X_waterfront = data['waterfront']
y_waterfront = data['price']

plt.scatter(X_waterfront, y_waterfront, color='cyan')
plt.xlabel('Waterfront (1: Yes, 0: No)')
plt.ylabel('Price')
plt.title('Waterfront vs Price')
plt.show()
