import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
data = pd.DataFrame({
    'square_footage': [1500, 2000, 1800, 1900, 2100],
    'bedrooms': [3, 4, 3, 2, 4],
    'bathrooms': [2, 2.5, 2, 1.5, 3],
    'location': ['A', 'B', 'A', 'C', 'B'],
    'price': [300000, 400000, 350000, 250000, 450000]
})

numerical_features = ['square_footage', 'bedrooms', 'bathrooms']
categorical_features = ['location']

X = data[numerical_features + categorical_features]
y = data['price']

numerical_imputer = SimpleImputer(strategy='mean')
X.loc[:, numerical_features] = numerical_imputer.fit_transform(X.loc[:, numerical_features])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

X = preprocessor.fit_transform(X)

print(X[:5]) 