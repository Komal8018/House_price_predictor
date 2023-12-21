import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = r'C:\Users\komal khatri\Downloads\kc_house_data.csv  final\kc_house_data.csv\kc_house_data.csv'
data = pd.read_csv(file_path)

sns.set_palette("husl")

numerical_columns = data.select_dtypes(include='number').columns
num_cols = len(numerical_columns)
num_subplot_rows = (num_cols - 1) // 3 + 1

plt.figure(figsize=(16, 4 * num_subplot_rows))
for i, column in enumerate(numerical_columns, 1):
    plt.subplot(num_subplot_rows, 3, i)
    sns.histplot(data[column], bins=20, kde=True)
    plt.title(column, fontsize=14)
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(rotation=45, ha='right') 
plt.tight_layout(pad=3)  
plt.suptitle('Histograms of Numerical Attributes', y=1.02, fontsize=16)
plt.show()
