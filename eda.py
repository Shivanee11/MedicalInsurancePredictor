import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('insurance.csv')

# Display the first few rows
print("First 5 rows of the dataset:")
print(data.head())

# Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Basic statistics summary
print("\nBasic statistics of the dataset:")
print(data.describe())

# Select only numeric columns for correlation
numeric_data = data.select_dtypes(include=['float64', 'int64'])

# Correlation matrix
correlation_matrix = numeric_data.corr()

# Display correlation matrix as text output
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Visualizing the correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=1)
plt.title("Correlation Heatmap")
plt.show()  # This will display the heatmap plot

# Visualizing the distribution of insurance charges
plt.figure(figsize=(8, 6))
sns.histplot(data['charges'], bins=30, kde=True, color='blue')
plt.title("Distribution of Insurance Charges")
plt.xlabel("Charges")
plt.ylabel("Frequency")
plt.show()  # This will display the distribution plot

# Visualizing the relationship between age and charges
plt.figure(figsize=(8, 6))
sns.scatterplot(x='age', y='charges', data=data)
plt.title("Age vs Charges")
plt.xlabel("Age")
plt.ylabel("Charges")
plt.show()  # This will display the scatter plot

