import pandas as pd
import matplotlib.pyplot as plt

# Load data
sales = pd.read_csv('data/sales_data.csv')

# Check missing values
print('Missing values:\n', sales.isnull().sum())

# Plot 1: Revenue over time
plt.figure(figsize=(10,5))
plt.plot(sales['Date'], sales['Revenue'], marker='o')
plt.title('Revenue Over Time')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.savefig('outputs/plot1.png')
plt.close()

# Plot 2: Units sold per product
plt.figure(figsize=(8,5))
sales['Product'].value_counts().plot(kind='bar')
plt.title('Units Sold by Product')
plt.xlabel('Product')
plt.ylabel('Units Sold')
plt.savefig('outputs/plot2.png')
plt.close()

# Plot 3: Revenue distribution
plt.figure(figsize=(8,5))
sales['Revenue'].plot(kind='pie', autopct='%1.1f%%')
plt.title('Revenue Distribution')
plt.savefig('outputs/plot3.png')
plt.close()