import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
customers = pd.read_csv("data/Customers.csv")
products = pd.read_csv("data/Products.csv")
transactions = pd.read_csv("data/Transactions.csv")

# Data overview
print(customers.head())
print(products.head())
print(transactions.head())

# Merge transactions with customer and product data
data = transactions.merge(customers, on="CustomerID").merge(products, on="ProductID")

# Example EDA: Sales by region
sales_by_region = data.groupby("Region")["TotalValue"].sum().sort_values(ascending=False)
sales_by_region.plot(kind="bar", title="Sales by Region")
plt.show()
