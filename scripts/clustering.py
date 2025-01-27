from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load and prepare data
transactions = pd.read_csv("data/Transactions.csv")
customers = pd.read_csv("data/Customers.csv")

customer_profiles = transactions.groupby("CustomerID").agg({"TotalValue": "sum", "Quantity": "sum"}).reset_index()

# Normalize features
scaler = StandardScaler()
features = scaler.fit_transform(customer_profiles[["TotalValue", "Quantity"]])

# Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
customer_profiles["Cluster"] = kmeans.fit_predict(features)

# Evaluate clustering
db_index = davies_bouldin_score(features, customer_profiles["Cluster"])
print("Davies-Bouldin Index:", db_index)

# Visualize clusters
sns.scatterplot(data=customer_profiles, x="TotalValue", y="Quantity", hue="Cluster", palette="viridis")
plt.title("Customer Clusters")
plt.show()
