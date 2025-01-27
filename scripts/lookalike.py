from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load and prepare data
transactions = pd.read_csv("data/Transactions.csv")
customers = pd.read_csv("data/Customers.csv")

customer_profiles = transactions.groupby("CustomerID").agg({"TotalValue": "sum", "Quantity": "sum"})
customer_profiles = customer_profiles.reset_index()

# Normalize features
scaler = StandardScaler()
features = scaler.fit_transform(customer_profiles[["TotalValue", "Quantity"]])

# Compute similarity
similarity = cosine_similarity(features)

# Generate lookalike recommendations
recommendations = {}
for i, cust_id in enumerate(customer_profiles["CustomerID"]):
    similar = sorted(list(enumerate(similarity[i])), key=lambda x: x[1], reverse=True)[1:4]
    recommendations[cust_id] = [(customer_profiles["CustomerID"].iloc[j], round(score, 2)) for j, score in similar]

# Save to CSV
recommendations_df = pd.DataFrame([(key, value) for key, value in recommendations.items()], columns=["CustomerID", "Recommendations"])
recommendations_df.to_csv("Lookalike.csv", index=False)
