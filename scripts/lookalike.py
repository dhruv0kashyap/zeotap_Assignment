from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load and prepare data
transactions = pd.read_csv("data/Transactions.csv")
customers = pd.read_csv("data/Customers.csv")

# Aggregate transaction data to build customer profiles
customer_profiles = transactions.groupby("CustomerID").agg({"TotalValue": "sum", "Quantity": "sum"}).reset_index()

# Normalize features
scaler = StandardScaler()
features = scaler.fit_transform(customer_profiles[["TotalValue", "Quantity"]])

# Compute cosine similarity
similarity = cosine_similarity(features)

# Generate lookalike recommendations for the first 20 customers
recommendations = {}
for i, cust_id in enumerate(customer_profiles["CustomerID"][:20]):
    similar = sorted(list(enumerate(similarity[i])), key=lambda x: x[1], reverse=True)[1:4]
    recommendations[cust_id] = [(customer_profiles["CustomerID"].iloc[j], round(score, 2)) for j, score in similar]

# Save recommendations to Lookalike.csv in the required format
recommendations_df = pd.DataFrame({
    "CustomerID": list(recommendations.keys()),
    "Recommendations": [
        str([(cust, float(score)) for cust, score in recs]) for recs in recommendations.values()
    ]
})
recommendations_df.to_csv("Lookalike.csv", index=False)

print("Lookalike.csv has been generated successfully!")
