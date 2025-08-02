import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import json
import numpy as np

# --- Load and preprocess data ---
df = pd.read_csv('online_retail.csv')
df = df[['CustomerID', 'Description', 'Quantity']]
df.dropna(inplace=True)
df['Description'] = df['Description'].str.strip()  # Strip spaces for accurate matching

# --- Create User-Item Matrix ---
user_item_matrix = df.pivot_table(index='CustomerID',
                                  columns='Description',
                                  values='Quantity',
                                  aggfunc='sum')
user_item_matrix.fillna(0, inplace=True)

# --- Compute Item Similarity ---
item_item_matrix = user_item_matrix.T
item_similarity = cosine_similarity(item_item_matrix)
item_similarity_df = pd.DataFrame(item_similarity,
                                  index=item_item_matrix.index,
                                  columns=item_item_matrix.index)

# --- Recommendation Logic ---
def get_similar_products(product_name, n=5):
    if product_name not in item_similarity_df.columns:
        return ["Product not found."]
    similar_scores = item_similarity_df[product_name]
    similar_scores = similar_scores.sort_values(ascending=False)[1:n+1]
    return similar_scores.index.tolist()

# --- Streamlit App ---
st.set_page_config(page_title="E-commerce Insights", layout="centered")

# --- App Styling ---
st.markdown("""
    <style>
    body {
        background-color: white;
        color: black;
    }
    .product-card {
        background-color: #f9f9f9;
        padding: 15px;
        margin: 10px 0;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: black;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Menu ---
st.sidebar.title("🧠 Modules")
module = st.sidebar.radio("Select Module", ["Home", "Clustering", "Recommendation"])

# --- Main App Interface ---
if module == "Home":
    st.title("📊 E-commerce Analytics Dashboard")
    st.write("Welcome to your personalized analytics tool. Choose a module from the left to begin.")

elif module == "Clustering":
    st.title("🔬 Customer Segmentation")
    try:
        # Load model and mapping
        kmeans = joblib.load("kmeans_model.pkl")
        scaler = joblib.load("scaler.pkl")
        with open("cluster_mapping.json", "r") as f:
            cluster_map = json.load(f)

        # Input fields
        recency = st.number_input("🕒 Recency (in days)", min_value=0)
        frequency = st.number_input("🔁 Frequency (number of purchases)", min_value=0)
        monetary = st.number_input("💰 Monetary (total spend)", min_value=0.0, format="%.2f")

        if st.button("Predict Cluster"):
            input_data = np.array([[recency, frequency, monetary]])
            scaled_input = scaler.transform(input_data)
            cluster = kmeans.predict(scaled_input)[0]
            segment = cluster_map.get(str(cluster), "Unknown Segment")

            st.success(f"🧍 This customer belongs to: **{segment}**")

    except FileNotFoundError:
        st.error("Required model or mapping file not found.")
    except Exception as e:
        st.error("An error occurred.")
        st.exception(e)

elif module == "Recommendation":
    st.title("🛍️ Product Recommendation System")
    product_input = st.text_input("Enter a Product Name", "")

    if st.button("Get Recommendations"):
        if product_input.strip() == "":
            st.warning("Please enter a product name.")
        else:
            recommendations = get_similar_products(product_input.strip())
            if recommendations[0] == "Product not found.":
                st.error("Product not found in database. Please check the name.")
            else:
                st.subheader("🔍 Recommended Products:")
                for product in recommendations:
                    st.markdown(f"<div class='product-card'>{product}</div>", unsafe_allow_html=True)

