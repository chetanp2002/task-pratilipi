import streamlit as st
import pickle
import numpy as np


st.set_page_config(page_title="Pratilipi Recommendation System", layout="wide")

# Load Pickled Objects Using Caching

@st.cache_resource(show_spinner=True)
def load_model():
    """Load the pre-trained LightFM model from disk."""
    with open("models/lightfm_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_resource(show_spinner=True)
def load_mapping():
    """Load the user and item mapping dictionaries from disk."""
    with open("models/lightfm_mapping.pkl", "rb") as f:
        mapping = pickle.load(f)
    return mapping  

@st.cache_resource(show_spinner=True)
def load_item_features():
    """Load the item features matrix from disk."""
    with open("models/lightfm_item_features.pkl", "rb") as f:
        item_features = pickle.load(f)
    return item_features

# Load the pickled objects
model = load_model()
mapping = load_mapping()
item_features_matrix = load_item_features()

# Process the Mapping
# Unpack mapping; if more than two elements, use only the first two.
if isinstance(mapping, tuple):
    if len(mapping) >= 2:
        user_mapping, item_mapping = mapping[:2]
    else:
        st.error("Mapping tuple does not contain enough elements.")
        user_mapping, item_mapping = {}, {}
else:
    st.error("Mapping is not a tuple; found type: " + str(type(mapping)))
    user_mapping, item_mapping = {}, {}

# To avoid data type mismatches, cast all user IDs to strings.
user_mapping = {str(k): v for k, v in user_mapping.items()}
item_mapping = {str(k): v for k, v in item_mapping.items()}

# Create a reverse mapping for items: internal -> external pratilipi ID
rev_item_mapping = {v: k for k, v in item_mapping.items()}

# Streamlit App Layout

st.title("Pratilipi Recommendation System")
st.markdown("""
Welcome to the Pratilipi Recommendation App!  
Enter your User ID below to receive personalized story recommendations.
""")

# Sidebar input for User ID
st.sidebar.header("User Input")
user_input = st.sidebar.text_input("Enter your User ID", value="1")

st.sidebar.markdown("""
## Example User IDs

Here are some sample User IDs you can try:

- **5506791954036110**
- **5506791961431145**
- **5506791988747277**
- **5506791966136056**
- **5506791974854999**
""")


# Recommendation Function

def recommend(user_ext_id, model, user_mapping, item_mapping, item_features_matrix, num_rec=5):
    """
    Given an external user ID, return the top recommended pratilipi IDs.
    """
    # Ensure the user ID is a string
    user_ext_id = str(user_ext_id).strip()
    
    # Check if the user exists in the mapping
    if user_ext_id not in user_mapping:
        return None
    
    # Convert external user ID to internal index
    internal_user_id = user_mapping[user_ext_id]
    n_items = len(item_mapping)
    
    # Predict scores for all items for this user using LightFM's predict method
    scores = model.predict(internal_user_id, np.arange(n_items), item_features=item_features_matrix)
    
    # Get the indices of the top scoring items (highest first)
    top_indices = np.argsort(-scores)[:num_rec]
    
    # Convert internal item indices back to external pratilipi IDs
    recommended_items = [rev_item_mapping[i] for i in top_indices]
    return recommended_items

# Display Recommendations Based on User Input

if st.sidebar.button("Get Recommendations"):
    user_ext_id = user_input.strip()
    recs = recommend(user_ext_id, model, user_mapping, item_mapping, item_features_matrix, num_rec=5)
    
    if recs is None or len(recs) == 0:
        st.warning("User not found or no recommendations available.")
        # Optionally, display a sample of available User IDs to help debugging:
        st.info("Example User IDs: " + ", ".join(list(user_mapping.keys())[:10]))
    else:
        st.success(f"Recommendations for User ID {user_ext_id}:")
        for rec in recs:
            st.write(f"**Pratilipi ID: {rec}**")
