import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('svm_pca_model_grid_search.pkl')

# Define the Streamlit app
st.title("Startup Success Prediction")

# Create input fields for user input
funding_total_usd = st.number_input("Funding Total (in USD)", min_value=0.0, value=1000000.0, step=10000.0)
funding_rounds = st.number_input("Number of Funding Rounds", min_value=0, value=3, step=1)
seed = st.selectbox("Has Seed Funding?", options=[0, 1])  # 0 for No, 1 for Yes

# Initialize remaining features with default values
default_values = {
    'venture': 0,
    'equity_crowdfunding': 0,
    'undisclosed': 0,
    'convertible_note': 0,
    'debt_financing': 0,
    'angel': 0,
    'grant': 0,
    'private_equity': 0,
    'post_ipo_equity': 0,
    'post_ipo_debt': 0,
    'secondary_market': 0,
    'product_crowdfunding': 0,
    'round_a': 0,
    'round_b': 0,
    'round_c': 0,
    'round_d': 0,
    'round_e': 0,
    'round_f': 0,
    'round_g': 0,
    'round_h': 0
}

# Combine user input and default values into a single array
user_input = np.array([
    funding_total_usd,
    seed,
    funding_rounds,
    default_values['venture'],
    default_values['equity_crowdfunding'],
    default_values['undisclosed'],
    default_values['convertible_note'],
    default_values['debt_financing'],
    default_values['angel'],
    default_values['grant'],
    default_values['private_equity'],
    default_values['post_ipo_equity'],
    default_values['post_ipo_debt'],
    default_values['secondary_market'],
    default_values['product_crowdfunding'],
    default_values['round_a'],
    default_values['round_b'],
    default_values['round_c'],
    default_values['round_d'],
    default_values['round_e'],
    default_values['round_f'],
    default_values['round_g'],
    default_values['round_h']
]).reshape(1, -1)  # Reshape to 2D array as required by scikit-learn

# Create a button to trigger prediction
if st.button("Predict"):
    # Make a prediction
    prediction = model.predict(user_input)
    probabilities = model.predict_proba(user_input)  # Get the probabilities for each class

    # Display the predicted label
    predicted_label = prediction[0]

    # Get the probability for each class ('acquired', 'closed', 'operating')
    class_probs = {label: prob for label, prob in zip(model.classes_, probabilities[0])}

    # Display results
    st.success(f"The predicted status of the startup is: {predicted_label}")
    st.write("Prediction probabilities:")
    st.write(class_probs)
