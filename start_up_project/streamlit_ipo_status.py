import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('random_forest_model.pkl')

# Mapping for IPO status prediction
ipo_status_mapping = {0: 'pre IPo', 1: 'Closed to going IPO', 2: 'post IPO'}

# Streamlit interface
st.title("IPO Status Prediction")

# User inputs
total_funds = st.number_input("Total Funds (USD)", min_value=0, value=1000000)
venture = st.selectbox("Venture (Yes/No)", options=['Yes', 'No'])
private_equity = st.selectbox("Private Equity (Yes/No)", options=['Yes', 'No'])
round_selected = st.selectbox("Select Funding Round", options=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])

# Encode categorical inputs manually (1 for Yes, 0 for No)
venture_encoded = 1 if venture == 'Yes' else 0
private_equity_encoded = 1 if private_equity == 'Yes' else 0

# Manually map the funding round (A = 1, B = 2, ..., H = 8)
round_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8}
round_encoded = round_mapping[round_selected]

# Default values for the other columns
round_a = 0
round_b = 0
round_c = 0
round_d = 0
round_e = 0
round_f = 0
round_g = 0
round_h = 0
post_ipo_equity = 0
post_ipo_debt = 0

# Set the selected funding round to 1
if round_selected == 'A':
    round_a = 1
elif round_selected == 'B':
    round_b = 1
elif round_selected == 'C':
    round_c = 1
elif round_selected == 'D':
    round_d = 1
elif round_selected == 'E':
    round_e = 1
elif round_selected == 'F':
    round_f = 1
elif round_selected == 'G':
    round_g = 1
elif round_selected == 'H':
    round_h = 1

# Create input array with all 15 features
input_data = np.array([[total_funds, 
                        0, # market (default or dummy value)
                        0, # funding_rounds (default or dummy value)
                        round_a, round_b, round_c, round_d, round_e, 
                        round_f, round_g, round_h, 
                        venture_encoded, private_equity_encoded, 
                        post_ipo_equity, post_ipo_debt]])

# Predict IPO status
if st.button('Predict'):
    prediction = model.predict(input_data)
    
    # Decode the predicted IPO status to a human-readable label
    predicted_status = ipo_status_mapping[prediction[0]]
    
    # Display the prediction
    st.write(f"The predicted IPO status is: {predicted_status}")
