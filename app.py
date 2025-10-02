import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Cache the model and encoders to avoid reloading on every interaction
@st.cache_resource
def load_model_and_encoders():
    """Load the model and all preprocessing objects"""
    try:
        model = tf.keras.models.load_model('model.h5')
        
        with open('onehot_encoder.pkl', 'rb') as file:
            onehot_encoder_geo = pickle.load(file)
        
        with open('label_encoder.pkl', 'rb') as file:
            label_encoder_gender = pickle.load(file)
        
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        
        return model, onehot_encoder_geo, label_encoder_gender, scaler
    except FileNotFoundError as e:
        st.error(f"Error loading files: {e}")
        st.stop()

# Load everything
model, onehot_encoder_geo, label_encoder_gender, scaler = load_model_and_encoders()

# Title and description
st.title("🏦 Customer Churn Prediction")
st.markdown("""
This app predicts the likelihood of a customer churning based on their profile.
Fill in the customer details below to get a prediction.
""")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Personal Information")
    geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('Gender', label_encoder_gender.classes_)
    age = st.slider('Age', 18, 92, 35)
    
    st.subheader("💰 Financial Information")
    credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=650, step=1)
    balance = st.number_input('Balance', min_value=0.0, value=50000.0, step=1000.0, format="%.2f")
    estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=50000.0, step=1000.0, format="%.2f")

with col2:
    st.subheader("🏦 Account Information")
    tenure = st.slider('Tenure (years)', 0, 10, 5)
    num_of_products = st.slider('Number of Products', 1, 4, 1)
    has_cr_card = st.selectbox('Has Credit Card', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    is_active_member = st.selectbox('Is Active Member', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

# Add a predict button
if st.button('🔮 Predict Churn', type="primary", use_container_width=True):
    try:
        # Prepare the input data
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Gender': [label_encoder_gender.transform([gender])[0]],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_member],
            'EstimatedSalary': [estimated_salary]
        })
        
        # One Hot Encode Geography
        geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
        geo_encoded_df = pd.DataFrame(
            geo_encoded, 
            columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
        )
        
        # Combine one hot data into input data
        input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
        
        # Scale the data
        input_scaled_data = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled_data, verbose=0)
        prediction_prob = prediction[0][0]
        
        # Display results with better formatting
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        # Create metrics display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Churn Probability", f"{prediction_prob:.2%}")
        
        with col2:
            churn_status = "High Risk" if prediction_prob > 0.5 else "Low Risk"
            st.metric("Risk Level", churn_status)
        
        with col3:
            confidence = max(prediction_prob, 1 - prediction_prob)
            st.metric("Confidence", f"{confidence:.2%}")
        
        # Visual indicator
        if prediction_prob > 0.5:
            st.error(f"⚠️ **High Churn Risk**: This customer is likely to churn (probability: {prediction_prob:.2%})")
            st.markdown("**Recommended Actions:**")
            st.markdown("- Reach out with retention offers")
            st.markdown("- Investigate customer satisfaction")
            st.markdown("- Offer personalized incentives")
        else:
            st.success(f"✅ **Low Churn Risk**: This customer is likely to stay (probability: {(1-prediction_prob):.2%})")
            st.markdown("**Recommended Actions:**")
            st.markdown("- Continue providing excellent service")
            st.markdown("- Consider upselling opportunities")
            st.markdown("- Maintain regular engagement")
        
        # Progress bar visualization
        st.markdown("### Churn Risk Meter")
        st.progress(float(prediction_prob))
        
        # Show input summary
        with st.expander("📝 View Input Summary"):
            st.dataframe(input_data, use_container_width=True)
    
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        st.exception(e)

# Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p style='color: gray;'>Customer Churn Prediction Model | Built with Streamlit & TensorFlow</p>
</div>
""", unsafe_allow_html=True)