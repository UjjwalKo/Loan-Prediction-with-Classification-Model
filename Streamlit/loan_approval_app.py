import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import requests
import math

st.set_page_config(
    page_title="Loan Prediction App",
    page_icon="💰",
    layout="wide"
)

st.title("Loan Eligibility Prediction System")
st.markdown("Enter your details below to check if your loan application is likely to be approved.")

@st.cache_resource
def load_model():
    try:
        return joblib.load('loan_prediction_model.pkl')
    except:
        categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 
                           'Self_Employed', 'Property_Area']
        numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                         'Loan_Amount_Term', 'Credit_History']
        
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])
        
        model = LogisticRegression(random_state=0)
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        return pipeline

def get_explanation_from_ollama(features, prediction):
    try:
        formatted_features = "\n".join([f"- {key}: {value[0]}" for key, value in features.items()])
        
        if prediction == 1:
            prompt = f"""You are a banking expert assistant. A loan application has been APPROVED. 
            
Applicant details:
{formatted_features}

Explain in 3-4 sentences why a bank would likely approve this loan application. Focus on the key factors that led to approval."""
        else:
            prompt = f"""You are a banking expert assistant. A loan application has been DENIED. 
            
Applicant details:
{formatted_features}

Explain in 3-4 sentences why a bank would likely reject this loan application. Focus on the key factors that led to rejection."""
       
        ollama_endpoint = "http://localhost:11434/api/generate"
        
        payload = {
            "model": "deepseek-r1:1.5b",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 200
            }
        }
        
        response = requests.post(ollama_endpoint, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            explanation = result.get("response", "")
            return explanation.strip()
        else:
            return f"Error from Ollama API: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"Unable to generate explanation: {str(e)}"

def get_explanation(features, prediction):
    try:
        explanation = get_explanation_from_ollama(features, prediction)
        
        if explanation and not explanation.startswith("Error") and not explanation.startswith("Unable"):
            return explanation
            
        st.warning("Could not connect to Ollama. Using fallback explanations.")
        if prediction == 1:
            return "Based on the applicant's strong financial profile, steady income, and good credit history, the loan application was approved. The debt-to-income ratio appears favorable, suggesting the applicant has sufficient income to manage the loan repayments. Additionally, the loan amount requested is reasonable relative to the applicant's financial situation and the property value."
        else:
            return "The loan application was rejected primarily due to concerns about repayment capacity. The applicant's income may be insufficient relative to the requested loan amount, or their credit history shows some past payment issues. The debt-to-income ratio appears unfavorable, suggesting potential difficulty in managing additional debt. The bank may also have concerns about the property valuation or the loan term requested."
        
    except Exception as e:
        return f"Unable to generate explanation: {str(e)}"

model = load_model()

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["None","Male", "Female"], index=0)
    married = st.selectbox("Marital Status", [ "None", "Married", "Unmarried"], index=0)
    dependents = st.selectbox("Number of Dependents", ["None","0", "1", "2", "3+"], index=0)
    education = st.selectbox("Education", ["None", "Graduate", "Not Graduate"], index=0)
    self_employed = st.selectbox("Self Employed", [ "None", "Yes", "No"], index=0)
    
with col2:
    applicant_income = st.number_input("Applicant Income (monthly in $)", min_value=0, value=0)
    coapplicant_income = st.number_input("Coapplicant Income (monthly in $)", min_value=0, value=0)
    loan_amount = st.number_input("Loan Amount (in $1000s)", min_value=0, value=100)
    loan_term = st.selectbox("Loan Term (months)", options=[360, 300, 240, 180, 120, 84, 60, 36, 12, 0], index=0)
    credit_history = st.selectbox("Credit History", [ "None","1 (Good)", "0 (Bad)"], index=0)
    property_area = st.selectbox("Property Area", [ "None","Urban", "Semiurban", "Rural"], index=0)

if st.button("Check Loan Eligibility", type="primary"):
    input_data = {
        'Gender': [None if gender == "None" else gender],
        'Married': [None if married == "None" else married],
        'Dependents': [None if dependents == "None" else dependents],
        'Education': [None if education == "None" else education],
        'Self_Employed': [None if self_employed == "None" else self_employed],
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_term],
        'Credit_History': [None if credit_history == "None" else (1 if credit_history.startswith('1') else 0)],
        'Property_Area': [None if property_area == "None" else property_area],
    }
   
    input_data['Total_Applicant_Income'] = [input_data['ApplicantIncome'][0] + input_data['CoapplicantIncome'][0]]
   
    input_data['Total_Applicant_Income_log'] = [math.log(input_data['Total_Applicant_Income'][0] + 1) if input_data['Total_Applicant_Income'][0] > 0 else 0]
    input_data['LoanAmount_log'] = [math.log(input_data['LoanAmount'][0] + 1) if input_data['LoanAmount'][0] > 0 else 0]
    
    input_df = pd.DataFrame(input_data)
    
    with st.expander("Debug: Input Data"):
        st.write(input_df)
    
    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][prediction]
        
        if prediction == 1:
            st.success(f"🎉 Loan Application: APPROVED (Confidence: {probability:.2%})")
        else:
            st.error(f"❌ Loan Application: REJECTED (Confidence: {probability:.2%})")
        
        with st.expander("See detailed explanation", expanded=True):
            with st.spinner("Generating explanation with deepseek-r1:1.5b..."):
                explanation = get_explanation(input_data, prediction)
                st.write(explanation)
            
        if hasattr(model.named_steps['model'], 'coef_'):
            with st.expander("See factors that influenced this decision"):
                st.write("The model considers these factors when making predictions:")
                st.info("Note: This is a simplified view of feature importance and may not capture all nuances of the decision.")
                
                try:
                    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
                    coefficients = model.named_steps['model'].coef_[0]
                    
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': np.abs(coefficients)
                    }).sort_values('Importance', ascending=False).head(10)
                    
                    st.bar_chart(importance_df.set_index('Feature'))
                except Exception as e:
                    st.write(f"Feature importance information is not available: {str(e)}")
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.error("Please check if the model expects any additional features or has different feature names.")

    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            if any(model.get("name", "").startswith("deepseek-r1:1.5b") for model in models):
                st.success("✅ Connected to Ollama - deepseek-r1:1.5b is available")
            else:
                st.warning("Connected to Ollama but deepseek-r1:1.5b is not found. Pull it with: 'ollama pull deepseek-r1:1.5b'")
        else:
            st.error("Could not connect to Ollama API")
    except:
        st.error("Could not connect to Ollama API. Make sure Ollama is running.")