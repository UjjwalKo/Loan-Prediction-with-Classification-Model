import streamlit as st
import matplotlib.pyplot as plt

st.title("Loan Eligibility Prediction App")
st.write("This app is developed by Ujjwal Koirala. Connect with me on [GitHub](https://github.com/UjjwalKo) and [LinkedIn](https://www.linkedin.com/in/ujjwal-koirala02/).")

name = st.text_input('Name')
gender = st.selectbox('Gender', ['','Male', 'Female'])
married = st.selectbox('Married', ['','Yes', 'No'])
dependents = st.selectbox('Dependents', ['0', '1', '2', '3+'])
education = st.selectbox('Education', ['','Graduate', 'Not Graduate'])
self_employed = st.selectbox('Self Employed', ['','Yes', 'No'])
applicant_income = st.number_input('Applicant Income', min_value=0)
coapplicant_income = st.number_input('Coapplicant Income', min_value=0)
loan_amount = st.number_input('Loan Amount', min_value=0)
loan_amount_term = st.number_input('Loan Amount Term (in months)', min_value=12)
credit_history = st.selectbox('Credit History', ['',1.0, 0.0])
property_area = st.selectbox('Property Area', ['','Urban', 'Semiurban', 'Rural'])


interest_rate = 0.10  
total_income = applicant_income + coapplicant_income
affordable_payment_threshold = 0.4 * total_income  

monthly_payment = (loan_amount + (loan_amount * interest_rate * (loan_amount_term / 12))) / loan_amount_term

if st.button('Predict'):
    if credit_history == 1.0: 
        if loan_amount_term >= 12 and loan_amount_term <= 360:  
            if loan_amount < 5000: 
                st.error(f'Sorry {name}, the loan amount of ${loan_amount:.2f} is too low for approval. Minimum amount is $5000.')
            elif loan_amount > 500000:  
                st.error(f'Sorry {name}, the loan amount of ${loan_amount:.2f} is too high for approval. Maximum amount is $500,000.')
            else:
                if monthly_payment <= affordable_payment_threshold: 
                    st.success(f'Congratulations {name}, your loan is likely to be approved!')
                else:
                    st.error(f'Sorry {name}, your monthly payment of ${monthly_payment:.2f} exceeds 40% of your total income, so your loan is likely to be rejected.')
        else:
            st.error(f'Sorry {name}, the loan term of {loan_amount_term} months is not within the acceptable range (12-360 months).')
    else:
        st.error(f'Sorry {name}, with a poor credit history, your loan is likely to be rejected.')

st.header("Income Visualization")
fig, ax = plt.subplots()
ax.bar(['Applicant Income', 'Coapplicant Income'], [applicant_income, coapplicant_income], color=['blue', 'green'])
ax.set_ylabel('Income')
ax.set_title('Income Distribution')
st.pyplot(fig)