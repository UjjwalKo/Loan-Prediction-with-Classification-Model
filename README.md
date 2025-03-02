# Loan Approval Prediction Project

## Project Overview
This project is focused on analyzing and predicting loan approval outcomes based on various applicant attributes. The dataset consists of 614 loan applications, with 13 attributes describing each application. The goal is to build machine learning models to predict the `Loan_Status` (approved or not approved) of loan applications based on the provided attributes.

### Dataset Details
The dataset contains the following columns:

| Column               | Description                                                               |
|----------------------|---------------------------------------------------------------------------|
| Gender               | Gender of the applicant (Male/Female)                                    |
| Married              | Marital status of the applicant (Yes/No)                                 |
| Dependents           | Number of dependents                                                     |
| Education            | Education level of the applicant (Graduate/Not Graduate)                |
| Self_Employed        | Employment status of the applicant (Yes/No)                              |
| ApplicantIncome      | Income of the applicant                                                  |
| CoapplicantIncome    | Income of the co-applicant                                               |
| LoanAmount           | Loan amount requested                                                   |
| Loan_Amount_Term     | Term of the loan in months                                               |
| Credit_History       | Credit history (1.0 indicates good credit history, 0.0 indicates poor)   |
| Property_Area        | Area of the property (Urban/Rural/Semiurban)                             |
| Loan_Status          | Target variable indicating loan approval (Y for approved, N for not)    |

### Dataset Summary
- **Number of Records**: 614
- **Number of Attributes**: 13 (including the target variable)
- **Target Variable**: Loan_Status

### Missing Values
Several columns in the dataset contain missing values. For instance:
- `Gender`: 13 missing values
- `Married`: 3 missing values
- `Dependents`: 15 missing values
- `Self_Employed`: 32 missing values
- `LoanAmount`: 22 missing values
- `Loan_Amount_Term`: 14 missing values
- `Credit_History`: 50 missing values

## Data Preprocessing

1. **Handling Missing Values**:
   - Categorical columns (`Gender`, `Married`, `Self_Employed`, `Dependents`, `Credit_History`) were filled with their mode.
   - Numerical columns (`LoanAmount`, `Loan_Amount_Term`) were filled with their mean using the `SimpleImputer` class from Scikit-learn.

2. **Feature Engineering**:
   - Normalized `LoanAmount` and `Total_Applicant_Income` (sum of `ApplicantIncome` and `CoapplicantIncome`) using logarithmic transformation to address skewness.

3. **One-Hot Encoding**:
   - Applied to categorical variables for compatibility with machine learning models.
   - Target variable `Loan_Status` was mapped to binary values (1 for approved, 0 for not approved).

## Exploratory Data Analysis

1. **Visualizations**:
   - Distribution of missing values visualized using heatmaps.
   - Histograms were used to examine skewness in `LoanAmount` and `Total_Applicant_Income` before and after normalization.
   - Violin plots and count plots were used to analyze relationships between variables such as `ApplicantIncome` and `Education`, and `Loan_Status` and `Credit_History`.

2. **Key Observations**:
   - Higher income levels do not always guarantee loan approval.
   - Good credit history strongly correlates with loan approval.

## Machine Learning Models
Four machine learning models were implemented and evaluated for this classification task:

### Logistic Regression
- **Pipeline**: Preprocessing (scaling, encoding) + Logistic Regression.
- **Accuracy**: 83.74%

### XGBoost Classifier
- **Parameters**: 1200 estimators, learning rate of 0.05.
- **Accuracy**: 82.11%

### Decision Tree Classifier
- **Criterion**: Entropy
- **Accuracy**: 71.54%

### Random Forest Classifier
- **Parameters**: 1000 estimators, Gini criterion.
- **Accuracy**: 78.05%

## Conclusion
Among the models tested, Logistic Regression provided the highest accuracy, followed by XGBoost. This suggests that linear relationships are sufficient to model the data effectively. Further optimization and experimentation with hyperparameters could potentially improve the model performance.
