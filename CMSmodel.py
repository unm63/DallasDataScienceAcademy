import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# Load your data (replace 'your_data.csv' with your actual data file)
data = pd.read_csv("C:/Users/ujwal/PycharmProjects/pythonProject/dataRemoved.csv")

# Display the first few rows of the dataframe
print(data.head())


# Define features and target
data = data.drop(columns = ['Unnamed: 0'])
print(data.info())
X = data.drop(columns=['CLM_PMT_AMT'])
y = data['CLM_PMT_AMT']

num_cols = ['NCH_PRMRY_PYR_CLM_PD_AMT', 'CLM_PASS_THRU_PER_DIEM_AMT', 'NCH_BENE_IP_DDCTBL_AMT',
 'NCH_BENE_PTA_COINSRNC_LBLTY_AM', 'NCH_BENE_BLOOD_DDCTBL_LBLTY_AM', 'CLM_UTLZTN_DAY_CNT', 'Time of Stay']

cat_cols = ['SEGMENT', 'PRVDR_NUM',
            'AT_PHYSN_NPI', 'OP_PHYSN_NPI', 'OT_PHYSN_NPI', 'ADMTNG_ICD9_DGNS_CD',
            'CLM_DRG_CD', 'ICD9_DGNS_CD_1', 'ICD9_DGNS_CD_2', 'ICD9_DGNS_CD_3',
 'ICD9_DGNS_CD_4', 'ICD9_DGNS_CD_5', 'ICD9_DGNS_CD_6', 'ICD9_DGNS_CD_7', 'ICD9_DGNS_CD_8',
             'ICD9_DGNS_CD_9', 'ICD9_DGNS_CD_10', 'ICD9_PRCDR_CD_1', 'ICD9_PRCDR_CD_2', 'ICD9_PRCDR_CD_3', 'ICD9_PRCDR_CD_4', 'ICD9_PRCDR_CD_5', 'ICD9_PRCDR_CD_6']

# Define the imputer for numerical and categorical features
numerical_imputer = SimpleImputer(strategy='mean')  # Impute missing values with the mean for numerical features
categorical_imputer = SimpleImputer(strategy='most_frequent')  # Impute missing values with the most frequent value for categorical features

# Preprocessing for numerical data: Imputation and Scaling
numerical_transformer = Pipeline(steps=[
    ('imputer', numerical_imputer),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data: Imputation and One-hot encoding
categorical_transformer = Pipeline(steps=[
    ('imputer', categorical_imputer),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps into a single pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])

# Create a pipeline that combines preprocessing with the Lasso model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Lasso())
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
model.fit(X_train, y_train)

# Predict on the training and testing sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate   R-squared, adjusted R-squared, training error, and test error
r2 = r2_score(y_test, y_test_pred)
n = len(y_test)
k = X_test.shape[1]  # Number of predictors
adjusted_r2 = 1 - ((n - 1) / (n - k - 1)) * (1 - r2)

train_error = r2_score(y_train, y_train_pred)
test_error = r2_score(y_test, y_test_pred)

# Print the results
print(f'R-squared: {r2}')
print(f'Adjusted R-squared: {adjusted_r2}')
print(f'Training Error (R-squared): {train_error}')
print(f'Test Error (R-squared): {test_error}')