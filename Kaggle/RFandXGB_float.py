import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import numpy as np

# Create file paths
train_path =  f"data/train_final.csv"
test_path =  f"data/test_final.csv"

# Load the training and testing data
train_data = pd.read_csv(train_path, sep=",")
test_data = pd.read_csv(test_path, sep=",")

# Define numerical and categorical columns
numerical_cols = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
categorical_cols = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
# categorical_cols_test = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']

# Impute missing values in numerical columns with mean
numerical_imputer = SimpleImputer(strategy='mean')
train_data[numerical_cols] = numerical_imputer.fit_transform(train_data[numerical_cols])
test_data[numerical_cols] = numerical_imputer.transform(test_data[numerical_cols])

# Impute missing values in categorical columns with most frequent values
categorical_imputer = SimpleImputer(strategy='most_frequent')
train_data[categorical_cols] = categorical_imputer.fit_transform(train_data[categorical_cols])
test_data[categorical_cols] = categorical_imputer.transform(test_data[categorical_cols])

# Include the missing category 'Holand-Netherlands' in the train data
if 'Holand-Netherlands' not in train_data['native.country']:
    new_row = {}
    
    # Fill numerical columns with mean value
    for col in numerical_cols:
        new_row[col] = train_data[col].mean()
    
    # Fill categorical columns with the most frequent value
    for col in train_data.columns:
        if col not in numerical_cols and col != 'native.country':
            new_row[col] = train_data[col].mode()[0]
    
    # Fill 'native.country' column
    new_row['native.country'] = 'Holand-Netherlands'
    
     # Append the new row to the data frame
    new_row_df = pd.DataFrame([new_row])
    train_data = pd.concat([train_data, new_row_df], ignore_index=True)
    
# Convert 'income' to categorical type with specific class names
train_data['income>50K'] = train_data['income>50K'].map({0: '<=50K', 1: '>50K'}).astype('category')

# Separate target from predictors
y = train_data['income>50K'].cat.codes  # Convert to numerical codes
X = train_data.drop('income>50K', axis=1)

# Encode categorical variables using one-hot encoding
encoder = OneHotEncoder(drop=None, sparse=False)
train_encoded = pd.DataFrame(encoder.fit_transform(train_data[categorical_cols]))
test_encoded = pd.DataFrame(encoder.transform(test_data[categorical_cols]))

# Combine numerical and hot-encoded categorical variables
X_train_full = pd.concat([pd.DataFrame(train_data[numerical_cols]), train_encoded], axis=1)
X_test_full = pd.concat([pd.DataFrame(test_data[numerical_cols]), test_encoded], axis=1)

# Initialize KFold with 5 splits
kf = KFold(n_splits=5, shuffle=True, random_state=0)

# Initialize the models
rf_model = RandomForestClassifier(n_estimators=1000, random_state=0) #n_estimators=100
xgb_model = XGBClassifier(n_estimators=1000, learning_rate=0.05, early_stopping_rounds=5)

# Store results
rf_auc_scores = []
xgb_auc_scores = []

# For demonstration: Split the data into train and test for each fold
for train_index, val_index in kf.split(X_train_full):
    X_train, X_val = X_train_full.iloc[train_index], X_train_full.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Train the Random Forest
    X_train.columns = X_train.columns.astype(str)  # Convert column names to strings
    X_val.columns = X_val.columns.astype(str)  # Convert column names to strings
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict_proba(X_val)[:, 1]
    rf_auc = roc_auc_score(y_val, rf_preds)
    rf_auc_scores.append(rf_auc)

    # Train the XGBoost
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_preds = xgb_model.predict_proba(X_val)[:, 1]
    xgb_auc = roc_auc_score(y_val, xgb_preds)
    xgb_auc_scores.append(xgb_auc)
    
# Print average results
print(f"Average Random Forest AUC: {np.mean(rf_auc_scores)}")
print(f"Average XGBoost AUC: {np.mean(xgb_auc_scores)}")

X_train_full.columns = X_train_full.columns.astype(str)  # Convert column names to strings
X_test_full.columns = X_test_full.columns.astype(str)  # Convert column names to strings
# Train the Random Forest on the full training data
rf_model.fit(X_train_full, y)

# Make predictions on the test data
rf_preds_test = rf_model.predict_proba(X_test_full)[:, 1]

# Train the XGBoost on the full training data
xgb_model.fit(X_train_full, y, eval_set=[(X_train_full, y)], verbose=False)

# Make predictions on the test data
xgb_preds_test = xgb_model.predict_proba(X_test_full)[:, 1]

# Create a dataframe with ID and Prediction columns
output = pd.DataFrame({'ID': test_data['ID'], 'Prediction': rf_preds_test})
# Save the dataframe to a CSV file
output.to_csv('output/test_predictions_rf_proba.csv', index=False)

# Create a dataframe with ID and Prediction columns
output = pd.DataFrame({'ID': test_data['ID'], 'Prediction': xgb_preds_test})
# Save the dataframe to a CSV file
output.to_csv('output/test_predictions_xgb_proba.csv', index=False)