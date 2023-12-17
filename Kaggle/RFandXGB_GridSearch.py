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
from sklearn.model_selection import GridSearchCV

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

# Convert all column names to strings
X_train_full.columns = X_train_full.columns.astype(str)
X_test_full.columns = X_test_full.columns.astype(str)

# Initialize KFold with 5 splits
kf = KFold(n_splits=5, shuffle=True, random_state=0)

# Initialize the models
rf_model = RandomForestClassifier(n_estimators=1000, random_state=0) #n_estimators=100
xgb_model = XGBClassifier(n_estimators=1000, learning_rate=0.05)

# Store results
rf_auc_scores = []
xgb_auc_scores = []

# Define parameter grid for RandomForest
rf_param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Define parameter grid for XGBoost
xgb_param_grid = {
    'n_estimators': [100, 500, 1000],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.5, 0.7, 1.0]
}

# Initialize the GridSearchCV object for RandomForest
rf_grid_search = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=2)
# Fit the GridSearchCV object to the data
rf_grid_search.fit(X_train_full, y)
# Print the best parameters and the corresponding AUC score
print("Best parameters for RandomForest: ", rf_grid_search.best_params_)
print("Best AUC score for RandomForest: ", rf_grid_search.best_score_)

# Initialize the GridSearchCV object for XGBoost
xgb_grid_search = GridSearchCV(estimator=xgb_model, param_grid=xgb_param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=2)
# Fit the GridSearchCV object to the data
xgb_grid_search.fit(X_train_full, y)
# Print the best parameters and the corresponding AUC score
print("Best parameters for XGBoost: ", xgb_grid_search.best_params_)
print("Best AUC score for XGBoost: ", xgb_grid_search.best_score_)

# Update the models with the best parameters found by GridSearchCV
rf_model_best = RandomForestClassifier(**rf_grid_search.best_params_, random_state=0)
xgb_model_best = XGBClassifier(**xgb_grid_search.best_params_)

# Convert column names to strings (required for fitting the models)
X_train_full.columns = X_train_full.columns.astype(str)
X_test_full.columns = X_test_full.columns.astype(str)

# Train the Random Forest on the full training data
rf_model_best.fit(X_train_full, y)
# Make predictions on the test data (probabilities)
rf_preds_test = rf_model_best.predict_proba(X_test_full)[:, 1]

# Train the XGBoost on the full training data
xgb_model_best.fit(X_train_full, y)  # No need for eval_set here as this is the final training
# Make predictions on the test data (probabilities)
xgb_preds_test = xgb_model_best.predict_proba(X_test_full)[:, 1]

# Create a dataframe with ID and Prediction columns for Random Forest
output_rf = pd.DataFrame({'ID': test_data['ID'], 'Prediction': rf_preds_test})
# Save the dataframe to a CSV file
output_rf.to_csv('output/test_predictions_rf_GS.csv', index=False)

# Create a dataframe with ID and Prediction columns for XGBoost
output_xgb = pd.DataFrame({'ID': test_data['ID'], 'Prediction': xgb_preds_test})
# Save the dataframe to a CSV file
output_xgb.to_csv('output/test_predictions_xgb_GS.csv', index=False)
