## Overview
All scripts in this folder are Python scripts for implementing a Kaggle competition which aims to leverage machine learning for the social good of better understanding income distributions across different demographics.

## Methods Used

- Random Forest:

- - with binary output ('RFandXGB_binary.py')

- - with float output ('RFandXGB_float.py')

- - with float output and grid search technique ('RFandXGB_GridSearch.py')

- - with float output, grid search technique and cross-validation ('RFandXGB_GridSearch_CrossVal.py')

- XGBoost:

- - with binary output ('RFandXGB_binary.py')

- - with float output ('RFandXGB_float.py')

- - with float output and grid search technique ('RFandXGB_GridSearch.py')

- - with float output, grid search technique and cross-validation ('RFandXGB_GridSearch_CrossVal.py')

## Running the Script
Run the script with the following command:
./run.sh

The script only runs 'RFandXGB_float.py' and 'RFandXGB_GridSearch_CrossVal.py', if you want to run all 4 files, please, uncomment the commands for them in 'run.sh' accordingly.

## Outputs
Each of the Python files primarily outputs 2 csv files with 'ID' and 'Prediction'. Apart from those csv files, some of the files will report extra information about the process and accuracy of the results themelves.