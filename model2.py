import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder


def read_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    # Remove the first 6 columns and the last column
    data = data.iloc[:, 6:-1]
    # Map categorical values to integers
    data = data.map(map_to_integer)
    # Calculate PSS score
    data['PSS'] = data.iloc[:, 4:14].sum(axis=1)
    return data

def map_to_integer(value):
    mapping = {'0 - Lowest': 0, '1': 1, '2': 2, '3': 3, '4 - Highest': 4}
    return mapping.get(value, value)

def encode_categorical_feature(data, column_name):
    encoder = LabelEncoder()
    data[column_name] = encoder.fit_transform(data[column_name])
    return data, encoder



# Function to read the csv file
data = pd.read_csv('PSS_All.csv')
data2 = pd.read_csv('PSS_Exe.csv')

def preprocess_data(data):
    # Remove the first 6 columns
    data = data.iloc[:, 6:]

    # Remove the last column
    data = data.iloc[:, :-1]
    
    return data

# Apply preprocessing function to data and data2
data = preprocess_data(data)
data2 = preprocess_data(data2)

def map_to_integer(value):
    mapping = {
        '0 - Lowest': 0,
        '1': 1,
        '2': 2,
        '3': 3,
        '4 - Highest': 4
    }
    return mapping.get(value, value)

# Apply function to both datasets
data = data.map(map_to_integer)
data2 = data2.map(map_to_integer)

# PSS score is determined by the sum of columns 5 to 14
data['PSS'] = data.iloc[:, 4:14].sum(axis=1)
data2['PSS'] = data2.iloc[:, 4:14].sum(axis=1)

# Encode gender values
gender_mapping = {
    'Male': True,
    'Female': False
}
data['Please select your gender:'] = data['Please select your gender:'].map(gender_mapping)

# Encode stage of study values
stage_encoder = LabelEncoder()
data['Please select your stage:'] = stage_encoder.fit_transform(data['Please select your stage:'])

# Split the data into X and y
X = data[['Please select your gender:', 'Please select your stage:']]  # Selecting the gender and stage columns
y = data[['PSS']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train.values.ravel())

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Predict PSS scores for each gender and stage of study
predicted_data = pd.DataFrame({
    'Please select your gender:': [True, True, True, True],  # Male, Male, Female, Female
    'Please select your stage:': [1, 2, 3, 4]  # Assuming 1 and 2 represent different stages
})

predicted_data2 = pd.DataFrame({
    'Please select your gender:': [False, False, False, False],  
    'Please select your stage:': [1, 2, 3, 4]
})

predicted_data['Predicted PSS'] = model.predict(predicted_data[['Please select your gender:', 'Please select your stage:']])
predicted_data2['Predicted PSS'] = model.predict(predicted_data2[['Please select your gender:', 'Please select your stage:']])
print(predicted_data)
print(predicted_data2)

# Split the test data into subsets for each gender
X_test_male = X_test[X_test['Please select your gender:'] == True]
y_test_male = y_test.loc[X_test_male.index]

X_test_female = X_test[X_test['Please select your gender:'] == False]
y_test_female = y_test.loc[X_test_female.index]

# Make predictions for each gender
y_pred_male = model.predict(X_test_male)
y_pred_female = model.predict(X_test_female)

# Calculate mean squared error for each gender
mse_male = mean_squared_error(y_test_male, y_pred_male)
mse_female = mean_squared_error(y_test_female, y_pred_female)

print("Mean Squared Error (Male):", mse_male)
print("Mean Squared Error (Female):", mse_female)

# Function to train the Random Forest model
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train.values.ravel())
    return model

# Train the initial Random Forest model
model = train_model(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Initial Mean Squared Error:", mse)

# Threshold for mean squared error
threshold = 1000  # Adjust as needed

# If the mean squared error is below the threshold, add predicted PSS scores to the dataset
if mse < threshold:
    predicted_data['Predicted PSS'] = model.predict(predicted_data[['Please select your gender:', 'Please select your stage:']])
    predicted_data2['Predicted PSS'] = model.predict(predicted_data2[['Please select your gender:', 'Please select your stage:']])
    
    # Add predicted PSS scores to the dataset
    data_with_predictions = pd.concat([data.reset_index(drop=True), 
                                       predicted_data.reset_index(drop=True), 
                                       predicted_data2.reset_index(drop=True)], 
                                      axis=1)

    # Train the model again using the updated dataset
    X_updated = data_with_predictions[['Please select your gender:', 'Please select your stage:']]
    y_updated = data_with_predictions[['PSS']]
    model_updated = train_model(X_updated, y_updated)

    # Make new predictions on the test set
    y_pred_updated = model_updated.predict(X_test)

    # Calculate new mean squared error
    mse_updated = mean_squared_error(y_test, y_pred_updated)
    print("Updated Mean Squared Error:", mse_updated)
else:
    print("Mean Squared Error is above the threshold. No updates were made.")