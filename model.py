import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from rich.console import Console
from rich.table import Table

def read_and_preprocess_data(file_path, dataset):
    data = pd.read_csv(file_path)
    # Remove the first 6 columns and the last column
    data = data.iloc[:, 6:-1]
    # Map categorical values to integers
    # dataTemp = data.applymap(map_to_integer)
    # Calculate PSS score
    data['PSS'] = data.iloc[:, 4:14].sum(axis=1)
    return data

def map_to_integer(value):
    mapping = {'0 - Lowest': 0, '1': 1, '2': 2, '3': 3, '4 - Highest': 4}
    return mapping.get(value, value)

dataset = 'PSS_All.csv'

data1 = read_and_preprocess_data('PSS_All.csv', dataset)
data2 = read_and_preprocess_data('PSS_Exe.csv', dataset)

# print(data1)
# print(data2)

def encode_categorical_feature(data, column_name):
    encoder = LabelEncoder()
    data[column_name] = encoder.fit_transform(data[column_name])
    return data, encoder

def encode_categorical_feature(data, column_name, encoder=None):
    if encoder is None:
        encoder = LabelEncoder()
        data[column_name] = encoder.fit_transform(data[column_name])
    else:
        data[column_name] = encoder.transform(data[column_name])
    return data, encoder

def train_linear_regression_model(X, y):
    model = LinearRegression()
    model.fit(X.values, y)
    return model

def predict_scores(model, values):
    return model.predict([values])[0]


def print_table(console, title, data):
    table = Table(title=title)
    for column in data.columns:
        table.add_column(column, justify="center")
    for index, row in data.iterrows():
        table.add_row(*[str(row[column]) for column in data.columns])
    # console.print(table)
    print(table)

def print_predicted_scores(console, predictions, title):
    table = Table(title=title)
    table.add_column("Category", justify="center")
    table.add_column("Predicted PSS Score", justify="center")
    for category, score in predictions.items():
        table.add_row(category, f"{score:.2f}")
    console.print(table)



def main(dataset):
    # Read and preprocess data
    data = read_and_preprocess_data('PSS_All.csv', dataset)
    data2 = read_and_preprocess_data('PSS_Exe.csv', dataset)

    data3 = read_and_preprocess_data(dataset)

    # Plot pie charts for each column
    # for column in data.columns:
    #     # Count the frequency of unique values in the column and sort in increasing order of value
    #     counts = data[column].value_counts().sort_index()
        
    #     # Plot pie chart
    #     plt.figure(figsize=(8, 6))
    #     plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
    #     plt.title(f'Pie Chart for {column}')
    #     plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    #     plt.show()

    # Encode categorical features
    data, gender_encoder = encode_categorical_feature(data, 'Please select your gender:')
    data2, _ = encode_categorical_feature(data2, 'Please select your gender:')
    data, ethnicity_encoder = encode_categorical_feature(data, 'Please state your ethnicity:')
    data, stage_encoder = encode_categorical_feature(data, 'Please select your stage:')

    # Combine data for analysis
    combined_data = pd.concat([data[['Please select your gender:', 'Please state your ethnicity:', 'Please select your stage:']], data['PSS']], axis=1)

    # Train linear regression models
    X_gender = data[['Please select your gender:']]
    X_ethnicity = data[['Please state your ethnicity:']]
    X_stage = data[['Please select your stage:']]
    y = combined_data['PSS']
    model_gender = train_linear_regression_model(X_gender, y)
    model_ethnicity = train_linear_regression_model(X_ethnicity, y)
    model_stage = train_linear_regression_model(X_stage, y)


    # Predict PSS scores
    predictions_gender = {
    'Male': predict_scores(model_gender, [0]),
    'Female': predict_scores(model_gender, [1]),
    'Non-binary': predict_scores(model_gender, [2]),
    'Prefer not to say': predict_scores(model_gender, [3])
    }
    predictions_ethnicity = {
    'White': predict_scores(model_ethnicity, [0]),
    'Asian': predict_scores(model_ethnicity, [1]),
    'Black': predict_scores(model_ethnicity, [2]),
    'Mixed': predict_scores(model_ethnicity, [3]),
    'Other': predict_scores(model_ethnicity, [4])
    }
    predictions_stage = {
    'Stage 1': predict_scores(model_stage, [0]),
    'Stage 2': predict_scores(model_stage, [1]),
    'Stage 3': predict_scores(model_stage, [2]),
    'Stage 4': predict_scores(model_stage, [3])
    }


    # Filter the dataset to obtain indices corresponding to male and female students
    male_indices = X_gender.index[X_gender['Please select your gender:'] == 0].tolist()
    female_indices = X_gender.index[X_gender['Please select your gender:'] == 1].tolist()

    # Randomly select two indices from each group
    random_male_indices = random.sample(male_indices, 2)
    random_female_indices = random.sample(female_indices, 2)

    # Extract the corresponding PSS scores
    actual_scores = y.loc[random_male_indices + random_female_indices]

    # Obtain the predicted PSS scores for the selected genders
    predicted_male_scores = [predictions_gender['Male'] for _ in range(2)]
    predicted_female_scores = [predictions_gender['Female'] for _ in range(2)]
    predicted_scores = predicted_male_scores + predicted_female_scores

    # Calculate the measure of error between actual and predicted scores
    error = [actual - predicted for actual, predicted in zip(actual_scores, predicted_scores)]
    mean_error = sum(error) / len(error)

    # Step 1: Filter the dataset to obtain indices corresponding to the other two categories (e.g., ethnicity and stage of study)
    # For ethnicity
    white_indices = X_ethnicity.index[X_ethnicity['Please state your ethnicity:'] == 0].tolist()
    asian_indices = X_ethnicity.index[X_ethnicity['Please state your ethnicity:'] == 1].tolist()
    black_indices = X_ethnicity.index[X_ethnicity['Please state your ethnicity:'] == 2].tolist()
    mixed_indices = X_ethnicity.index[X_ethnicity['Please state your ethnicity:'] == 3].tolist()
    other_indices = X_ethnicity.index[X_ethnicity['Please state your ethnicity:'] == 4].tolist()

    # For stage of study
    stage1_indices = X_stage.index[X_stage['Please select your stage:'] == 0].tolist()
    stage2_indices = X_stage.index[X_stage['Please select your stage:'] == 1].tolist()
    stage3_indices = X_stage.index[X_stage['Please select your stage:'] == 2].tolist()
    stage4_indices = X_stage.index[X_stage['Please select your stage:'] == 3].tolist()

    # Step 2: Randomly select two indices from each category
    random_white_indices = random.sample(white_indices, 1)
    random_asian_indices = random.sample(asian_indices, 1)
    random_black_indices = random.sample(black_indices, 1)
    random_mixed_indices = random.sample(mixed_indices, 1)
    random_other_indices = random.sample(other_indices, 1)

    random_stage1_indices = random.sample(stage1_indices, 1)
    random_stage2_indices = random.sample(stage2_indices, 1)
    random_stage3_indices = random.sample(stage3_indices, 1)
    random_stage4_indices = random.sample(stage4_indices, 1)

    # Step 3: Extract the corresponding PSS scores
    actual_scores_ethnicity = y.loc[random_white_indices + random_asian_indices + random_black_indices + random_mixed_indices + random_other_indices]
    actual_scores_stage = y.loc[random_stage1_indices + random_stage2_indices + random_stage3_indices + random_stage4_indices]

    # Step 4: Obtain the predicted PSS scores for the selected categories
    predicted_white_ethnicity = [predictions_ethnicity['White'] for _ in range(1)]
    predicted_black_ethnicity = [predictions_ethnicity['Black'] for _ in range(1)]
    predicted_asian_ethnicity = [predictions_ethnicity['Asian'] for _ in range(1)]
    predicted_mixed_ethnicity = [predictions_ethnicity['Mixed'] for _ in range(1)]
    predicted_other_ethnicity = [predictions_ethnicity['Other'] for _ in range(1)]
    predicted_scores_ethnicity = predicted_white_ethnicity + predicted_black_ethnicity + predicted_asian_ethnicity + predicted_mixed_ethnicity + predicted_other_ethnicity

    predicted_s1_stage = [predictions_stage['Stage 1'] for _ in range(1)]
    predicted_s2_stage = [predictions_stage['Stage 2'] for _ in range(1)]
    predicted_s3_stage = [predictions_stage['Stage 3'] for _ in range(1)]
    predicted_s4_stage = [predictions_stage['Stage 4'] for _ in range(1)]
    predicted_scores_stage = predicted_s1_stage + predicted_s2_stage + predicted_s3_stage + predicted_s4_stage

    # Step 5: Calculate the measure of error between actual and predicted scores
    error_ethnicity = [actual - predicted for actual, predicted in zip(actual_scores_ethnicity, predicted_scores_ethnicity)]
    mean_error_ethnicity = sum(error_ethnicity) / len(error_ethnicity)

    error_stage = [actual - predicted for actual, predicted in zip(actual_scores_stage, predicted_scores_stage)]
    mean_error_stage = sum(error_stage) / len(error_stage)


    # Create dataframes for error and mean error for gender
    error_data_gender = pd.DataFrame({
        'Actual Score': actual_scores.tolist(),
        'Predicted Score': predicted_scores,
        'Error': error
    })

    # Create dataframes for error and mean error for ethnicity
    error_data_ethnicity = pd.DataFrame({
        'Actual Score': actual_scores_ethnicity.tolist(),
        'Predicted Score': predicted_scores_ethnicity,
        'Error': error_ethnicity
    })

    # Create dataframes for error and mean error for stage of study
    error_data_stage = pd.DataFrame({
        'Actual Score': actual_scores_stage.tolist(),
        'Predicted Score': predicted_scores_stage,
        'Error': error_stage
    })

    # Calculate mean and standard deviation of PSS scores
    mean_pss_all = data['PSS'].mean()
    std_pss_all = data['PSS'].std()
    mean_pss_exe = data2['PSS'].mean()
    std_pss_exe = data2['PSS'].std()

    # Create dataframes for mean and standard deviation
    mean_std_data = pd.DataFrame({
        'Dataset': ['All Students', 'Exeter Students'],
        'Mean PSS Score': [mean_pss_all, mean_pss_exe],
        'Standard Deviation': [std_pss_all, std_pss_exe]
    })

    # Print tables
    console = Console()
    print_table(console, "PSS Dataset - Students", data)
    print_table(console, "Mean and Standard Deviation of PSS Scores", mean_std_data)

    # Print predicted PSS scores
    print_predicted_scores(console, predictions_gender, "Predicted PSS Scores by Gender")
    print_predicted_scores(console, predictions_ethnicity, "Predicted PSS Scores by Ethnicity")
    print_predicted_scores(console, predictions_stage, "Predicted PSS Scores by Stage of Study")
    print_table(console, "Error Analysis for Gender", error_data_gender)
    print_table(console, "Error Analysis for Ethnicity", error_data_ethnicity)
    print_table(console, "Error Analysis for Stage of Study", error_data_stage)

    # Print mean PSS score for white students
    white_students = data[data['Please state your ethnicity:'] == 0]
    mean_pss_white = white_students['PSS'].mean()
    console.print(f"Mean PSS Score for White Students: {mean_pss_white:.2f}")

    # Print mean PSS score for asian students
    asian_students = data[data['Please state your ethnicity:'] == 1]
    mean_pss_asian = asian_students['PSS'].mean()
    console.print(f"Mean PSS Score for Asian Students: {mean_pss_asian:.2f}")

    # Print mean PSS score for black students
    black_students = data[data['Please state your ethnicity:'] == 2]
    mean_pss_black = black_students['PSS'].mean()
    console.print(f"Mean PSS Score for Black Students: {mean_pss_black:.2f}")

    # Print mean PSS score for mixed students
    mixed_students = data[data['Please state your ethnicity:'] == 3]
    mean_pss_mixed = mixed_students['PSS'].mean()
    console.print(f"Mean PSS Score for Mixed Students: {mean_pss_mixed:.2f}")

    # Print mean PSS score for other students
    other_students = data[data['Please state your ethnicity:'] == 4]
    mean_pss_other = other_students['PSS'].mean()
    console.print(f"Mean PSS Score for Other Students: {mean_pss_other:.2f}")


if __name__ == '__main__':
    main('PSS_All.csv')
