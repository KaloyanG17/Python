import unittest
import pandas as pd
from model import map_to_integer, read_and_preprocess_data, encode_categorical_feature, train_linear_regression_model, predict_scores

class TestPredictiveModel(unittest.TestCase):
    def setUp(self):
        # Set up test data
        self.data = pd.DataFrame({
            'Column1': ['0 - Lowest', '1', '2', '3', '4 - Highest'],
            'Column2': [1, 2, 3, 4, 5],
            'PSS': [10, 20, 30, 40, 50]
        })
    
    # Test read_and_preprocess_data function
    def test_read_and_preprocess_data(self):
        # Apply read_and_preprocess_data function
        data = read_and_preprocess_data('PSS_All.csv')
        data2 = read_and_preprocess_data('PSS_Exe.csv')
        
        # Check if the data is read and preprocessed correctly
        self.assertEqual(data.shape, (5, 10))
        self.assertEqual(data2.shape, (5, 10))
        self.assertListEqual(list(data.columns), list(data2.columns))
        self.assertTrue(data.equals(data2))

    # Test map_to_integer function
    def test_map_to_integer(self):
        self.assertEqual(map_to_integer('0 - Lowest'), 0)
        self.assertEqual(map_to_integer('1'), 1)
        self.assertEqual(map_to_integer('2'), 2)
        self.assertEqual(map_to_integer('3'), 3)
        self.assertEqual(map_to_integer('4 - Highest'), 4)
        
        
    def test_pss_score_calculation(self):
        # Test PSS score calculation
        # Create sample data to test PSS score calculation
        sample_data = pd.DataFrame({
            'Column1': [0, 1, 2, 3, 4],
            'Column2': [1, 2, 3, 4, 5],
            'Column3': [1, 2, 3, 4, 5],
        })
        # Apply PSS score calculation
        pss_scores = sample_data.iloc[:, 2:12].sum(axis=1)
        expected_pss_scores = pd.Series([sum(range(1, 11)), sum(range(2, 12)), sum(range(3, 13)), sum(range(4, 14)), sum(range(5, 15))])
        # Check if the calculated PSS scores match the expected values
        self.assertTrue(pss_scores.equals(expected_pss_scores))

    # Test encode_categorical_feature function
    def test_encode_categorical_feature(self):
        # Apply encode_categorical_feature function
        data, encoder = encode_categorical_feature(self.data, 'Column1')
        # Check if the categorical feature is encoded correctly
        self.assertListEqual(list(data['Column1']), [0, 1, 2, 3, 4])
        self.assertEqual(encoder.classes_.tolist(), ['0 - Lowest', '1', '2', '3', '4 - Highest'])

    # Test train_linear_regression_model function
    def test_train_linear_regression_model(self):
        # Create sample data to test linear regression model training
        X = self.data[['Column2']]
        y = self.data['PSS']
        # Apply train_linear_regression_model function
        model = train_linear_regression_model(X, y)
        # Check if the model is trained correctly
        self.assertIsNotNone(model)
        self.assertEqual(model.coef_[0], 10)
        self.assertEqual(model.intercept_, 0)

    # Test predict_scores function
    def test_predict_scores(self):
        # Create sample data to test score prediction
        values = [3]
        model = train_linear_regression_model(self.data[['Column2']], self.data['PSS'])
        # Apply predict_scores function
        score = predict_scores(model, values)
        # Check if the score is predicted correctly
        self.assertEqual(score, 30)
        
if __name__ == '__main__':
    unittest.main()