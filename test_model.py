import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from experiment import loadData, buildModel, assessModel  # replace 'your_module_name' with the actual name

class TestModelMethods(unittest.TestCase):

    def setUp(self):
        # Mock data for testing
        self.mock_data = pd.DataFrame({
            'Feature1': np.random.rand(10),
            'Feature2': np.random.rand(10),
            'Activity': np.random.choice(['A', 'B'], 10)
        })
        self.X = self.mock_data.drop('Activity', axis=1)
        self.y = self.mock_data['Activity']

    @patch('pandas.read_csv')
    def test_loadData(self, mock_read_csv):
        # Setup mock
        mock_read_csv.return_value = self.mock_data

        # Call the function
        result = loadData('fake/path.csv')

        # Test the function
        pd.testing.assert_frame_equal(result, self.mock_data)

    def test_buildModel(self):
        model = buildModel(self.X, self.y)

        # Test if the model is correct
        self.assertIsInstance(model, LogisticRegression)

        # Test if model is trained (has coefficients)
        self.assertIsNotNone(model.coef_)

    def test_assessModel(self):
        model = LogisticRegression().fit(self.X, self.y)
        accuracy = assessModel(model, self.X, self.y)

        # Check if the returned accuracy is a float
        self.assertIsInstance(accuracy, float)

        # Check if accuracy is within expected range [0, 1]
        self.assertTrue(0 <= accuracy <= 1)

if __name__ == '__main__':
    unittest.main()
