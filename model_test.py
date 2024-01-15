import os
from experiment import buildModel, assessModel
import pandas as pd

# Load the model
model = buildModel

# Load the test dataset
test_data = pd.read_csv('archive/test.csv')
X_test = test_data.drop('Activity', axis=1)
y_test = test_data['Activity']


# Evaluate the model
evaluation_metrics = assessModel
# Output the evaluation metrics
print(evaluation_metrics)

# Optionally, you can write the evaluation metrics to a file or log them to GitHub
# If you use GitHub's logging commands, the metrics will be available in the Actions log
for metric, value in evaluation_metrics.items():
    print(f"::set-output name={metric}::{value}")