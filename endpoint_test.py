import requests
from azureml.core import Workspace, Dataset
import json

# Azure ML workspace information (fill in your details here)
subscription_id = '0772e46f-dbb0-4be0-8f04-137c44af0d28'
resource_group = 'CW2'
workspace_name = 'CW2'

# Authenticate and access your workspace
workspace = Workspace(subscription_id, resource_group, workspace_name)

# Access the dataset named 'test'
dataset = Dataset.get_by_name(workspace, name='test')

# Convert the dataset to Pandas DataFrame
df = dataset.to_pandas_dataframe()

# Assuming the endpoint expects the data in the following format:
# { "data": [ { "feature1": value1, "feature2": value2, ... }, ... ] }
data = json.dumps({"data": df.to_dict(orient='records')})


# The URL for the real-time endpoint
endpoint_url = 'https://cw2-xlrtj.northeurope.inference.ml.azure.com/score'



# Send the request
response = requests.post(endpoint_url, data=data, headers=headers)

# Check the response
if response.status_code == 200:
    print("Success: The response from the model:", response.json())
else:
    print("Failed to fetch predictions", response.text)
