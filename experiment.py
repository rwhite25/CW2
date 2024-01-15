import pandas as pd
import numpy as np
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, 
from sklearn.preprocessing import StandardScaler


def loadData(filepath):
    return pd.read_csv(filepath)

# Load datasets
train_data = loadData('archive/train.csv')
test_data = loadData('archive/test.csv')

mlflow.autolog()

# Splitting the dataset into features and target variable
X_train = train_data.drop('Activity', axis=1)
y_train = train_data['Activity']
X_test = test_data.drop('Activity', axis=1)
y_test = test_data['Activity']

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def buildModel(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model

def assessModel(model, X, y):
    testPredictions = model.predict(X)
    acc = np.average(testPredictions == y)
    return acc

# Model Training
model = LogisticRegression()
model.fit(X_train, y_train)

# Model Prediction
y_pred = model.predict(X_test)

# Model Evaluation
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

