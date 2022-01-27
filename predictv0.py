# Import libraries
import os
import joblib
from joblib import load
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score

PATH = os.getcwd()

clf = load(PATH + '/pickle/model.pkl')

data = pd.read_csv(PATH+"/data/input/test_cases.csv")

y = data['y']

## ValueError: X has 21 features, but ColumnTransformer is expecting 20 features as input.


data.drop('y', axis=1, inplace=True)

# customer_no is not of much value to dropping it
data = data.drop(['customer_no'], axis=1)
y_pred = clf.predict(data)

data["y_pred"] = y_pred

print(data.columns)

data.columns

print(data.head())
print("Accuracy = ", accuracy_score(y, y_pred))
print("Recall   = ", recall_score(y, y_pred, pos_label='yes'))
