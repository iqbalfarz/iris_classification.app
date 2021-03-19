import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
import joblib

data = pd.read_csv('iris.csv') # reading iris.csv
print(data.head()) # print the header of data

X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']] # independent variable/ Training data
Y = data['species'] # class values

clf = SGDClassifier(loss='log', max_iter=1000, tol=1e-3) # initializing the SGDClassifier

clf.fit(X, Y) # training the classifier

joblib.dump(clf, 'model.pkl') # Saving the model for further use
