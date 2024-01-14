# iris_model.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import joblib
import os
print("Current Working Directory:", os.getcwd())

# Creating Dataset and including the first row by setting no header as input
dataset = pd.read_csv('iris.csv')
# Renaming the columns
dataset.columns = ['sepal length in cm', 'sepal width in cm', 'petal length in cm', 'petal width in cm', 'species']

# Creating the dependent variable class
factor = pd.factorize(dataset['species'])
dataset.species = factor[0]
definitions = factor[1]

# Splitting the data into independent and dependent variables
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values

# Creating the Training and Test set from data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=21)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fitting Random Forest Classification to the Training set
classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=42)
classifier.fit(X_train, y_train)

# Save the model, reversefactor, and scaler
joblib.dump((classifier, definitions, scaler), 'reafforestation_model.pkl')
