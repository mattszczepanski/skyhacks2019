import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.externals import joblib

# Creating Dataset and including the first row by setting no header as input
dataset = pd.read_csv('main/labels.csv', header = 0)
print('Shape of the dataset: ' + str(dataset.shape))
print(dataset.head())

# Remove validation data
del dataset['standard']
dataset = dataset[dataset.task2_class != 'validation']
print('Shape of the dataset: ' + str(dataset.shape))
print(dataset.head())

#Creating the dependent variable class
factor = pd.factorize(dataset['task2_class'])
dataset.task2_class = factor[0]
definitions = factor[1]
print(definitions)
print(dataset.head())

#Splitting the data into independent and dependent variables
X = dataset.iloc[:,2:57].values
y = dataset.iloc[:,1].values
print('The independent features set: ')
print(X[:5,:])
print('The dependent variable: ')
print(y[:5])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 21)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fitting Random Forest Classification to the Training set
classifier = RandomForestClassifier(n_estimators = 150, criterion = 'entropy', random_state = 42)
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)
#Reverse factorize (converting y_pred from 0s,1s and 2s to names
reversefactor = dict(zip(range(6), definitions))
accuracy = accuracy_score(np.array(y_test), np.array(y_pred))
print("accuracy = " + str(accuracy))
y_test = np.vectorize(reversefactor.get)(y_test)
y_pred = np.vectorize(reversefactor.get)(y_pred)
# Making the Confusion Matrix
print(pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted']))