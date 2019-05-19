# -*- coding: utf-8 -*-
"""
Created on Tue May 14 10:54:32 2019

@author: Akshay Mathur
"""

# LR

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
from sklearn.utils import shuffle
dataset = pd.read_csv("onlyTraditionalDataset.csv")
dataset = shuffle(dataset)
X = dataset.iloc[:, : - 1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 7)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

'''
# Applying PCA to find no of features
#First initialize n_components with None and run. In the explained variance, find how many features make up more than 50%
#of the variance. then replace None with that number and run again.
from sklearn.decomposition import PCA
pca = PCA(n_components = 4)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
#explained_variance = pca.explained_variance_ratio_
'''

'''
# Now Applying LDA for 3 components as they make more than 50% of the variance
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 3)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)
'''

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(C = 4.5, penalty = 'l1', random_state = 7)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
fScore = f1_score(y_test,y_pred)



# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
import time
start = time.time()
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 5)
end = time.time()
print("Time taken by LR: ", end - start)
print("Mean Accuracy: ", accuracies.mean())
print("SD", accuracies.std())

print(cm)
print('Fscore = ', fScore)
print('Precsion = ', precision)
print('Recall = ', recall)

'''
# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
#parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
#             {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
penalty = ['l1','l2']
C = [4.5 ,4.6 ,4.7 ,4.8 ,4.9 ,5.0 ,5.1, 5.2, 5.3, 5.4, 5.5]
parameters = dict(C = C, penalty = penalty)

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)

grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
'''
