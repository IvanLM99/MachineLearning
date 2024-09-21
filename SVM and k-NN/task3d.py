import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from matplotlib.colors import ListedColormap

# Load the dataset
group = '04'
ds = 3
data_train = np.loadtxt('ds' + group + str(ds) + 'tr.txt')
data_test = np.loadtxt('ds' + group + str(ds) + 'te.txt')

X_train = data_train[:, 0:2]
y_train = data_train[:, 2]  # Remove the slicing, keep it as a 1D array

X_test = data_test[:, 0:2]
y_test = data_test[:, 2]

y_train = y_train.ravel()
y_test = y_test.ravel()


# Task 3a: Define the design strategy
param_grid = {'n_neighbors': [1, 3, 5, 7, 9], 'metric': ['euclidean', 'manhattan']}
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5, n_jobs=-1)

# Task 3b: Find the best performing classifier
grid_search.fit(X_train, y_train)
best_knn = grid_search.best_estimator_

# Task 3c: Plot the training samples on top of the classification map
h = .02
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = best_knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Task 3d: Report on classifier performance using the test dataset
y_pred = best_knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Test Accuracy:", accuracy)
print("Test Precision:", precision)
print("Test Recall:", recall)
print("Test F1-score:", f1)

plt.contourf(xx, yy, Z, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']), alpha=0.3)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=ListedColormap(['red', 'blue']), edgecolors='k', marker='o', s=50)
plt.title("Test Samples on Classification Map")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

