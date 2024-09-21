import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset
group = '04'
ds = 3
data_train = np.loadtxt('ds' + group + str(ds) + 'tr.txt')
data_test = np.loadtxt('ds' + group + str(ds) + 'te.txt')

X_train = data_train[:, 0:2]
y_train = data_train[:, 2]

X_test = data_test[:, 0:2]
y_test = data_test[:, 2]

y_train = y_train.ravel()

# Task 3a: Define the design strategy
param_grid = {'n_neighbors': [1, 3, 5, 7, 9], 'metric': ['euclidean', 'manhattan']}
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5, n_jobs=-1)

# Task 3b: Find the best performing classifier
grid_search.fit(X_train, y_train)
best_knn = grid_search.best_estimator_
best_accuracy = grid_search.best_score_
print("Best parameters: ", grid_search.best_params_)
print("Best Accuracy: ", best_accuracy)