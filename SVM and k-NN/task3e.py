import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

# Task 3e: Obtain an improved estimation using repeated, n-fold cross-validation
combined_data = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)
cv_scores = cross_val_score(best_knn, combined_data[:, :-1], combined_data[:, -1], cv=5, scoring='accuracy', n_jobs=-1)
mean_cv_accuracy = np.mean(cv_scores)

cv_precision = cross_val_score(best_knn, combined_data[:, :-1], combined_data[:, -1], cv=5, scoring='precision', n_jobs=-1)
cv_recall = cross_val_score(best_knn, combined_data[:, :-1], combined_data[:, -1], cv=5, scoring='recall', n_jobs=-1)
cv_f1 = cross_val_score(best_knn, combined_data[:, :-1], combined_data[:, -1], cv=5, scoring='f1', n_jobs=-1)

mean_cv_precision = np.mean(cv_precision)
mean_cv_recall = np.mean(cv_recall)
mean_cv_f1 = np.mean(cv_f1)

print("Mean Cross-Validation Accuracy:", mean_cv_accuracy)
print("Mean Cross-Validation Precision:", mean_cv_precision)
print("Mean Cross-Validation Recall:", mean_cv_recall)
print("Mean Cross-Validation F1-score:", mean_cv_f1)

