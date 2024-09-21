import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import svm
from matplotlib.colors import ListedColormap

# Load dataset
group = '04'
ds = 1
data = np.loadtxt(f'ds{group}{ds}.txt')
X = data[:, 0:2]
y = data[:, 2:3]

# SVM Solver using CVXPY
n_samples, n_features = X.shape
alpha = cp.Variable(n_samples)
w = cp.Variable(n_features)
w0 = cp.Variable()

# Objective function
Q = np.outer(y, y) * (X @ X.T)
Q_sqrt = np.linalg.cholesky(Q + 1e-6 * np.eye(n_samples))
objective = cp.Maximize(cp.sum(alpha) - cp.quad_form(Q_sqrt @ alpha, np.eye(n_samples)) / 2)

# Constraints
constraints = [cp.multiply(y.flatten(), X @ w + w0) >= 1 - alpha, alpha >= 0]

# Define and solve the CVXPY problem
svm_problem = cp.Problem(objective, constraints)
svm_problem.solve(solver=cp.OSQP, max_iter=40000)

w_val = w.value.flatten()
w0_val = w0.value

# Calculate the distance from each sample to the decision boundary
distances = np.abs(X @ w_val + w0_val) / np.linalg.norm(w_val)

# Identify the indices of the nearest samples from each class
nearest_indices_class_0 = np.argsort(distances[y.flatten() == 0])[:1]
nearest_indices_class_1 = np.argsort(distances[y.flatten() == 1])[:2]

# Extract the nearest support vectors
support_vectors_class_0 = X[y.flatten() == 0][nearest_indices_class_0]
support_vectors_class_1 = X[y.flatten() == 1][nearest_indices_class_1]

# Report support vectors and decision function
print("Support Vectors (Task 1a):")
print(support_vectors_class_0)
print(support_vectors_class_1)
print("\nDecision Function (Task 1a):")
print(f"g(x) = {w_val} @ x + {w0_val}")

# Generate a regular grid of points for the classification map
xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 500),
                     np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 500))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Evaluate the decision function for grid points
decision_values = grid_points @ w_val + w0_val

# Reshape decision values for contour plot
decision_values = decision_values.reshape(xx.shape)

# Plot with decision boundary
plt.figure()
plt.contourf(xx, yy, decision_values, cmap=plt.cm.Paired, levels=[-1, 0, 1], alpha=0.5)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, marker='o', edgecolors='k')
plt.scatter(support_vectors_class_0[:, 0], support_vectors_class_0[:, 1], s=100, facecolors='none', edgecolors='k', label='Support Vectors')
plt.scatter(support_vectors_class_1[:, 0], support_vectors_class_1[:, 1], s=100, facecolors='none', edgecolors='k')
plt.contour(xx, yy, decision_values, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

plt.title('SVM Decision Boundary with CVXPY')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# Set axis limits to cover the entire range of your data
plt.xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
plt.ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)

plt.show()

# Plot classification map with decision boundary
plt.figure()
plt.contourf(xx, yy, decision_values, cmap=ListedColormap(['red', 'green']), alpha=1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, marker='o', edgecolors='k')

plt.title('Classification map with CVXPY')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Set axis limits to cover the entire range of your data
plt.xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
plt.ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)

plt.show()

#############################################

# SVM solver using scikit-learn
clf = svm.SVC(kernel='linear', C=1e16)
clf.fit(X, y.ravel())
sklearn_support_vectors = clf.support_vectors_
sklearn_w = clf.coef_.flatten()
sklearn_w0 = clf.intercept_

# Compare results
print("\nSupport Vectors (Task 1c):")
print(sklearn_support_vectors)
print("\nDecision Function (Task 1c):")
print("g(x) = {} * x + {}".format(sklearn_w, sklearn_w0))

# Plot training samples
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, marker='o', edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Training Samples')

# Highlight support vectors
plt.scatter(sklearn_support_vectors[:, 0], sklearn_support_vectors[:, 1], s=100, facecolors='none', edgecolors='k', label='Support Vectors')

# Plot 2D decision curve
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
Z = np.dot(np.c_[xx.ravel(), yy.ravel()], sklearn_w) + sklearn_w0
Z = Z.reshape(xx.shape)

plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
plt.title('SVM Decision Boundary with SCIKIT')
plt.legend()
plt.show()

# Plot classification map with green and red regions
plt.figure()
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=ListedColormap(['red', 'green']), alpha=1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, marker='o', edgecolors='k')
plt.title('Classification Map with SCIKIT')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()