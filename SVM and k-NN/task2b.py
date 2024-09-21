import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Load dataset 
group = '04'
ds = 2
data = np.loadtxt(f'ds{group}{ds}.txt')
X = data[:, :2]
y = data[:, 2:3]

# Map samples onto the alternative 2-dimensional space (Task 2a)
Phi_X = np.column_stack((X[:, 0] * X[:, 1], X[:, 0]**2 + X[:, 1]**2))

# SVM Solver using CVXPY 
n_samples, n_features = Phi_X.shape
alpha = cp.Variable(n_samples)
w = cp.Variable(n_features)
w0 = cp.Variable()

# Objective function
Q = np.outer(y, y) * (Phi_X @ Phi_X.T)
Q_sqrt = np.linalg.cholesky(Q + 1e-6 * np.eye(n_samples))
objective = cp.Maximize(cp.sum(alpha) - cp.quad_form(Q_sqrt @ alpha, np.eye(n_samples)) / 2)

# Constraints
constraints = [cp.multiply(y.flatten(), Phi_X @ w + w0) >= 1 - alpha, alpha >= 0]

# Define and solve the CVXPY problem
svm_problem = cp.Problem(objective, constraints)
svm_problem.solve(solver=cp.OSQP, max_iter=40000)

w_val = w.value.flatten()
w0_val = w0.value

# PLOT 1

# Calculate decision function in the transformed space
w_transformed = np.array([w_val[0], w_val[1], w0_val])
g1_values = Phi_X @ w_transformed[:2] + w_transformed[2]

# Calculate the distance from each sample to the decision boundary
distances_transformed = np.abs(g1_values) / np.linalg.norm(w_transformed[:2])

# Identify the indices of the nearest samples from each class
nearest_indices_class_0 = np.argsort(distances_transformed[y.flatten() == 0])[:1]
nearest_indices_class_1 = np.argsort(distances_transformed[y.flatten() == 1])[:2]

# Extract the nearest support vectors in the transformed space
support_vectors_class_0_transformed = Phi_X[y.flatten() == 0][nearest_indices_class_0]
support_vectors_class_1_transformed = Phi_X[y.flatten() == 1][nearest_indices_class_1]

# Plot in the transformed space with decision boundary and support vectors (Task 2b1)
plt.figure(figsize=(8, 6))
plt.scatter(Phi_X[:, 0], Phi_X[:, 1], c=y, cmap=plt.cm.Paired, marker='o', edgecolors='k')
plt.scatter(support_vectors_class_0_transformed[:, 0], support_vectors_class_0_transformed[:, 1],
            s=100, facecolors='none', edgecolors='k', label='Support Vectors')
plt.scatter(support_vectors_class_1_transformed[:, 0], support_vectors_class_1_transformed[:, 1],
            s=100, facecolors='none', edgecolors='k')

# Define grid points for the decision boundary
xx, yy = np.meshgrid(np.linspace(np.min(Phi_X[:, 0])- 1, np.max(Phi_X[:, 0])+1, 100),
                     np.linspace(np.min(Phi_X[:, 1])- 1, np.max(Phi_X[:, 1])+1, 100))

# Calculate decision values for the decision boundary
grid_points_transformed = np.c_[xx.ravel(), yy.ravel()]
decision_values_transformed = grid_points_transformed @ w_val[:2] + w0_val
decision_values_transformed = decision_values_transformed.reshape(xx.shape)

# Set axis limits to cover the entire range of your data
plt.xlim(-0.7, 0.7)
plt.ylim(0, 1.4)

# Plot decision boundary
plt.contour(xx, yy, decision_values_transformed, colors='k', levels=[0], alpha=0.5, linestyles=['-'])
plt.title('Training Samples in Transformed Space')
plt.xlabel('Φ(x)_1')
plt.ylabel('Φ(x)_2')
plt.legend()
plt.show()

# PLOT 2

# Define grid points in the original space (Task 2b2)
xx_orig, yy_orig = np.meshgrid(np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100),
                               np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100))
grid_points_orig = np.c_[xx_orig.ravel(), yy_orig.ravel()]

# Transform grid points to the alternative 2-dimensional space
grid_points_transformed = np.column_stack((grid_points_orig[:, 0] * grid_points_orig[:, 1],
                                           grid_points_orig[:, 0]**2 + grid_points_orig[:, 1]**2))

# Calculate decision values in the original space
decision_values_original  = grid_points_transformed @ w_transformed[:2] + w_transformed[2]
decision_values_original  = decision_values_original .reshape(xx_orig.shape)

# Plot in the original space with decision boundary and support vectors (Task 2b2)
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, marker='o', edgecolors='k')
# Set axis limits to cover the entire range of your data
plt.xlim(-1.1, 1.1)
plt.ylim(-1.2, 1.1)

# Plot decision boundary in the original space
plt.contour(xx_orig, yy_orig, decision_values_original , colors='k', levels=[0], alpha=0.5, linestyles=['-'])
plt.title('Training Samples in Original Space')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# PLOT 3

# Plot classification map (Task 2b3)
plt.figure(figsize=(8, 6))

# Classify based on the decision boundary
classified_map = np.zeros_like(decision_values_original)
classified_map[decision_values_original > 0] = 1  
classified_map[decision_values_original <= 0] = 0 

# Set axis limits to cover the entire range of your data
plt.xlim(-1.1, 1.1)
plt.ylim(-1.2, 1.1)

# Plot classification map using pcolormesh
plt.pcolormesh(xx_orig, yy_orig, classified_map, cmap=ListedColormap(['red', 'green']), alpha=1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, marker='o', edgecolors='k')
plt.title('Classification Map in Original Space')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
