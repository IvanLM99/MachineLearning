import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap

# Load dataset (Task 1)
group = '04'
ds = 2
data = np.loadtxt(f'ds{group}{ds}.txt')
X = data[:, :2]
y = data[:, 2:3]

# Map samples onto the alternative 2-dimensional space (Task 2a)
Phi_X = np.column_stack((X[:, 0] * X[:, 1], X[:, 0]**2 + X[:, 1]**2))

# SVM Solver using CVXPY (Task 2a)
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

# Report support vectors in the original space
print("\nSupport Vectors (Original Space):")
print(support_vectors_class_0_transformed)
print(support_vectors_class_1_transformed)

# Report decision function in the transformed space
print("\nDecision Function (Transformed Space):")
print(f"g1(x') = {w_val} @ x' + {w0_val}")

# Report decision function in the original space
print("\nDecision Function (Original Space):")
print(f"g2(x) = {w_val} @ Φ(x) + {w0_val}")

# Plot in the transformed space with decision boundary and support vectors (Task 2b1)
plt.figure(figsize=(8, 6))
plt.scatter(Phi_X[:, 0], Phi_X[:, 1], c=y, cmap=plt.cm.Paired, marker='o', edgecolors='k')
plt.scatter(support_vectors_class_0_transformed[:, 0], support_vectors_class_0_transformed[:, 1],
            s=100, facecolors='none', edgecolors='k', label='Support Vectors')
plt.scatter(support_vectors_class_1_transformed[:, 0], support_vectors_class_1_transformed[:, 1],
            s=100, facecolors='none', edgecolors='k')

# Define grid points for the decision boundary
xx, yy = np.meshgrid(np.linspace(np.min(Phi_X[:, 0]), np.max(Phi_X[:, 0]), 100),
                     np.linspace(np.min(Phi_X[:, 1]), np.max(Phi_X[:, 1]), 100))

# Calculate decision values for the decision boundary
grid_points_transformed = np.c_[xx.ravel(), yy.ravel()]
decision_values_transformed = grid_points_transformed @ w_transformed[:2] + w_transformed[2]
decision_values_transformed = decision_values_transformed.reshape(xx.shape)

# Plot decision boundary
plt.contour(xx, yy, decision_values_transformed, colors='k', levels=[0], alpha=0.5, linestyles=['-'])
plt.title('Training Samples in Transformed Space')
plt.xlabel('Φ(x)_1')
plt.ylabel('Φ(x)_2')
plt.legend()
plt.show()

# Plot in the original space (Task 2b2)
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, marker='o', edgecolors='k')
plt.scatter(support_vectors_orig[:, 0], support_vectors_orig[:, 1],
            s=100, facecolors='none', edgecolors='k', label='Support Vectors')
plt.title('Training Samples in Original Space')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plot decision boundary (circular) in the original space
theta = np.linspace(0, 2 * np.pi, 100)
circle_x = np.cos(theta)
circle_y = np.sin(theta)
plt.plot(circle_x, circle_y, color='k', linestyle='--', alpha=0.5, label='Decision Boundary')

plt.legend()
plt.show()

# Define grid points in the original space
xx_orig, yy_orig = np.meshgrid(np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100),
                               np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100))
grid_points_orig = np.c_[xx_orig.ravel(), yy_orig.ravel()]

# Calculate decision values in the original space
decision_values_orig_map = grid_points_orig @ w_val[:2] + w0_val
decision_values_orig_map = decision_values_orig_map.reshape(xx_orig.shape)

# Plot classification map in the original space (Task 2b3)
plt.figure(figsize=(8, 6))
plt.contourf(xx_orig, yy_orig, decision_values_orig_map, cmap=ListedColormap(['red', 'green']), alpha=1)

# Plot circular decision boundary in the original space
theta = np.linspace(0, 2 * np.pi, 100)
circle_x = np.cos(theta)
circle_y = np.sin(theta)
plt.plot(circle_x, circle_y, color='k', linestyle='-', alpha=0.5)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, marker='o', edgecolors='k')
plt.title('Classification Map in Original Space')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()


# Use scikit-learn SVC with precomputed kernel (Task 2c)
Gram_matrix = Phi_X @ Phi_X.T
clf_linear = SVC(kernel='precomputed', C=1e16)
clf_linear.fit(Gram_matrix, y.ravel())

# Report support vectors and decision function from scikit-learn SVC (Task 2c)
print("\nSupport Vectors (SVC - Linear Kernel):")
print(X[clf_linear.support_])
print("\nDecision Function (SVC - Linear Kernel):")
print("g(x) = {} @ Φ(x) + {}".format(clf_linear.coef_.flatten(), clf_linear.intercept_))

# Use scikit-learn SVC with RBF kernel (Task 2d)
clf_rbf = SVC(kernel='rbf', C=1e16, gamma=1)
clf_rbf.fit(X, y.ravel())

# Report support vectors and decision function from scikit-learn SVC with RBF kernel (Task 2d)
print("\nSupport Vectors (SVC - RBF Kernel):")
print(clf_rbf.support_vectors_)
print("\nDecision Function (SVC - RBF Kernel):")
print("Intercept (w0):", clf_rbf.intercept_)
print("Dual Coefficients (alphas):", clf_rbf.dual_coef_)

# Plot in the original space with decision boundary from scikit-learn SVC with RBF kernel (Task 2d)
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, marker='o', edgecolors='k')
plt.scatter(clf_rbf.support_vectors_[:, 0], clf_rbf.support_vectors_[:, 1],
            s=100, facecolors='none', edgecolors='k', label='Support Vectors')
plt.title('Training Samples in Original Space (SVC - RBF Kernel)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plot decision boundary from scikit-learn SVC with RBF kernel
decision_values_orig_rbf = clf_rbf.decision_function(grid_points_orig)
decision_values_orig_rbf = decision_values_orig_rbf.reshape(xx_orig.shape)
plt.contour(xx_orig, yy_orig, decision_values_orig_rbf, colors='k', levels=[0], alpha=0.5, linestyles=['-'])
plt.legend()
plt.show()

