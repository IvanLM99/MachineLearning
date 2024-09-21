import numpy as np
import cvxpy as cp

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