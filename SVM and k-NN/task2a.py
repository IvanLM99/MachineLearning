import numpy as np
import cvxpy as cp


# Load dataset (Task 2)
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