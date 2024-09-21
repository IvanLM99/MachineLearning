import numpy as np
from sklearn.svm import SVC


# Load dataset 
group = '04'
ds = 2
data = np.loadtxt(f'ds{group}{ds}.txt')
X = data[:, :2]
y = data[:, 2:3]

# Map samples onto the alternative 2-dimensional space (Task 2a)
Phi_X = np.column_stack((X[:, 0] * X[:, 1], X[:, 0]**2 + X[:, 1]**2))

# (c) Compare results with scikit-learn SVC for 'precomputed' kernel
C = 1e16  # High value for perfect classification
svm_sklearn_precomputed = SVC(C=C, kernel='precomputed')
gram_matrix = Phi_X @ Phi_X.T  # Gram matrix calculation

svm_sklearn_precomputed.fit(gram_matrix, y.ravel())

# Report support vectors and decision function from scikit-learn SVC with 'precomputed' kernel
print("\nSupport Vectors from scikit-learn SVC (Precomputed Kernel):")
print(X[svm_sklearn_precomputed.support_])
print("Decision Function from scikit-learn SVC (Precomputed Kernel):")
print(svm_sklearn_precomputed.decision_function(gram_matrix))




