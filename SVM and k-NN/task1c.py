import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from matplotlib.colors import ListedColormap

# Load dataset
group = '04'
ds = 1
data = np.loadtxt(f'ds{group}{ds}.txt')
X = data[:, 0:2]
y = data[:, 2:3]

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






