import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import v_measure_score, homogeneity_completeness_v_measure
from sklearn.metrics.cluster import contingency_matrix

class KMeans(BaseEstimator):

    def __init__(self, k, max_iter=100, random_state=0, tol=1e-4):
        self.k = k
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol

    def _e_step(self, X):
        self.labels_ = euclidean_distances(X, self.cluster_centers_,
                                     squared=True).argmin(axis=1)

    def _average(self, X):
        return X.mean(axis=0)

    def _m_step(self, X):
        X_center = None
        for center_id in range(self.k):
            center_mask = self.labels_ == center_id
            if not np.any(center_mask):
                # The centroid of empty clusters is set to the center of
                # everything
                if X_center is None:
                    X_center = self._average(X)
                self.cluster_centers_[center_id] = X_center
            else:
                self.cluster_centers_[center_id] = \
                    self._average(X[center_mask])

    def fit(self, X, y=None):
        n_samples = X.shape[0]
        vdata = np.mean(np.var(X, 0))

        random_state = check_random_state(self.random_state)
        self.labels_ = random_state.permutation(n_samples)[:self.k]
        self.cluster_centers_ = X[self.labels_]

        for i in range(self.max_iter):
            centers_old = self.cluster_centers_.copy()

            self._e_step(X)
            self._m_step(X)

            if np.sum((centers_old - self.cluster_centers_) ** 2) < self.tol * vdata:
                break

        return self

    def predict(self, X):
        labels = euclidean_distances(X, self.cluster_centers_,
                                     squared=True).argmin(axis=1)
        return labels

class FuzzyKMeans(KMeans):

    def __init__(self, k, m=2, max_iter=100, random_state=0, tol=1e-4):
        """
        m > 1: fuzzy-ness parameter
        The closer to m is 1, the closer to hard kmeans.
        The bigger m, the fuzzier (converge to the global cluster).
        """
        self.k = k
        assert m > 1
        self.m = m
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol

    def _e_step(self, X):
        D = 1.0 / euclidean_distances(X, self.cluster_centers_, squared=True)
        D **= 1.0 / (self.m - 1)
        D /= np.sum(D, axis=1)[:, np.newaxis]
        # shape: n_samples x k
        self.fuzzy_labels_ = D
        self.labels_ = self.fuzzy_labels_.argmax(axis=1)

    def _m_step(self, X):
        weights = self.fuzzy_labels_ ** self.m
        # shape: n_clusters x n_features
        self.cluster_centers_ = np.dot(X.T, weights).T
        self.cluster_centers_ /= weights.sum(axis=0)[:, np.newaxis]

    def fit(self, X, y=None):
        n_samples, n_features = X.shape
        vdata = np.mean(np.var(X, 0))

        random_state = check_random_state(self.random_state)
        self.fuzzy_labels_ = random_state.rand(n_samples, self.k)
        self.fuzzy_labels_ /= self.fuzzy_labels_.sum(axis=1)[:, np.newaxis]
        self._m_step(X)

        for i in range(self.max_iter):
            centers_old = self.cluster_centers_.copy()

            self._e_step(X)
            self._m_step(X)

            if np.sum((centers_old - self.cluster_centers_) ** 2) < self.tol * vdata:
                break

        return self
    
    def predict(self, X):
        D = 1.0 / euclidean_distances(X, self.cluster_centers_, squared=True)
        D **= 1.0 / (self.m - 1)
        D /= np.sum(D, axis=1)[:, np.newaxis]
        labels = np.argmax(D, axis=1)
        return labels

# Cargar el conjunto de datos de dígitos
digits = load_digits()
samples = digits.data
labels = digits.target

# Group 4 (classes 0, 2, 4)
group_classes = [0, 2, 4]
selected_indices = [i for i in range(len(labels)) if labels[i] in group_classes]
selected_samples = samples[selected_indices]

#T3. Fuzzy K-means with Euclidean distance
# Original Dataset for m = 2,3,4 and 5 clusters
print("Original Dataset:")
for m in range(2, 6):
    fuzzy_kmeans_model = FuzzyKMeans(k=m)
    fuzzy_kmeans_model.fit(selected_samples)
    fuzzy_kmeans_predicted_labels = fuzzy_kmeans_model.predict(selected_samples)
    v_measure = v_measure_score(labels[selected_indices], fuzzy_kmeans_predicted_labels)
    print(f'Clusters: {m}, V-Measure: {v_measure}')
    
# Aplicar PCA
pca = PCA(n_components=0.95)
lower_dimensional_samples = pca.fit_transform(selected_samples)

# PCA 95% for m = 2,3,4 and 5 clusters
print("\nLower-Dimensional Dataset (PCA Retaining 95% of Variance):")
for m in range(2, 6):
    fuzzy_kmeans_model = FuzzyKMeans(k=m)
    fuzzy_kmeans_model.fit(lower_dimensional_samples)
    fuzzy_kmeans_predicted_labels = fuzzy_kmeans_model.predict(lower_dimensional_samples)
    v_measure = v_measure_score(labels[selected_indices], fuzzy_kmeans_predicted_labels)
    print(f'Clusters: {m}, V-Measure: {v_measure}')

# Best case -> Original Dataset 3 clusters
print("\nBest Case m=3 Original Dataset")
best_m = 3
best_fuzzy_kmeans = FuzzyKMeans(k=best_m)
best_fuzzy_kmeans.fit(selected_samples)
best_fuzzy_kmeans_predicted_labels = best_fuzzy_kmeans.predict(selected_samples)
best_fuzzy_kmeans_predicted_labels = [2 * label for label in best_fuzzy_kmeans_predicted_labels] #Align labels to 0,2,4 classes
best_fuzzy_kmeans_predicted_labels = [4 if label == 0 else (0 if label == 4 else label) for label in best_fuzzy_kmeans_predicted_labels]

# Detailed Analysis for Fuzzy K-Means
fuzzy_kmeans_contingency_matrix = contingency_matrix(labels[selected_indices], best_fuzzy_kmeans_predicted_labels)
fuzzy_kmeans_homogeneity, fuzzy_kmeans_completeness, fuzzy_kmeans_v_measure = homogeneity_completeness_v_measure(labels[selected_indices], best_fuzzy_kmeans_predicted_labels)
fuzzy_kmeans_incorrectly_clustered_samples = sum([1 for i in range(len(selected_indices)) if labels[selected_indices][i] != best_fuzzy_kmeans_predicted_labels[i]])
fuzzy_kmeans_error_percentage = fuzzy_kmeans_incorrectly_clustered_samples / len(selected_indices) * 100

# Plot Errors
for i in range(len(selected_indices)):
    if labels[selected_indices][i] != best_fuzzy_kmeans_predicted_labels[i]:
        plt.figure()
        plt.gray()
        plt.matshow(selected_samples[i].reshape(8, 8))
        plt.title(f'sample from class {labels[selected_indices][i]} clustered as class {best_fuzzy_kmeans_predicted_labels[i]}')
        plt.show()
        
# Print Results for Fuzzy K-Means
print("\nResults for Fuzzy K-Means:")
print("Contingency Matrix:")
print(fuzzy_kmeans_contingency_matrix)
print("\nHomogeneity:", fuzzy_kmeans_homogeneity)
print("Completeness:", fuzzy_kmeans_completeness)
print("V-Measure:", fuzzy_kmeans_v_measure)
print("\nNumber of Incorrectly Clustered Samples:", fuzzy_kmeans_incorrectly_clustered_samples)
print("Error Percentage:", fuzzy_kmeans_error_percentage)