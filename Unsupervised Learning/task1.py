import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import v_measure_score, homogeneity_completeness_v_measure
from sklearn.metrics.cluster import contingency_matrix
from sklearn.cluster import AgglomerativeClustering

# Cargar el conjunto de datos de dígitos
digits = load_digits()
samples = digits.data
labels = digits.target

# Group 4 (classes 0, 2, 4)
group_classes = [0, 2, 4]
selected_indices = [i for i in range(len(labels)) if labels[i] in group_classes]
selected_samples = samples[selected_indices]

#T1 Ward Algorithm with Euclidean distance
# Original Dataset for m = 2,3,4 and 5 clusters
print("Original Dataset:")
for m in range(2, 6):
    ward_model = AgglomerativeClustering(n_clusters=m, linkage='ward')
    ward_predicted_labels = ward_model.fit_predict(selected_samples)
    #ward_predicted_labels = [2 * label for label in ward_predicted_labels]
    v_measure = v_measure_score(labels[selected_indices], ward_predicted_labels)
    print(f'Clusters: {m}, V-Measure: {v_measure}')
    
# Apply PCA
pca = PCA(n_components=0.95)
lower_dimensional_samples = pca.fit_transform(selected_samples)

# PCA 95% for m = 2,3,4 and 5 clusters
print("\nLower-Dimensional Dataset (PCA Retaining 95% of Variance):")
for m in range(2, 6):
    ward_model = AgglomerativeClustering(n_clusters=m, linkage='ward')
    ward_predicted_labels = ward_model.fit_predict(lower_dimensional_samples)
    v_measure = v_measure_score(labels[selected_indices], ward_predicted_labels)
    print(f'Clusters: {m}, V-Measure: {v_measure}')

# Best case -> Original Dataset 3 clusters
print("\nBest Case m=3 Original Dataset")
best_m = 3
best_ward = AgglomerativeClustering(n_clusters=best_m, linkage='ward')
best_predicted_labels = best_ward.fit_predict(selected_samples)
best_predicted_labels = [2 * label for label in best_predicted_labels] #Align labels to 0,2,4 classes

#Class 0 and 4 are swapped
best_predicted_labels = [4 if label == 0 else (0 if label == 4 else label) for label in best_predicted_labels]

# Detailed Analysis for Ward
ward_contingency_matrix = contingency_matrix(labels[selected_indices], best_predicted_labels)
ward_homogeneity, ward_completeness, ward_v_measure = homogeneity_completeness_v_measure(labels[selected_indices], best_predicted_labels)
ward_incorrectly_clustered_samples = sum([1 for i in range(len(selected_indices)) if labels[selected_indices][i] != best_predicted_labels[i]])
ward_error_percentage = ward_incorrectly_clustered_samples / len(selected_indices) * 100

# Plot Errors

for i in range(len(selected_indices)):
    if labels[selected_indices][i] != best_predicted_labels[i]:
        plt.figure()
        plt.gray()
        plt.matshow(selected_samples[i].reshape(8, 8))
        plt.title(f'sample from class {labels[selected_indices][i]} clustered as class {best_predicted_labels[i]}')
        plt.show()

# Print Results for Ward
print("\nResults for Ward (Hierarchical Agglomerative Clustering):")
print("Contingency Matrix:")
print(ward_contingency_matrix)
print("\nHomogeneity:", ward_homogeneity)
print("Completeness:", ward_completeness)
print("V-Measure:", ward_v_measure)
print("\nNumber of Incorrectly Clustered Samples:", ward_incorrectly_clustered_samples)
print("Error Percentage:", ward_error_percentage)