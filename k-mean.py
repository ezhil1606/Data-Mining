#1) Develop k-Means Clustering algorithm to apply clustering on the following data objects referred by
# (x, y) pair: (k =3) A1(2, 10), A2(2, 5), A3(8, 4), A4(5, 8), A5(7, 5), A6(6, 4), A7(1, 2), A8(4, 9)
# Use Euclidian distance metric to determine closest cent.- Load IRIS data set (IRIS.csv)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Euclidean distance function
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Function to initialize centroids randomly
def initialize_centroids(data, k):
    centroids_indices = np.random.choice(range(len(data)), k, replace=False)
    centroids = [data[i] for i in centroids_indices]
    return centroids

# Assign each data point to the closest centroid
def assign_to_clusters(data, centroids):
    clusters = [[] for _ in range(len(centroids))]
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        closest_centroid_index = np.argmin(distances)
        clusters[closest_centroid_index].append(point)
    return clusters

# Update centroids based on the mean of the points in each cluster
def update_centroids(clusters):
    centroids = [np.mean(cluster, axis=0) for cluster in clusters]
    return centroids

# Implementing k-Means algorithm
def k_means(data, k, max_iterations=100):
    centroids = initialize_centroids(data, k)
    for _ in range(max_iterations):
        clusters = assign_to_clusters(data, centroids)
        new_centroids = update_centroids(clusters)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return clusters, centroids

# Test data
data = np.array([[2, 10], [2, 5], [8, 4], [5, 8], [7, 5], [6, 4], [1, 2], [4, 9]])
k = 3

# Applying k-Means clustering
clusters, centroids = k_means(data, k)

# Print clusters and centroids
for i, cluster in enumerate(clusters):
    print(f"Cluster {i+1}: {cluster}")

print("Centroids:")
for centroid in centroids:
    print(centroid)

# Load IRIS dataset
iris_data = pd.read_csv("Iris.csv")

# Display first few rows of IRIS dataset
print("\nFirst few rows of IRIS dataset:")
print(iris_data.head())



#2)  Remove Class Label column from IRIS data set -
# Apply developed k-Means clustering in Question 1 on the unlabelled IRIS data set with k = 3

# Remove Class Label column
iris_data = iris_data.drop(columns=["Species"])

# Convert dataframe to numpy array
iris_data_array = iris_data.values

# Applying k-Means clustering on IRIS dataset
k = 3
clusters, centroids = k_means(iris_data_array, k)

# Print clusters and centroids
for i, cluster in enumerate(clusters):
    print(f"Cluster {i+1}: {len(cluster)} points")

print("\nCentroids:")
for centroid in centroids:
    print(centroid)
