# Task 1

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = "./Electricity Consumption.csv"  # Replace with your file path
data = pd.read_csv(file_path)
print (data.head())
postalcode_data = pd.read_csv('./Postal Codes - Lleida.csv')
postalcode = postalcode_data['CODPOS']

import zipfile
with zipfile.ZipFile("/content/Spain_shapefile.zip","r") as zip_ref:
    zip_ref.extractall("./Spain_shapefile")
# !zip -r  ./

# Convert time to datetime and extract date and hour
data['time'] = pd.to_datetime(data['time'])
data['date'] = data['time'].dt.date
data['hour'] = data['time'].dt.hour
print (data.head())

# Pivot to create a matrix of hourly consumption per day per postal code
pivot_data = data.pivot_table(index=['postalcode', 'date'], columns='hour', values='consumption', aggfunc='sum', fill_value=0)
print (pivot_data.head())

# postalcode_grouped = pivot_data.groupby('postalcode').mean()
# print (postalcode_grouped.head())
features_scaled = pivot_data.div(pivot_data.sum(axis=1), axis=0)
# print (features_scaled.head())

# Determine the optimal number of clusters using silhouette scores
range_n_clusters = range(2, 11)
silhouette_scores = []

for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features_scaled)
    silhouette_avg = silhouette_score(features_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Plot silhouette scores
plt.figure(figsize=(8, 5))
plt.plot(range_n_clusters, silhouette_scores, marker='o')
plt.title("Silhouette Scores for Different Cluster Sizes")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.show()

# Choose the optimal number of clusters and fit KMeans
optimal_clusters = range_n_clusters[np.argmax(silhouette_scores)]
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(features_scaled)

# Add cluster labels to the data
pivot_data['cluster'] = cluster_labels
pivot_data.to_csv('postalcode_clusters.csv', index=True)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_scaled)

# Add PCA components for visualization
pivot_data['pca1'] = features_pca[:, 0]
pivot_data['pca2'] = features_pca[:, 1]

# Visualize the clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x='pca1',
    y='pca2',
    hue='cluster',
    palette='Set2',
    data=pivot_data,
    style='cluster'
)
plt.title('Clustering of Typical Daily Electricity Load Curves by Postal Code')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# Analyze the cluster centroids to identify common patterns
centroids = kmeans.cluster_centers_
centroid_df = pd.DataFrame(centroids, columns=features_scaled.columns)
print("Cluster Centroids (Typical Daily Patterns):")
print(centroid_df)