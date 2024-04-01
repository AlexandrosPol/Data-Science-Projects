# Soil Data Clustering Analysis
# Author: Alexandros Polyzoidis
# Date: 1/4/2024
# This script performs k-means clustering analysis on the 'SOIL DATA GR' dataset to identify different soil types based on their characteristics.
# The analysis includes data preparation, determining the optimal number of clusters, and visualization of the results.

# Load required libraries
library(readxl)
library(cluster)
library(factoextra)
library(ggplot2)

# Load the dataset
SOIL_DATA_GR <- read_excel("data/SOIL DATA GR.xlsx")

# Data cleaning

# Remove the first column - usually an ID or non-informative feature
soil_data <- SOIL_DATA_GR[,-1]

# Count and remove records with NA values to ensure the quality of our analysis
na_count_before <- sum(is.na(soil_data))
soil_data <- na.omit(soil_data)
na_count_after <- sum(is.na(soil_data))

# Data Transformation
# Scaling the data for clustering analysis
# This standardizes the feature values to have a mean of 0 and a standard deviation of 1.
# It's crucial for k-means to scale the data so that each feature contributes equally to the distance calculations.
scaled_soil_data <- scale(soil_data)

# K-Means Clustering Analysis

# Initialize a vector to store the within-cluster sum of squares (WSS) for different k values
wss <- numeric(6)

# Calculate WSS for a range of k and store the values
# The WSS is used to determine the optimal number of clusters by the Elbow Method
for (k in 1:6) {
  set.seed(123) # Ensure reproducibility
  kmeans_result <- kmeans(scaled_soil_data, centers = k, nstart = 25)
  wss[k] <- kmeans_result$tot.withinss
}

# Create a scree plot to visualize the WSS for each k value
# The 'elbow' in the plot, where the WSS begins to level off, indicates the optimal k
png("plots/elbow_plot.png")
plot(1:6, wss, type = 'b', pch = 19, col = 'darkorange', lwd = 2,
     main = "Elbow Method for Choosing Optimal k",
     xlab = "Number of Clusters (k)",
     ylab = "Total Within-Cluster Sum of Squares (WSS)")
dev.off() # Close the PNG device

# Determine the optimal number of clusters using the average silhouette method.
# The silhouette method measures the similarity of an object to its own cluster compared to other clusters.
# A higher silhouette score indicates better-defined clusters and the optimal number of clusters.

# Initialize a vector to store the average silhouette scores for k from 2 to 6
avg_silhouette_scores <- numeric(6)

# Calculate average silhouette scores for each k
for (k in 2:6) {
  set.seed(123) # Ensure reproducibility
  km_res <- kmeans(scaled_soil_data, centers = k, nstart = 25)
  silhouette_scores <- silhouette(km_res$cluster, dist(scaled_soil_data))
  avg_silhouette_scores[k] <- mean(silhouette_scores[,"sil_width"])
}

# Save and plot the average silhouette scores
png("plots/silhouette_plot.png")
plot(2:6, avg_silhouette_scores[2:6], type = 'b', pch = 19, lwd = 2,
     col = 'blue', main = "Average Silhouette Scores for Different k",
     xlab = "Number of Clusters (k)", ylab = "Average Silhouette Score")
dev.off() # Close the PNG device

# Identify the optimal number of clusters as the one with the highest average silhouette score
optimal_k <- which.max(avg_silhouette_scores)
cat("The optimal number of clusters based on the silhouette score is:", optimal_k, "\n")


# Visualize the k-means clusters for k=3 using the factoextra package.
# This visualization will help us interpret the clustering results.

# Perform k-means clustering with k=3
set.seed(123)
km <- kmeans(scaled_soil_data, centers = 3, nstart = 25)

# Visualize the clusters
png("plots/cluster_plot.png")
fviz_cluster(km, data = scaled_soil_data, palette = c("darkblue", "darkgreen", "red"),
             geom = "point", ellipse.type = "convex", ggtheme = theme_bw())
dev.off()

# Extract and report the centers of each cluster
cluster_centers <- data.frame(km$centers)
round(cluster_centers, 3)  # rounding for display purposes

# Displaying the cluster assignments for specific data instances
row_100_cluster <- km$cluster[100]
row_101_cluster <- km$cluster[101]
cat("The data instance in row 100 is assigned to cluster:", row_100_cluster, "\n")
cat("The data instance in row 101 is assigned to cluster:", row_101_cluster, "\n")

# Interpretation: The output shows the centroids of the clusters in the scaled feature space, giving us an
# indication of the central tendency of each cluster. The assignments for specific rows can help us understand
# the distribution of data instances and validate our clustering approach.


# Simplifying the dataset to focus on key soil properties: Sand %, Clay %, Silt %, and pH.
# This approach aims to determine if focusing on these core attributes improves clustering coherence.

# Use a selected subset of the scaled data for clustering
mod_scaled_soil_data <- scaled_soil_data[,c("Sand %", "Clay %", "Silt %", 'pH')]

# Re-run k-means clustering with the modified dataset
set.seed(123)
km_mod <- kmeans(mod_scaled_soil_data, centers = 3, nstart = 25)

# Save and visualize the clustering results
png("plots/cluster_plot_modified.png")
fviz_cluster(km_mod, data = mod_scaled_soil_data, palette = c("darkblue", "darkgreen", "red"),
             geom = "point", ellipse.type = "convex", ggtheme = theme_bw(),
             main = "Clustering with Simplified Dataset")
dev.off()

# Comparison and interpretation
# By focusing on core soil properties, the clustering appears to be more distinct and well-defined.
# The visualization indicates improved cluster separation, suggesting these features capture key variations in the dataset.
print("Compared to the full dataset, clustering with the simplified dataset shows clearer separation and potentially more meaningful groupings.")