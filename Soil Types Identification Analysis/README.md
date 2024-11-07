# Soil Data Clustering

- **Objective**: Performed k-means clustering on soil data from Northern Greece to identify distinct soil types, supporting optimized agricultural practices.
- **Tools**: R, KMeans Clustering, ggplot2, factoextra.
- **Process**:
  - **Data Cleaning**: Removed non-informative columns and handled missing values to ensure data quality.
  - **Data Transformation**: Scaled the data to standardize features, ensuring equal contribution in distance calculations for clustering.
  - **Modeling & Evaluation**:
    - Used the **Elbow Method** to identify the optimal number of clusters based on within-cluster sum of squares (WSS).
    - Applied the **Silhouette Method** to validate cluster cohesion and separation.
  - **Visualization**: Visualized clusters using ggplot2 and factoextra to interpret cluster separation and evaluate feature importance.
  - **Feature Reduction**: Tested clustering on a simplified dataset with key soil properties (e.g., Sand %, Clay %, Silt %, pH) to improve clustering coherence.
- **Result**: Identified three main soil clusters with clearer separation in the simplified dataset, providing actionable insights for targeted soil management.

![Elbow Plot](images/elbow_plot.png)
![Silhouette Plot](images/silhouette_plot.png)
