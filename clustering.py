"""
Customer Segmentation & Clustering Module for Market Basket Analysis (CSM 355).

This module implements three unsupervised machine learning algorithms:
1. K-Means Clustering (Centroid-based)
2. DBSCAN Clustering (Density-based)
3. Agglomerative Hierarchical Clustering (Connectivity-based)

Evaluation Metrics Implemented:
- Silhouette Score: S(i) = [b(i) - a(i)] / max(a(i), b(i))
- Davies-Bouldin Index: DB = (1/k) * sum(max_{j != i} ((sigma_i + sigma_j) / d(c_i, c_j)))
"""

from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score


def run_kmeans(
    rfm_scaled: pd.DataFrame,
    n_clusters: int = 3,
    max_k_elbow: int = 10,
    random_state: int = 42
) -> Tuple[KMeans, np.ndarray, pd.DataFrame, List[float]]:
    """
    Fits K-Means Clustering model on standardized RFM features and calculates
    Within-Cluster Sum of Squares (WCSS / Inertia) across k in [2, max_k_elbow] for Elbow plot analysis.

    Mathematical Centroid Update Rule:
    c_k = (1 / |S_k|) * sum_{x_i in S_k} x_i

    Args:
        rfm_scaled (pd.DataFrame): Standardized RFM DataFrame.
        n_clusters (int): Selected number of clusters k. Default is 3.
        max_k_elbow (int): Maximum k to compute inertia for Elbow curve. Default is 10.
        random_state (int): Random seed for centroid initialization determinism.

    Returns:
        Tuple[KMeans, np.ndarray, pd.DataFrame, List[float]]:
            - fitted model: Sklearn KMeans object.
            - labels: Cluster assignments for each row (0 to k-1).
            - centroids: DataFrame of cluster centroids in original scaled feature space.
            - wcss_list: List of WCSS (inertia) values for k=1 to max_k_elbow.
    """
    if rfm_scaled.empty:
        raise ValueError("Provided RFM dataset is empty.")

    # Calculate WCSS / Inertia for Elbow Method visualization
    wcss_list = []
    max_k = min(max_k_elbow, len(rfm_scaled))
    for k in range(1, max_k + 1):
        km_temp = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
        km_temp.fit(rfm_scaled)
        wcss_list.append(float(km_temp.inertia_))

    # Fit final K-Means model with user-specified n_clusters
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=random_state)
    labels = kmeans.fit_predict(rfm_scaled)

    centroids = pd.DataFrame(
        kmeans.cluster_centers_,
        columns=rfm_scaled.columns,
        index=[f"Cluster {i}" for i in range(n_clusters)]
    )

    return kmeans, labels, centroids, wcss_list


def run_dbscan(
    rfm_scaled: pd.DataFrame,
    eps: float = 0.5,
    min_samples: int = 5
) -> Tuple[DBSCAN, np.ndarray]:
    """
    Fits Density-Based Spatial Clustering of Applications with Noise (DBSCAN).
    Detects arbitrary shaped clusters and flags noise observations with label -1.

    Mathematical Core Concept:
    - Core point: |N_eps(p)| >= min_samples
    - Noise point: Point not reachable from any core point (Label = -1).

    Args:
        rfm_scaled (pd.DataFrame): Standardized RFM DataFrame.
        eps (float): Epsilon neighborhood radius distance. Default is 0.5.
        min_samples (int): Minimum points required to form a dense region core. Default is 5.

    Returns:
        Tuple[DBSCAN, np.ndarray]:
            - fitted model: Sklearn DBSCAN object.
            - labels: Cluster assignments (-1 for noise, >=0 for valid clusters).
    """
    if rfm_scaled.empty:
        raise ValueError("Provided RFM dataset is empty.")

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(rfm_scaled)

    return dbscan, labels


def run_hierarchical(
    rfm_scaled: pd.DataFrame,
    n_clusters: int = 3,
    linkage: str = "ward"
) -> Tuple[AgglomerativeClustering, np.ndarray]:
    """
    Fits Agglomerative Hierarchical Clustering using bottom-up linkage tree construction.

    Linkage Metrics:
    - Ward: Minimizes total variance within merged clusters.
    - Complete: Uses maximum pairwise distance between clusters.
    - Average: Uses average pairwise distance between clusters.

    Args:
        rfm_scaled (pd.DataFrame): Standardized RFM DataFrame.
        n_clusters (int): Number of target clusters. Default is 3.
        linkage (str): Linkage criterion ('ward', 'complete', 'average'). Default is 'ward'.

    Returns:
        Tuple[AgglomerativeClustering, np.ndarray]:
            - fitted model: Sklearn AgglomerativeClustering object.
            - labels: Cluster assignments for each observation.
    """
    if rfm_scaled.empty:
        raise ValueError("Provided RFM dataset is empty.")

    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = agg_clustering.fit_predict(rfm_scaled)

    return agg_clustering, labels


def calculate_cluster_metrics(
    rfm_scaled: pd.DataFrame,
    labels: np.ndarray
) -> Dict[str, Any]:
    """
    Computes rigorous evaluation metrics for clustering performance:
    Silhouette Score and Davies-Bouldin Index (DBI).

    --- MATHEMATICAL FORMULAS FOR VIVA PREPARATION ---

    1. Silhouette Score:
       S(i) = [b(i) - a(i)] / max(a(i), b(i))
       Where:
       - a(i): Mean intra-cluster distance of sample i to all other points in the same cluster.
       - b(i): Mean nearest-cluster distance of sample i to points in the closest neighboring cluster.
       Range: [-1, +1]. Higher is better (+1 indicates perfectly distinct, dense clusters).

    2. Davies-Bouldin Index (DBI):
       DB = (1 / k) * sum_{i=1}^k max_{j != i} [ (sigma_i + sigma_j) / d(c_i, c_j) ]
       Where:
       - sigma_i: Average distance of all points in cluster i to centroid c_i.
       - d(c_i, c_j): Distance between centroids c_i and c_j.
       Range: [0, inf). Lower is better (0 indicates optimal cluster separation and compactness).

    Args:
        rfm_scaled (pd.DataFrame): Standardized feature matrix.
        labels (np.ndarray): Array of cluster labels.

    Returns:
        Dict[str, Any]: Dictionary containing Silhouette Score, Davies-Bouldin Index,
                        number of clusters, and noise count.
    """
    unique_labels = set(labels)
    # Exclude noise label (-1) if present
    non_noise_labels = unique_labels - {-1}
    num_clusters = len(non_noise_labels)
    noise_count = int(np.sum(labels == -1))

    # Defensive check: Metrics require at least 2 distinct clusters and no pure singletons
    if num_clusters < 2 or len(rfm_scaled) <= num_clusters:
        return {
            "silhouette_score": None,
            "davies_bouldin_index": None,
            "n_clusters": num_clusters,
            "noise_count": noise_count,
            "status": "Insufficient valid clusters for metric calculation (Requires >= 2 non-noise clusters)."
        }

    # Filter out noise points for valid evaluation if DBSCAN produced noise
    if -1 in unique_labels:
        valid_mask = labels != -1
        eval_data = rfm_scaled[valid_mask]
        eval_labels = labels[valid_mask]
    else:
        eval_data = rfm_scaled
        eval_labels = labels

    try:
        sil_score = float(silhouette_score(eval_data, eval_labels))
        dbi_score = float(davies_bouldin_score(eval_data, eval_labels))
    except Exception as e:
        return {
            "silhouette_score": None,
            "davies_bouldin_index": None,
            "n_clusters": num_clusters,
            "noise_count": noise_count,
            "status": f"Error computing metrics: {str(e)}"
        }

    return {
        "silhouette_score": round(sil_score, 4),
        "davies_bouldin_index": round(dbi_score, 4),
        "n_clusters": num_clusters,
        "noise_count": noise_count,
        "status": "Success"
    }


def compute_cluster_profiles(rfm_raw: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """
    Computes unscaled mean RFM values per cluster to allow intuitive business profiling.

    Args:
        rfm_raw (pd.DataFrame): Original unscaled RFM DataFrame.
        labels (np.ndarray): Cluster labels.

    Returns:
        pd.DataFrame: Summary table with mean Recency, Frequency, Monetary, and Customer Count per cluster.
    """
    df_profile = rfm_raw.copy()
    df_profile["Cluster"] = labels
    
    profile = df_profile.groupby("Cluster").agg(
        Recency_Mean=("Recency", "mean"),
        Frequency_Mean=("Frequency", "mean"),
        Monetary_Mean=("Monetary", "mean"),
        Customer_Count=("Recency", "count")
    ).reset_index()

    profile["Recency_Mean"] = profile["Recency_Mean"].round(1)
    profile["Frequency_Mean"] = profile["Frequency_Mean"].round(1)
    profile["Monetary_Mean"] = profile["Monetary_Mean"].round(2)
    
    return profile
