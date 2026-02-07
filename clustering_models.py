import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as sch

class ClusteringModule:
    def __init__(self, data):
        self.data = data
        self.results = {}

    def run_kmeans(self, n_clusters=3):
        model = KMeans(n_clusters=n_clusters, random_state=42)
        self.results['K-Means'] = model.fit_predict(self.data)
        return model

    def run_hierarchical(self, n_clusters=3):
        model = AgglomerativeClustering(n_clusters=n_clusters)
        self.results['Hierarchical'] = model.fit_predict(self.data)
        return model

    def run_dbscan(self, eps=0.5, min_samples=5):
        model = DBSCAN(eps=eps, min_samples=min_samples)
        self.results['DBSCAN'] = model.fit_predict(self.data)
        return model

    def run_gmm(self, n_components=3):
        model = GaussianMixture(n_components=n_components, random_state=42)
        self.results['GMM'] = model.fit_predict(self.data)
        return model

    def run_meanshift(self):
        model = MeanShift()
        self.results['Mean Shift'] = model.fit_predict(self.data)
        return model

    def get_elbow_data(self, max_k=10):
        wcss = []
        for i in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(self.data)
            wcss.append(kmeans.inertia_)
        return wcss

    def get_pca_projection(self, labels, n_components=2):
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(self.data)
        df_pca = pd.DataFrame(data=components, columns=[f'PC{i+1}' for i in range(n_components)])
        df_pca['Cluster'] = labels
        return df_pca

if __name__ == "__main__":
    # Test clustering
    X = np.random.rand(100, 5)
    cm = ClusteringModule(X)
    cm.run_kmeans(3)
    pca_df = cm.get_pca_projection(cm.results['K-Means'])
    print("Clustering PCA projection head:")
    print(pca_df.head())
