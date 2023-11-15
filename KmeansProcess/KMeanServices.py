import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.spatial import ConvexHull
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
import os
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


class KMeansService:
    def __init__(self, n_clusters=3, random_state=42, n_init=10):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init

    def fit_predict(self, df):
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=self.n_init)
        df['cluster'] = kmeans.fit_predict(df.drop(columns='cluster', errors='ignore'))
        self.centers = kmeans.cluster_centers_
        return df
    
    def get_silhouette_score(self, df):
        features = df.drop(columns='cluster', errors='ignore')
        labels = df['cluster']
        score = silhouette_score(features, labels)
        return score

    def get_davies_bouldin_score(self, df):
        features = df.drop(columns='cluster', errors='ignore')
        labels = df['cluster']
        score = davies_bouldin_score(features, labels)
        return score
    
    
    def get_calinski_harabasz_score(self, df):
        features = df.drop(columns='cluster', errors='ignore')
        labels = df['cluster']
        score = calinski_harabasz_score(features, labels)
        return score

    def plot_clusters(self, df):
        cluster_colors = {0: 'red', 1: 'green', 2: 'blue'}

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

        sns.scatterplot(ax=ax1, data=df, x=df.columns[0], y=df.columns[1], hue='cluster',
                        palette=cluster_colors, alpha=0.7, style='cluster', markers=['o', '^', 's'], s=100)


        for i in range(self.n_clusters):
            points = df[df['cluster'] == i][df.columns[:2]].values
            if len(points) > 2:
                hull = ConvexHull(points)
                x_hull = np.append(points[hull.vertices, 0], points[hull.vertices, 0][0])
                y_hull = np.append(points[hull.vertices, 1], points[hull.vertices, 1][0])
                ax1.fill(x_hull, y_hull, alpha=0.3, c=cluster_colors[i])

        ax1.set_title("Scatter plot with Convex Hull")
        ax1.legend(title='Cluster', loc='upper left', fontsize='small')

        melted_df = df.melt(id_vars='cluster', value_vars=[df.columns[0]], var_name='Feature', value_name='Value')
        sns.boxplot(ax=ax2, x='cluster', y='Value', data=melted_df, palette=cluster_colors)

        ax2.set_title("Boxplot of Species by cluster")

        plt.tight_layout()
        plt.show()
        
        

    
    
    
    