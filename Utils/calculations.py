from Utils import *
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def get_best_k(skin_pix_mask):
    silhouette_scores = []
    k_range = range(2, 8)   # Only consider k clusters from 2 through 7

    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=10)
        labels = kmeans.fit_predict(skin_pix_mask)
        score = silhouette_score(skin_pix_mask, labels)
        silhouette_scores.append(score)

    # Get the best k for kmeans
    best_index = np.argmax(silhouette_scores)
    best_k = list(k_range)[best_index]

    # Plot silhouette scores
    plt.plot(k_range, silhouette_scores, 'go-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Optimal number of clusters via Silhouette Score')
    plt.grid(True)
    plt.show()

    return best_k

def get_dom_hue(skin_pix_mask):
    best_k = get_best_k(skin_pix_mask)

    # Run KMeans
    kmeans = KMeans(n_clusters=best_k, n_init=10)
    kmeans.fit(skin_pix_mask)

    # Get cluster centres
    hue_centres = kmeans.cluster_centers_[:, 0]
    labels = kmeans.labels_
    label_counts = np.bincount(labels)

    # Get dominant cluster (based on most pixels)
    dominant_index = labels[np.argmax(label_counts)]
    dominant_hue = hue_centres[dominant_index]

    return dominant_hue



