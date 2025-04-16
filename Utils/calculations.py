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
    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    kmeans.fit(skin_pix_mask)
    centres = kmeans.cluster_centers_   # Each: [Hue, Lum, Sat]
    labels = kmeans.labels_

    label_counts = np.bincount(labels)

    # Get dominant cluster (based on most pixels)
    dominant_index = labels[np.argmax(label_counts)]
    dominant_hue = centres[dominant_index]

    return dominant_hue

def bin_kmeans_values(hue_centres, label_counts):
    bins = np.linspace(0, 180, 19)
    bin_labels = np.digitize(hue_centres, bins) -1

    bin_weights = np.zeros(len(bins)-1)

    for i, bin_index in enumerate(bin_labels):
        if 0 <= bin_index < len(bin_weights):
            bin_weights[bin_index] += label_counts[i]

    # Plot histogram
    plt.bar(bins[:-1], bin_weights, width=10, color='coral', edgecolor='black', align='edge')
    plt.title("Hue Distribution by KMeans Clusters")
    plt.xlabel('Hue (HLS)')
    plt.ylabel("Weighted Pixel Count")
    plt.grid(True)
    plt.show()


def convert_hsl_opencv_to_standard(hue_cv2, sat_cv2, lum_cv2):
    hue_standard = hue_cv2 * 2
    sat_standard = (sat_cv2 / 255) * 100
    lum_standard = (lum_cv2 / 255) * 100
    return hue_standard, sat_standard, lum_standard

def convert_hsl_standard_to_opencv(hue_360, sat_pcnt, lum_pcnt):
    hue_cv2 = round(hue_360 / 2)
    sat_cv2 = round(sat_pcnt * 2.55)
    lum_cv2 = round(lum_pcnt * 2.55)
    return hue_cv2, sat_cv2, lum_cv2

