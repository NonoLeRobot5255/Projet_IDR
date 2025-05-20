import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import haversine_distances
import time

print("\n\n\n###################################################################")
print("#                     Clustering avec DBscan                      #")
print("################################################################### \n\n\n")


# Read the CSV with the same dtype as in balltree.py
# LAT, LON, PIR columns
dtype_dict = {
    'LAT': np.float32,
    'LON': np.float32,
    'PIR': np.float32
}
RADIUS_KM = 45
MAX_PIR = 2000.0
EPS = RADIUS_KM / 6371.0

df = pd.read_csv('generated.csv', dtype=dtype_dict)
coords = np.radians(df[['LAT', 'LON']].values)

# Try different values for eps and min_samples
# You can adjust these values for better clustering results
EPS = 0.5 * RADIUS_KM / 6371.0  # Increase eps (try 0.5x the original radius)
MIN_SAMPLES = 2  # Lower min_samples to allow smaller clusters

clustering = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, metric='haversine').fit(coords)
df['cluster'] = clustering.labels_

# Post-process clusters to enforce max diameter and PIR constraints
def cluster_diameter_km(cluster_coords):
    if len(cluster_coords) < 2:
        return 0
    dists = haversine_distances(cluster_coords)
    return np.max(dists) * 6371

def split_cluster(indices, coords, pirs):
    # Greedy split: sort by PIR, split if PIR sum or diameter exceeded
    clusters = []
    current = []
    current_pir = 0
    for idx in indices:
        if current:
            temp_coords = coords[current + [idx]]
            temp_pir = current_pir + pirs[idx]
            if (cluster_diameter_km(temp_coords) > RADIUS_KM) or (temp_pir > MAX_PIR):
                clusters.append(current)
                current = [idx]
                current_pir = pirs[idx]
            else:
                current.append(idx)
                current_pir = temp_pir
        else:
            current = [idx]
            current_pir = pirs[idx]
    if current:
        clusters.append(current)
    return clusters

new_labels = np.full(len(df), -1, dtype=int)
next_label = 0
print("\nDébut du post-traitement des clusters DBSCAN...")
total_clusters = len([label for label in sorted(df['cluster'].unique()) if label != -1])
processed_clusters = 0
start_time = time.time()

for label in sorted(df['cluster'].unique()):
    if label == -1:
        continue  # skip noise
    indices = df.index[df['cluster'] == label].tolist()
    cluster_coords = coords[indices]
    cluster_pirs = df['PIR'].values[indices]
    if (cluster_diameter_km(cluster_coords) <= RADIUS_KM) and (cluster_pirs.sum() <= MAX_PIR):
        new_labels[indices] = next_label
        next_label += 1
    else:
        # Split cluster
        splits = split_cluster(indices, coords, df['PIR'].values)
        for split in splits:
            new_labels[split] = next_label
            next_label += 1
    processed_clusters += 1
    if processed_clusters % 1000 == 0 or processed_clusters == total_clusters:
        elapsed = time.time() - start_time
        percent_done = processed_clusters / total_clusters
        estimated_total = elapsed / percent_done if percent_done > 0 else 0
        remaining = estimated_total - elapsed
        print(f"Progression: {percent_done*100:.1f}% - Temps écoulé: {elapsed:.1f}s - Temps restant estimé: {remaining:.1f}s")

# Assign each noise point to its own unique cluster label (after all clusters)
noise_indices = df.index[df['cluster'] == -1].tolist()
for idx in noise_indices:
    new_labels[idx] = next_label
    next_label += 1
# Assign new labels
result = df[['LAT', 'LON', 'PIR']].copy()
result['cluster'] = new_labels
result.to_csv('res.csv', index=False)

# Print the number of clusters (excluding noise if present)
num_clusters = len(set(new_labels))
print(f"Number of clusters: {num_clusters}")

print("\n\n\n###################################################################")
print("#                       fin du clustering                         #")
print("################################################################### \n\n\n")

