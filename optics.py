import pandas as pd
import numpy as np
from sklearn.cluster import OPTICS
from sklearn.metrics.pairwise import haversine_distances
import time
import sys

print("\n\n\n###################################################################")
print("#                     Clustering avec OPTICS                      #")
print("################################################################### \n\n\n")

if len(sys.argv) != 2:
    print("Usage: python optics.py <max_pir_per_cluster>")
    print("Utilisation de la valeur 4000 par défaut\n\n")
    MAX_PIR_PER_CLUSTER = 4000.0
else: 
    MAX_PIR_PER_CLUSTER = float(sys.argv[1])

# Paramètres
RADIUS_KM = 45
EPS_RAD = RADIUS_KM / 6371.0  # pour post-traitement

# Lecture des données
print("Lecture du fichier CSV...")
dtype_dict = {
    'LAT': np.float32,
    'LON': np.float32,
    'PIR': np.float32
}
df = pd.read_csv("generated.csv", dtype=dtype_dict)
print(f"Chargement terminé: {len(df)} lignes")

coords = np.radians(df[['LAT', 'LON']].values)

# Clustering OPTICS
print("\nLancement du clustering OPTICS...")
optics_model = OPTICS(metric='haversine', min_samples=2, cluster_method='xi', xi=0.05)
optics_model.fit(coords)
df['cluster'] = optics_model.labels_

print("Clustering OPTICS terminé.")

# --- Post-traitement : respect des contraintes de rayon et PIR ---
def cluster_diameter_km(cluster_coords):
    if len(cluster_coords) < 2:
        return 0
    dists = haversine_distances(cluster_coords)
    return np.max(dists) * 6371

def split_cluster(indices, coords, pirs):
    clusters = []
    current = []
    current_pir = 0
    for idx in indices:
        if current:
            temp_coords = coords[current + [idx]]
            temp_pir = current_pir + pirs[idx]
            if (cluster_diameter_km(temp_coords) > RADIUS_KM) or (temp_pir > MAX_PIR_PER_CLUSTER):
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

print("\nDébut du post-traitement pour contraintes de PIR et rayon...")
new_labels = np.full(len(df), -1, dtype=int)
next_label = 1
start_time = time.time()

unique_clusters = sorted(df['cluster'].unique())
for i, label in enumerate(unique_clusters):
    if label == -1:
        continue  # bruit
    indices = df.index[df['cluster'] == label].tolist()
    cluster_coords = coords[indices]
    cluster_pirs = df['PIR'].values[indices]
    if (cluster_diameter_km(cluster_coords) <= RADIUS_KM) and (cluster_pirs.sum() <= MAX_PIR_PER_CLUSTER):
        new_labels[indices] = next_label
        next_label += 1
    else:
        # split en sous-clusters respectant les contraintes
        splits = split_cluster(indices, coords, df['PIR'].values)
        for split in splits:
            new_labels[split] = next_label
            next_label += 1
    if (i + 1) % 100 == 0 or (i + 1) == len(unique_clusters):
        elapsed = time.time() - start_time
        percent_done = (i + 1) / len(unique_clusters)
        remaining = (elapsed / percent_done) - elapsed
        print(f"Progression: {percent_done*100:.1f}% - Temps restant estimé: {remaining:.1f}s")

# Points bruit → cluster individuel
noise_indices = df.index[df['cluster'] == -1].tolist()
for idx in noise_indices:
    new_labels[idx] = next_label
    next_label += 1

df['cluster'] = new_labels
df.to_csv("res_optics.csv", index=False)

print(f"\nNombre final de clusters: {len(set(new_labels))}")
print("\n\n\n###################################################################")
print("#                     Fin du clustering OPTICS                   #")
print("################################################################### \n\n\n")
