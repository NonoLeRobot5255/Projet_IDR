import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import time
import sys

print("\n\n\n###################################################################")
print("#                     Clustering avec BallTree                    #")
print("################################################################### \n\n\n")

if len(sys.argv) != 2:
    print("Usage: python balltree.py <max_pir_per_cluster>")
    print("Utilisation de la valeur 4000 par défaut\n\n")
    MAX_PIR_PER_CLUSTER = 4000.0
else : 
    MAX_PIR_PER_CLUSTER = float(sys.argv[1])

# Paramètres de clustering
RADIUS_KM = 45  # rayon géographique de voisinage

# Lecture du CSV
dtype_dict = {
    'LAT': np.float32,
    'LON': np.float32,
    'PIR': np.float32  # colonne bande passante
}

# Afficher message de progression
print("Lecture du fichier CSV...")
donnees = pd.read_csv("generated.csv", delimiter=",", dtype=dtype_dict)
print(f"Chargement terminé: {len(donnees)} lignes")

# Initialisation des colonnes
donnees['nb_voisins'] = np.zeros(len(donnees), dtype=np.int32)
donnees['cluster'] = np.zeros(len(donnees), dtype=np.int32)
donnees['VID'] = [[] for _ in range(len(donnees))]

# Calcul des voisins avec BallTree
coords = np.radians(donnees[['LAT', 'LON']].values.astype(np.float32))
print("\n\nConstruction du BallTree...")
tree = BallTree(coords, metric='haversine', leaf_size=100)  # Leaf size augmenté pour performance
radius = RADIUS_KM / 6371.0  # Conversion km -> radians

# Traiter par lots pour éviter les interruptions de mémoire
print("\n\nRecherche des voisins...")
batch_size = 5000  # Ajuster selon la mémoire disponible
num_points = len(coords)
indices = []

start_time = time.time()
for i in range(0, num_points, batch_size):
    end_idx = min(i + batch_size, num_points)
    print(f"\n\nTraitement du lot {i//batch_size + 1}/{(num_points + batch_size - 1)//batch_size}: points {i} à {end_idx-1}")
    batch_indices = tree.query_radius(coords[i:end_idx], r=radius)
    
    # Ajuster les indices pour correspondre à la position globale
    for j, idx_array in enumerate(batch_indices):
        batch_indices[j] = idx_array.astype(np.int32)
    
    indices.extend(batch_indices)
    
    # Afficher la progression
    elapsed = time.time() - start_time
    percent_done = end_idx / num_points
    if percent_done > 0:
        estimated_total = elapsed / percent_done
        remaining = estimated_total - elapsed
        print(f"Progression: {percent_done*100:.1f}% - Temps écoulé: {elapsed:.1f}s - Temps restant estimé: {remaining:.1f}s")

print("\n\nCalcul des statistiques de voisinage...")
donnees['nb_voisins'] = np.array([len(neighs) - 1 for neighs in indices], dtype=np.int32)  # -1 pour exclure le point lui-même
donnees['VID'] = [list(neighs[neighs != i]) for i, neighs in enumerate(indices)]

# Assignation des clusters avec contraintes
print("\nCréation des clusters...")
n = 1  # identifiant de cluster
cluster_map = np.zeros(len(donnees), dtype=np.int32)
sorted_indices = donnees['nb_voisins'].sort_values(ascending=False).index

start_time = time.time()
processed = 0
num_to_process = len(sorted_indices)

for index in sorted_indices:
    if cluster_map[index] != 0:
        processed += 1
        continue  # déjà assigné

    pir_initial = donnees.at[index, 'PIR']
    lat_initial = donnees.at[index, 'LAT']
    lon_initial = donnees.at[index, 'LON']

    if pir_initial <= MAX_PIR_PER_CLUSTER:
        cluster_members = [index]
        total_pir = pir_initial
        latlons = [(lat_initial, lon_initial)]

        voisins = [(v, donnees.at[v, 'PIR'], donnees.at[v, 'LAT'], donnees.at[v, 'LON']) for v in donnees.at[index, 'VID'] if cluster_map[v] == 0]
        voisins.sort(key=lambda x: x[1])  # Trie par PIR croissante

        for voisin, pir_voisin, lat_voisin, lon_voisin in voisins:
            if total_pir + pir_voisin > MAX_PIR_PER_CLUSTER:
                continue
            # Tentatively add the neighbor
            temp_latlons = latlons + [(lat_voisin, lon_voisin)]
            centroid_lat = np.mean([lat for lat, lon in temp_latlons])
            centroid_lon = np.mean([lon for lat, lon in temp_latlons])
            # Calculate max distance from centroid to any point
            max_dist = 0
            for lat, lon in temp_latlons:
                # Haversine formula (in km)
                lat1, lon1, lat2, lon2 = map(np.radians, [centroid_lat, centroid_lon, lat, lon])
                dlon = lon2 - lon1
                dlat = lat2 - lat1
                a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                c = 2 * np.arcsin(np.sqrt(a))
                r = 6371
                dist = c * r
                if dist > max_dist:
                    max_dist = dist
            if max_dist <= RADIUS_KM:
                cluster_members.append(voisin)
                total_pir += pir_voisin
                latlons.append((lat_voisin, lon_voisin))
    else:
        cluster_members = [index]
        total_pir = pir_initial

    for i in cluster_members:
        cluster_map[i] = n
    n += 1
    processed += 1
    if processed % 5000 == 0 or processed == num_to_process:
        elapsed = time.time() - start_time
        percent_done = processed / num_to_process
        estimated_total = elapsed / percent_done
        remaining = estimated_total - elapsed
        print(f"Création des clusters: {percent_done*100:.1f}% - Clusters créés: {n-1} - Temps restant: {remaining:.1f}s")

donnees['cluster'] = cluster_map

# Export des résultats
donnees.to_csv("res.csv", sep=',', index=False)

print("\n\nTraitement terminé. Résultats enregistrés dans 'res.csv'.")

print("\n\n\n###################################################################")
print("#                       fin du clustering                         #")
print("################################################################### \n\n\n")
