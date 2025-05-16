import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import time

# Paramètres
RADIUS_KM = 45
MAX_PIR_PER_CLUSTER = 1000.0
EARTH_RADIUS_KM = 6371.0

# Chargement des données
print("Lecture du fichier CSV...")
dtype_dict = {
    'LAT': np.float32,
    'LON': np.float32,
    'PIR': np.float32
}
donnees = pd.read_csv("generated.csv", delimiter=",", dtype=dtype_dict)
print(f"Chargement terminé: {len(donnees)} lignes")

# Conversion en radians
coords_rad = np.radians(donnees[['LAT', 'LON']].values)

# DBSCAN avec haversine
eps_rad = RADIUS_KM / EARTH_RADIUS_KM
print("Exécution de DBSCAN...")
start_time = time.time()
db = DBSCAN(eps=eps_rad, min_samples=1, metric='haversine', algorithm='ball_tree').fit(coords_rad)
elapsed = time.time() - start_time
print(f"Clustering géographique terminé en {elapsed:.2f}s")

donnees['dbscan_cluster'] = db.labels_

# Cluster final à affecter
final_cluster_id = 1
final_clusters = np.zeros(len(donnees), dtype=np.int32)

print("Découpage des clusters DBSCAN avec contrainte sur PIR...")

for db_id in np.unique(db.labels_):
    indices = np.where(db.labels_ == db_id)[0]
    
    # Trier les points du cluster DBSCAN par PIR croissante
    sous_df = donnees.iloc[indices]
    sous_df_sorted = sous_df.sort_values(by='PIR')
    
    current_pir = 0.0
    current_members = []

    for idx, row in sous_df_sorted.iterrows():
        pir = row['PIR']
        
        # Si ajouter ce point dépasse la limite, créer un nouveau cluster
        if current_pir + pir > MAX_PIR_PER_CLUSTER:
            # Attribuer le cluster courant
            for member in current_members:
                final_clusters[member] = final_cluster_id
            final_cluster_id += 1
            # Réinitialiser
            current_members = []
            current_pir = 0.0

        current_members.append(idx)
        current_pir += pir

    # Attribuer le dernier groupe s’il reste des points
    if current_members:
        for member in current_members:
            final_clusters[member] = final_cluster_id
        final_cluster_id += 1

print(f"Nombre total de clusters finaux créés : {final_cluster_id - 1}")
donnees['cluster'] = final_clusters

# Export
donnees.to_csv("res_dbscan_pir.csv", sep=',', index=False)
print("Résultats exportés dans 'res_dbscan_pir.csv'")
