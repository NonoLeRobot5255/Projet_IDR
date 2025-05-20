import matplotlib.pyplot as plt
import pandas as pd

donnees = pd.read_csv("res.csv")

plt.figure(figsize=(10, 8))
plt.scatter(donnees['LON'], donnees['LAT'], c=donnees['cluster'], cmap='tab20', s=2)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Clusters DBSCAN (vue statique)")
plt.tight_layout()
plt.savefig("clusters_dbscan.png", dpi=300)
plt.show()
