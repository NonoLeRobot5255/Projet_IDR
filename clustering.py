import pandas as pd
import haversine as ha
import numpy as np
import time

start = time.time()
#on import le CSV et on le met sous forme de DB pour python
donnees = pd.read_csv("generated.csv", delimiter=",")

#on calcule les distances dans un tableau triangulaire supérieur
donnees['nb_voisins'] = 0
donnees['cluster'] = 0

a = ha.haversine(
    (donnees.loc[0, 'LAT'], donnees.loc[0, 'LON']),  # LAT d'abord !
    (donnees.loc[1, 'LAT'], donnees.loc[1, 'LON']),
)

traitement = 1000
for i in range (1,(traitement)):
    for j in range (i,traitement):

        #on calcule la distance entre les deux points
        dist = ha.haversine(
        (donnees.loc[i, 'LAT'], donnees.loc[i, 'LON']), 
        (donnees.loc[j, 'LAT'], donnees.loc[j, 'LON']),
        )
        
        if dist < 45:
            donnees.loc[i,'nb_voisins'] += 1

print(donnees)
print(time.time() - start)