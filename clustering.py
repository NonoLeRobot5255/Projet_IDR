import pandas as pd
import haversine as ha
import numpy as np


#on import le CSV et on le met sous forme de DB pour python
donnees = pd.read_csv("generated.csv", delimiter=",")


donnees['nb_voisins'] = 0
donnees['cluster'] = 0
#donnees['VID'] = []

a = ha.haversine(
    (donnees.loc[0, 'LAT'], donnees.loc[0, 'LON']),  # LAT d'abord !
    (donnees.loc[1, 'LAT'], donnees.loc[1, 'LON']),
)

traitement = 100
for i in range (1,(traitement)):
    for j in range (i,traitement):

        #on calcule la distance entre les deux points
        dist = ha.haversine(
        (donnees.loc[i, 'LAT'], donnees.loc[i, 'LON']), 
        (donnees.loc[j, 'LAT'], donnees.loc[j, 'LON']),
        )
        
        if dist < 45:
            donnees.loc[i,'nb_voisins'] += 1
            #donnees.loc[i,'VID'].add(j)
            donnees.loc[j,'nb_voisins'] +=1
            #donnees.loc[j,'VID'].add(i)


donnees = donnees.sort_values(by=['nb_voisins'], ascending=False)
#for loop in range(len(donnees )-1):
#    donnees=pd.DataFrame
#    if 0 in donnees['cluster'].values:

    

                
print(donnees) 