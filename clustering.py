import pandas as pd
import haversine as ha
import numpy as np

#on import le CSV et on le met sous forme de DB pandas
donnees = pd.read_csv("generated.csv", delimiter=",")

#on rajoute des colonnes pour nos 3 données suplémentaires
donnees['nb_voisins'] = 0
donnees['cluster'] = 0
donnees['VID'] = [[] for d in range(len(donnees))]
#le nombre de données qu'on traite
traitement = 1000

#on calcul ici toutes les distances et les potentiels voisins
for i in range (traitement):
    for j in range (i+1,traitement):

        #on calcule la distance entre les deux points
        dist = ha.haversine(
        (donnees.loc[i, 'LAT'], donnees.loc[i, 'LON']), 
        (donnees.loc[j, 'LAT'], donnees.loc[j, 'LON']),
        )
        
        #si on est dans le cercle de 90km de diamètre, on ews théoriquement tous les deux dans un cluster
        if dist < 45:
            donnees.loc[i,'nb_voisins'] += 1
            donnees.loc[i,'VID'].append(j)
            donnees.loc[j,'nb_voisins'] +=1
            donnees.loc[j,'VID'].append(i)




#numéro du premier cluster
n = 1

#on trie par nombres de voisins potentiels 
sorted_indices = donnees['nb_voisins'].sort_values(ascending=False).index


#on assigne ici les clusters
for index in sorted_indices:
    if donnees.at[index, 'cluster'] == 0:
        donnees.at[index, 'cluster'] = n
        voisins = donnees.at[index, 'VID']
        for i in voisins:
            donnees.at[i, 'cluster'] = n
        n += 1
        
donnees = donnees.sort_values(by=['nb_voisins'], ascending=False)

#on affiche les données
print(donnees) 

#on affiche tout dans un autre csv nommé "res.csv"
donnees.to_csv("res.csv", sep=',')