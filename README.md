# Projet_IDR

## algo 1 (BallTree)

Pour les différentes limites de PIR on a :

| Limite du PIR | Nombre de clusters |
| :-----------: | :----------------: |
| 1Gb/s         | 25 365             |
| 2Gb/s         | 20 059             |
| 4Gb/s         | 18 781             |

Pour un temps d'exécution moyen de 3.3s sur 10 exécutions.

### 

On a une complexité pour chaque aspect de :

| étape  | complexité |
| :-----------: | :----------------: |
| chargement des données         | $$ O(n)$$           |
| Ball Tree         | $$ O(n\ log(n))$$             |
| reecherche de voisins        | $$ O(m\ n\ log(n))$$             |
| assignemetn des clusters        | $$ O( n\ log(n) + n \ k)$$             |

Pour une complexité globale en$$O(n\ log(n) + n \ k \ log(k))$$
