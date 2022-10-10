# 📌Cheatsheet | Statistics

---
[Cheatsheet Anthony : https://asardell.github.io/statistique-python/](https://asardell.github.io/statistique-python/)

[Meme contenu copié sur evernote](evernote:///view/6367254/s57/f1dae14f-b0c0-4024-a6f5-7b2535f53308/67117fc9-036c-4028-b61e-04a2b3349d73/)

---

<details>
<summary> <h2>Vocabulaire </h2></summary>

*   Stat descriptives (qui mesure) vs probabilités = statistiques inférentielles (qui prédit) : _descriptive sur le passé, inférentielle sur le futur)_ 
    

*   En stat inférentiel, on utilise des tests satistiques = estimateurs, on crée des modèles statistiques
    

*   lignes = individus = unité d'observation = réalisation
    
*   colonnes = variables = caractères

* population vs échantillon = jeu de données = dataset = observation
    
*   variables quantitatives : 
    * discrètes vs continues
    
*   variables qualitatives => modalités
    *   soit nominale, soit ordinale    

* Noir = nominal (quali), ordinal (#quali), interval, ratio (all quanti)

    
*   ordinale ( e.g. dates, timestamp). [Conversion](www.epochconverter.com)
    
*   boolÃ©enne
    
</details>


<details>

<summary> <h2>Nettoyage </h2></summary>

*   Prévoir Aller retour nettoyage et analyse
    
*   Valeurs manquantes 

```myDF.isnull().sum() #somme par colonne le nb de manquant``` 
    

1.  Trouver la bonne valeur (Ã  la main)
    
2.  Travailler avec un gruyÃ¨re
    
3.  Oublier la variable
    
4.  Oublier les individus (mais les individus restants ne sont pas forcÃ©ment reprÃ©sentatifs)
    
5.  Imputer (= deviner, e.g. imputation par la moyenne, ou imputer intelligemment, eg selon Ã¢ge pour la taille)
    

*   Traiter les outliers (= valeur aberrantes)
    
    trouvÃ©es par Z-score ou Ã©cart interquartile
    

1.  Trouver la bonne valeur (Ã  la main)
    
2.  Supprimer la valeur ou conserver la valeur ... en fonction des Ã©tudes (e.g. moyenne vs mÃ©diane)
    
3.  ... les valeurs atypiques sont intÃ©ressantes, et Ã  mentionner
    

*   Eliminer les doublons... si on peut
    

*   Regrouper en gérant les contradictions
    

*     
    

  

MÃ©thodeÂ :

*   MÃ©thodeÂ :
    

*   AllerÂ retour nettoyage et analyse
    
*   ValeursÂ manquantes :
    

*   TrouverÂ la bonne valeur (Ã  la main)
    
*   TravaillerÂ avec un gruyÃ¨re
    
*   OublierÂ la variable
    
*   OublierÂ les individus (mais les individus restants ne sont pas forcÃ©ment reprÃ©sentatifs)
    
*   ImputerÂ (= deviner, e.g. imputation par la moyenne, ou imputer intelligemment, eg selon Ã¢ge pour la taille)
    

*   TraiterÂ les outliers (= valeur aberrantes)
    
*   trouvÃ©esÂ par Z-score ou Ã©cart interquartile
    

*   TrouverÂ la bonne valeur (Ã  la main)
    
*   SupprimerÂ la valeur ou conserver la valeur ... en fonction des Ã©tudes (e.g. moyenne vs mÃ©diane)
    
*   ...Â les valeurs atypiques sont intÃ©ressantes, et Ã  mentionner
    

*   EliminerÂ les doublons... si on peut
    

*   RegrouperÂ en gÃ©rant les contradictions
    

MÃ©thodeÂ :

*   AllerÂ retour nettoyage et analyse
    
*   ValeursÂ manquantes :
    

*   TrouverÂ la bonne valeur (Ã  la main)
    
*   TravaillerÂ avec un gruyÃ¨re
    
*   OublierÂ la variable
    
*   OublierÂ les individus (mais les individus restants ne sont pas forcÃ©ment reprÃ©sentatifs)
    
*   ImputerÂ (= deviner, e.g. imputation par la moyenne, ou imputer intelligemment, eg selon Ã¢ge pour la taille)
    

*   TraiterÂ les outliers (= valeur aberrantes)
    
*   trouvÃ©esÂ par Z-score ou Ã©cart interquartile
    

*   TrouverÂ la bonne valeur (Ã  la main)
    
*   SupprimerÂ la valeur ou conserver la valeur ... en fonction des Ã©tudes (e.g. moyenne vs mÃ©diane)
    
*   ...Â les valeurs atypiques sont intÃ©ressantes, et Ã  mentionner
    

*   EliminerÂ les doublons... si on peut
    

*   RegrouperÂ en gÃ©rant les contradictions
    
</details>

<details>
<summary> <h2>Erreurs et imputations </h2></summary>

7 types d'erreurs :
1.  Valeurs manquantes
2.  **Erreur lexicale** (e.g. texte quand nombre attendu, ou liste limitative de pays possibles,,)
3.  **Irrégularité** (e.g. cm quand m attendu)
    
4.  formatage incorrect
5. formatage parfois lié à hypothèses de contenu ( e.g. 2 emails pour 1 personne))
6.  **doublon** (+ parfois **contradiction** si les doublons ont des valeurs différentes)
7.  valeur extrème = **atypique** (pas fausse) ou **aberrante** (fausse)
    
Comment résoudre les erreurs 

0. On peut **suprrimer** les individus avec erreur ... si ceux qui restent sont suffisants / non biaisés.

1. Valeur manquante = 
    * **imputation** e.g. imputation par la moyenne (simple) -> méthode de hot-deck, Machine Learning / KNN, régressions
    * ou travailler avec un gruyère (données à trou, selon le traitement statistique)

6.  Doublon 
    * methodes `myDF.duplicated() myDF.duplicate() myDF.unique()`
    * contradiction : à ignorer, ou prendre la moyenne
    * parfois regroupement (information 1 individu répartie sur plusieurs lignes)

7.  Valeur extrème  = 
    * choix des traitements, e.g. la moyenne est sensible aux outliers, pas la médiane 
    * midspread ou Z-score, 
    * boite a moustache (boxplot)

</details>    

* * *
<details>
<summary> <h2> ReprÃ©senter des variables </h2> </summary>

  

  

  

  

  

  

  
</details>
* * *

<details>
<summary> <h2> ReprÃ©senter des variables </h2> </summary>

SupervisÃ© => j'ai dÃ©ja des tag d'apprentissage. On parle de **classement**\= classification supervisÃ©e (en EN = "classification").Â 

Non supervisÃ© =Â  **clusteringÂ** 

![](ðŸ“ŒCheatsheet  Statistics_files/Image.png)

  

D

* * *

Distance (erreur = risque = eloignement des donnÃ©es vs prediction modele)

Attention : erreur = risque empirique != performance du modele

*   erreur quadratique (le + utilisÃ©)
    

*   distance euclidienne = sqr(x^2 + y^2)
    

*   Distance manhattan = x + y
    
*   Pour chaines de caracteres = distance de Levenshtein = nbre mini d'operation (substitution, insertion, suppression) pour passer de l'une a l'autre.Â 
    

*   a connaitre = algo de Wagner et Fischer pour le calcul de la distance de Levenshtein.
    

algo paramÃ©triques (eg regression = droite) => on cherche le parametreÂ Î¸ (qui peut etre multidimensionel)

algos non parametriques (+ complexitÃ©) => egg k-means qui est 'memory based' (garde toutes les donnÃ©es en memoire)

  

  

  

fuction loss = perte d'information

vraisemblance d'un jeu d'observations (x1...xN) par rapport Ã  un modÃ¨le en statistiques est la fonction suivante :Â Â L(Î¸)=p(x1...xN|Î¸)Â Â .= proba d'avoir x1...xN sachant \\Theta

Â Î¸^Â avec un accent circonflexeÂ lorsqu'on parle d'unÂ estimateur (eet non de la valeur reelle, intrinseque)

  

* * *

1.  MÃ©thode factorielle = la + connue ACP
    
2.  Clulstering = Classification non supervisÃ©e = la + connue k-means (K-moyennes)
    

  

Factorielle :Â 

ACPÂ  ( = EN PCA) = Principal component analysis

*   Â  Â  recehche d'un (hyperplan) avec moment d'inertie max (Ã©talement des points) = axe orthogonal Ã  l'hyperplan = donne indication sur la variabilitÃ© =
    

*   espace Rp de dimension p variables, contient Ni le nuage des individus
    

*   Rechreche des corrÃ©lations entre variablesÂ 
    

*   espace Rn de dimension n individus, contient Np le nuage des variables
    

De prÃ©fÃ©rence ACP normÃ©e (centrÃ©e rÃ©duite)

3 graphiques :Â 

1.  1\. Pour l'objectif 1, ce sera la projection du nuage des individus NI sur les 2 premiers axes dâ€™inertie, câ€™est-Ã -dire sur le premier plan factoriel.
    
2.  Le second sâ€™appelle le cercle des corrÃ©lations.
    
3.  2\. Pour l'objectif 2, ce sera la projection du nuage des variables NK sur le premier plan factoriel.
    

  

combien de composantes = min (p nbr de varialbes et n-1 nombre individus)

\=> eboulis des valeurs propres (classÃ©es en valeur dÃ©croissante)

\=> frequent de n'analyser que le 1er plan (2 composantes). Critere du coude - reperer le # oÃ¹ le % inertie diminue + lentement. Criter de Kaiser (~contribution moyenen 100% / p)

  

k-meansÂ 

k est un **hyperparamÃ¨tre** (c'est Ã  nous de l'optimiser, ce n'est pas l'algo qui va le proposer).Â 

  

  

Trainig set vs testing set = 80% / 20% des donnÃ©es fournies

  

  

* * *

Conversion de timestamp unix =Â Â [www.epochconverter.com](http://www.epochconverter.com/)Â !

  

Erreur lexicale => Technique du dictionnaire.

Date => Format normalisÃ© ISO8601Â 1977-04-22T06:00:00Z.

  

  

</details> 

* * *

<details>  
<summary>
**Comment centrer reduire :** </summary>

import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler

DÃ©finissons nos donnÃ©es :

  

\# Notre matrice de base :

X = \[\[12,Â  Â  30,Â  Â  80,Â  -100\],Â  Â  \[-1000, 12,Â  Â  -23,Â  10\],Â  Â  \[14,Â  Â  1000,Â  0,Â  Â  0\]\]

  

\# Version numpy :

X = np.asarray(X)

\# Version pandas :

X = pd.DataFrame(X)

Avec Â pandasÂ  , on peut calculer la moyenne et l'Ã©cart-type de chaque dimensionÂ :

  

\# On applique la methode .describe() pour avoir la moyenne et la .std(), et la mÃ©thode .round(2) pour arrondir Ã  2 dÃ©cimales aprÃ¨s la virgule :

X.describe()

On peut ensuite Â«Â scalerÂ Â» nos donnÃ©es :

  

\# On instancie notre scaler :

scaler = StandardScaler()

\# On le fit :

scaler.fit(X)

\# On l'entraine :

X\_scaled = scaler.transform(X)

\# On peut faire les 2 opÃ©rations en une ligne :

X\_scaled = scaler.fit\_transform(X)

\# On le transforme en DataFrame :

X\_scaled = pd.DataFrame(X\_scaled)

\# On peut appliquer la mÃ©thode .describe() et .round()

X\_scaled.describe().round(2)

</details>

***

<!--details-->  


<summary>
**Modeles prédictifs:** </summary>
  
- modeles predictifs linéaires
    - Si linéarité+normalité+indépendance (i.i.d.) 
        - => regression, recherche $β$ qui maximise la vraissemblance ($p(D|β)$) = minimise la somme des carrés des erreurs
        -  `LinearRegression` dans le module `linear_model`.
        - $β=(X^⊤X)^{−1}X^⊤y$ 
        - ... et si $X^TX$ non inversible, utiliser pseudo-inversible
    - si correlation, ou trop peu d'observation, la matrice des XtX n'est pas inversible => Sur-apprentissage car modele trop complexe
        - => Alors on minimise une fonction objectif = erreur + complexité - minimum en $β$ du carré des erreurs + λ.régularisateur(β) 
        - où $λ$ hyperparamètre du poids de la regularisation (cf validation croisée)
        - **régularisation de Tykhonov = ridge regression** où regulateurs=carré de la norme de $β$ 
            - dans `scikit-learn : linear_model.Ridge` et `linear_model.RidgeCV` pour déterminer la valeur optimale du $λ$ par validation croisée.
            - => toujours solution unique explicite $β=(λI+X^⊤X)^{−1}X^⊤y$
            - mais il faut toujours standardiser les variables $X$ pour $σ=1$ avec `sklearn.preprocessing.StandardScaler`
            - chemin de régression : comment évoluent les $β_j$ avec $λ$
[image](cheminregression.png)
        - **Lasso = modele parcimonieux (_sparse_)** pour réduire nombre de coeff $β$ = en avoir bcp nuls = 0
            - on utilise regularisateur norme1 de $β$
            - lasso = _Least Absolute Shrinkage and Selection Operator_
            - si plusieurs variables corrélées, le Lasso va en choisir une seule au hazard => modele instable, solution non unique
            - Lasso est un algo de réduction de dimension non supervisé
        - **selection groupée = elastic net** 
            - consiste à combiner normes 1 et 2 sur beta, avec cette fois 2 hyperparamètres 
            - => solution moins parcimonnieuse, mais plus stable que Lasso
    - regression logistique = pour classification binaire
        - classification binaire
        - SVM = support vector machine = separatrice a vaste marge
        - Hinge loss = perte charniere
        - 
    - regression multple classes
        - one-versus-rest OVR = One-versus-all = OVA



  

</details-->

* * *
