# 📌Cheatsheet | Statistics


<details> <summary> <h2>Vocabulaire </h2></summary>

*   Stat **descriptives** (qui mesurent) vs probabilités = statistiques **inférentielles** (qui prédisent) : _descriptive sur le passé, inférentielle sur le futur)_ 
    * descriptive => moyenne, écart-type, ...
    *   En stat inférentielles, on utilise des tests satistiques = estimateurs = pour créer des modèles statistiques
    

*   lignes = individus = unité d'observation = réalisation
    
*   colonnes = variables = caractères
    *   variables quantitatives : 
        * discrètes vs continues
        * timestamp = nb de secondes depuis 1jan1970 [cf unix time](www.epochconverter.com).  Format ISO 8601 = `1977-04-22T06:00:00Z`
    
    *   variables qualitatives => modalités
        *   soit nominale (y/c booléen ) 
        - soit ordinale (grand, petit, etc)   
    * Noir = nominal (quali), ordinal (#quali), interval, ratio (all quanti)


* échantillon = jeu de données = dataset = observation
    - **echantillon <> population** 

* Midspread - Boxplot - boite à moustache : 
![image]https://user-images.githubusercontent.com/7408762/197854536-b36e92b2-3057-4bbe-a9d7-d12d7600148a.png    

</details>

<details> <summary> <h2> Nettoyer </h2></summary>

7 types d'erreurs :
1.  **Valeurs manquantes**
2.  **Erreur lexicale** (e.g. texte quand nombre attendu, ou liste limitative de pays possibles,,)
3.  **Irrégularité** (e.g. cm quand m attendu)
    
4.  formatage incorrect
5. formatage parfois lié à hypothèses de contenu ( e.g. 2 emails pour 1 personne))
6.  **doublon** (+ parfois **contradiction** si les doublons ont des valeurs différentes)
7.  valeur extrème = **atypique** (pas fausse) ou **aberrante** (fausse)
    
Comment résoudre les erreurs (Prévoir des aller-retours entre nettoyage et analyse) : 

<details> <summary> <h3> N.1. Valeurs manquantes : imputation </h3> </summary>

Bibliotheque spécialisée : `missingno` 
1.  Trouver la bonne valeur (à la main)
    
2.  Travailler avec un gruyère (données à trou, selon le traitement statistique)
3.  Oublier la variable
    
4.  Oublier les individus (mais les individus restants ne sont pas forcÃ©ment représentatifs)
    
5.  **Imputer** = deviner, e.g. imputation par la moyenne, ou imputer intelligemment, eg selon âge pour la taille, ou méthode de hot-deck, Machine Learning / KNN, régressions)

```(python)
myDF.isnull().sum() #somme par colonne le nb de manquant
data['nom_colonne'] = nouvelle_colonne
mask = # condition à vérifier pour cibler spécifiquement certaines lignes
data.loc[mask, 'ma_colonne'] = nouvelles_valeurs

data['taille'] = data['taille'].str[:-1] # supprimer le dernier caractere
data['taille'] = pd.to_numeric(data['taille'], errors='coerce')

data['Dept'].value_counts()
# ou
data['Dept'].unique()
```

</details>

<details> <summary> <h3> N.2.Eliminer les doublons... si on peut </h3> </summary>

* Identifier les doublons : pas de règles, à identifier en fonction du contexte.

*   Regrouper en gérant les contradictions
    * methodes `myDF.duplicated() myDF.duplicate() myDF.unique()`
    * contradiction : à ignorer, ou prendre la moyenne
    * parfois regroupement (information 1 individu répartie sur plusieurs lignes)

```(python)
data.loc[data.duplicated(keep=False),:]
```

</details>

<details> <summary> <h3> N.3.Traiter les outliers (= valeur aberrantes)  </h3> </summary>

-    trouvées par Z-score ou écart interquartile IQR (outliers are defined as mild above Q3 + 1.5 IQR and extreme above Q3 + 3 IQR.)
    * midspread ou Z-score, 
    * boite a moustache (boxplot)
-  Trouver la bonne valeur (à  la main)
-  Supprimer la valeur ou conserver la valeur ... en fonction des études (e.g. moyenne vs médiane)
-  ... les valeurs **atypiques** sont intéressantes, et à mentionner

![1024px-Boxplot_vs_PDF svg](https://user-images.githubusercontent.com/7408762/197854536-b36e92b2-3057-4bbe-a9d7-d12d7600148a.png)

</details>

<details><summary> <h3>N.4.Autres erreurs </h3></summary>

* On peut **supprimer** les individus avec erreur ... si ceux qui restent sont suffisants / non biaisés.

* Erreur lexicale = souvent pas de correction possible
* Irrégularité, formatage  = parfois correction à la main possible 

</details>    

* * *

<summary> <h2> Representer des variables </h2> </summary>

  

  

  

  

  

  

  
</details>
* * *

<details>
<summary> <h2> Composantes et clustering </h2> </summary>

Supervise => j'ai déja des tag d'apprentissage. On parle de **classement**\= classification supervisÃ©e (en EN = "classification").Â 

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
** Centrer-reduire, training split :** </summary>

```import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
```
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

* Training split*

(https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
`X_train, X_test, y_train, y_test = train_test_split(
...     X, y, test_size=0.33, random_state=42)`
Le 42 est un seed du random pour que ce soit toujours le même 

</details>

***

<details>  


<summary>
<h2>Modeles prédictifs:</h2> </summary>

<details>  
<summary>
<h3> M.1 Modeles predictifs linéaires = approximations supervisées </h3>
</summary>

- Si linéarité+normalité+indépendance (i.i.d.) => regression
    - recherche $β$ qui maximise la vraissemblance=  la probabilité de la distribution constatée ( $p(D|β)$ ) = minimise la somme des carrés des erreurs (MSE = RMSE)
        -  `LinearRegression` dans le module `linear_model`.
    
    <details> <summary>code</summary> 
            ```(python)
             ajouter ici code pandas
            ``` 

    </details>

    - $β=(X^⊤X)^{−1}X^⊤y$ 
    - ... et si $X^TX$ non inversible (notamment si colonnes corrélées), utiliser pseudo-inversible. Mais le modèle (la signification des $β_i$) est alors moins interprétable...
    - si correlation, ou trop peu d'observation, la matrice des $X^TX$ n'est pas inversible => Sur-apprentissage car modele trop complexe
        - => Alors on minimise une fonction objectif = erreur + complexité 
        = minimum en $β$ du carré des erreurs + λ.régularisateur(β) = $min_{β ∈ \mathbb{R}^{p+1}} (y−Xβ)^⊤(y−Xβ) + λ Regularisateur(β)$
        - où $λ$ = hyperparamètre du poids de la regularisation (cf validation croisée)
        - **régularisation de Tykhonov = ridge regression** pour diminuer le poids des coefs
            - regulateurs=carré de la norme de $β$ = norme $l2$
            - dans `scikit-learn : linear_model.Ridge` et `linear_model.RidgeCV` pour déterminer la valeur optimale du $λ$ par validation croisée.
            - => toujours solution unique explicite $β=(λI+X^⊤X)^{−1}X^⊤y$
            - mais il faut **toujours standardiser** les variables $X$ pour $σ=1$ avec `sklearn.preprocessing.StandardScaler`
            - chemin de régression : comment évoluent les $β_j$ avec $λ$, avec homogénéisation des coeff pour les variables corrélées entre elles
[image](cheminregression.png)
        - **LASSO = modele parcimonieux (_sparse_)** pour réduire nombre de coeff $β$ = en avoir bcp nuls = 0
            - on utilise regularisateur norme1 de $β$
            - LASSO = _Least Absolute Shrinkage and Selection Operator_
            - si plusieurs variables corrélées, le Lasso va en choisir une seule au hazard => modele instable, solution non unique
            - Lasso est un **algo de réduction de dimension non supervisé** 
        - **selection groupée = elastic net** 
            - consiste à combiner normes 1 et 2 sur $β$, avec cette fois 2 hyperparamètres $λ$ et $α$
            - $min_{β ∈ \mathbb{R}^{p+1}} (y−Xβ)^⊤(y−Xβ) + λ ((1-α)||β||_1 + α)||β||_2)$
            - => solution moins parcimonieuse, mais plus stable que LASSO

- Evaluer la performance d'une régression  
    - Avec ordre de grandeur : MSE et RMSE = mean squared error (mean of RSS = residual sum of squares = somme des carrés des résidus)
    - Sans ordre de grandeur : RMSLE et R^2
        - RMSLE = squared log error, si on veut une comparer sur des données à ordre de grandeur différents (erreur en % écart de la prédiction)
        - coef de détermination R^2 = 1- RSE (Relative Squared Error = erreur en % écart à la moyenne) = corrélation de Pearson entre valeurs vraies et prédites. See `sklearn.metrics.r2_score`


</details>
<details>
<summary> <h3> M.2 Modèles prédictifs linéaires pour classification </h3> </summary>

- regression logistique = pour classification binaire
    - classification binaire =  $y$ vaut 0 ou 1.
    - on on ne prédit plus les valeurs, mais la probabilité $p(y = 1|x)$ composée avec la fonction logistique $u\mapsto {1\over{1+e^{-u}}} $
    - Pas de solution exacte, calcul numérique par gradient
    - Pour éviter le sur-apprentissage, régularisation  ℓ2 (par défaut dans `scikit-learn`, 
    - Pour un modèle parcimonieux, régularisation ℓ1 (dans `scikit-learn`, option`'penalty'=l1`
- SVM binaire = support vector machine = separatrice a vaste marge
    - recherche d'un hyperplan séparateur maximisant la marge
    - risque d'erreur (observations impossibles à séparer par hyperplan, typiquement outliers). On utilise Hinge loss = perte charniere
    - 
- SVM multiclasse : regression multiple classes
    - one-versus-rest OVR = One-versus-all = OVA
        - on construit k SVM, en cherchant à optimiser
    - one-versus-one OVO
- evaluer la qualité d'une prédiction
        - [`sklearn.metrics.mean_squared_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html') pour calculer MSE ou RMSE entre la prédiction et la réalité. (R= root square)\
</details>

<details>
<summary> <h3> M.3 Modèles prédictifs non linéaires </h3> </summary>
- "Kernel trick" : transformer les x d'input des 
![image](https://user-images.githubusercontent.com/7408762/197527731-29e2ad2b-2a1e-48a7-b92c-26df55445280.png)

- Neural networks : fonction d'activation sur entrées. "Perceptron"
    - Le Perceptron = "neurone" : 
        - Combi linéaire des entrées x activation
        - poids appris par descende de gradient
    - Empiler les perceptrons : 
        - poids sur chaque perceptron
        - à entrainer avec EN back-propagation (FR rétro-propagation) : $derreur/dw_hji= d/d * d/d * d/d$

    - Pour approximation
        - technique de descente du gradient
        - entropie croisée 
    - Pour classification
        - possible d'utiliser activation à seuil
        - mieux : utiliser sigmoide (typiquement : activation logistique) pour probabilité d'appartenance à une classe 
    - Limitation : les réseaux de neurones ne sont pas la solution à tous les problèmes car...
        
</details>


<details>
<summary> <h3> M.4 Modèles ensemblistes </h3> </summary>

- Gist = combine several models together
    - "Bootstrap" first idea = sampling with remise échantillonage avec remise
    - Méthodes parallèles: train several models simultaneously, recombine them at the end
    - utilisant des "apprenants faibles" : des méthodes simples et peu efficaces, qui en se combinant donnent de meilleurs résultat que les méthodes complexes
    - Méthodes séquentielles : **boosting**

<details>
<summary>
- **Bagging** = Bootstrap aggregation </summary>
    - Moyenne pour prédiction, vote majoritaire pour classification
    - 
```(python)
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, noise=0.25)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

from sklearn.ensemble import BaggingClassifier 

bagging = BaggingClassifier(n_estimators=5)
bagging.fit(X_train, y_train)
from mglearn.plot_interactive_tree import plot_tree_partition
from mglearn.plot_2d_separator import plot_2d_separator
from mglearn.tools import discrete_scatter

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
for i, (ax, tree) in enumerate(zip(axes.ravel(), bagging.estimators_)):
    ax.set_title("Tree {}".format(i))
    plot_tree_partition(X_train, y_train, tree, ax=ax)
plot_2d_separator(bagging, X_train, fill=True, ax=axes[-1, -1],
                                    alpha=.4)
axes[-1, -1].set_title("Bagging")
discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
```

</details>

<details>
<summary>
- **Random Forest** = arbres de décisions binaires combinés  la majorité de vote
</summary>
    - Pb: les arbres de décision ont tendance à overfitter. 
    - Pour faire grandir chaque noeud, on n'utilise qu'un sous-ensemble de features (et pas toutes comme le bagging).
        - sous ensemble choisi de manière aléatoire : arbres aléatoires
    - Avantage  : complexité peu élevés, on a estimation de l'importance des features. Pas d'overfitting, peu de mémoire utilisée. 
 ```(python)   
import pandas as pd

train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")
print(train.shape)
train.isna().sum()
train = train.loc[train.Activity.notna()]
train = train.fillna(train.median(), inplace=True)

X_train = train[train.columns[:-2]]
y_train = train['Activity']

X_test = test[test.columns[:-2]]
y_test = test['Activity']

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=500, oob_score=True)
model = rfc.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

pred = rfc.predict(X_test)
print("accuracy {:.2f}".format(accuracy_score(y_test, pred)))
from sklearn.feature_selection import SelectFromModel
select = SelectFromModel(rfc, prefit=True, threshold=0.003)
X_train2 = select.transform(X_train)
print(X_train2.shape)
import timeit

rfc2 = RandomForestClassifier(n_estimators=500, oob_score=True)

start_time = timeit.default_timer()

rfc2 = rfc2.fit(X_train2, y_train)

X_test2 = select.transform(X_test)

pred = rfc2.predict(X_test2)
elapsed = timeit.default_timer() - start_time
accuracy = accuracy_score(y_test, pred)

print("accuracy {:.2f} time {:.2f}s".format(accuracy, elapsed))

```

</details>

<details>
<summary> Boosting & Gradient Boosting </summary>

- Le Boosting, dont adaboost
    - on pondere chacun des points à chaque generation

- Gradient de Boosting : 
    - jhkjhj

</details>

</details>

</details>

<details>
<summary> Sources </summary>
---
[Cheatsheet Anthony : https://asardell.github.io/statistique-python/](https://asardell.github.io/statistique-python/)

[Meme contenu copié sur evernote](evernote:///view/6367254/s57/f1dae14f-b0c0-4024-a6f5-7b2535f53308/67117fc9-036c-4028-b61e-04a2b3349d73/)
</details
>
---

***


chgt 02nov22 2021

*** 
