# 📌Cheatsheet | Statistics DataScience


<details><summary><h2>Vocabulaire</h2></summary>

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
    - distribution empirique =constatée.

* Midspread - Boxplot - boite à moustache : 
    - Rappel $±1.σ = 68\%$ ; $±2.σ = 95\%$ ; $±3.σ = 99.7\% $
![image](https://user-images.githubusercontent.com/7408762/197854536-b36e92b2-3057-4bbe-a9d7-d12d7600148a.png)

* **Machine learning** = modélisation statistique à partir des données
    * deep learning = apprentissage direct à partir des données brutes
    * les 5 classiques à connaitre= **regression, Knn, SVM, réseaux de neurones, random forests**. 
        * supervisé = j'ai un jeu d'apprentissage, sur lequel je connais le résultat attendu = on espère prédire
        * non supervisé = l'algorithme doit trouver tout seul les similarités = on espère découvrir des relations latentes. Aussi ∃ semi-supervised = une partie des données sont annotées
        * reinforcement learning = non supervisé, mais récompense améliore les opérations (typique : jeu d'échec)
    * 3 sorties : quanti, quali, ordre (regression, classification, ranking)
    * Librairies python : scikit-learn, tensorflow, torch, theano, caffe. 
    * Généralisation = prédictions sur nouvelles données (test ou autre)


</details>

***

<h2> Using the tools : JupyterNB, Colab </h2>

<details><summary><h2> Colab </h2></summary>

### [dedicated cheatsheet on Colab](https://colab.research.google.com/drive/13IO3-gfyS9mSPuzAo6-wsYBUOVpxb_va?usp=sharing)
- **help** on all shortcuts => Ctrl  M  H
- create a new cell below/ above 'ctrl M B'  / `ctrl M A`
- convert a text cell to code / to text ctrl M  Y / ctrl M M
- run current cell 'ctrl + enter'
- delete a cell ctrl+m d

</details>

<details><summary><h2> Nettoyer </h2></summary>

7 types d'erreurs :
1.  **Valeurs manquantes**
2.  **Erreur lexicale** (e.g. texte quand nombre attendu, ou liste limitative de pays possibles,,)
3.  **Irrégularité** (e.g. cm quand m attendu)
    
4.  formatage incorrect
5. formatage parfois lié à hypothèses de contenu ( e.g. 2 emails pour 1 personne))
6.  **doublon** (+ parfois **contradiction** si les doublons ont des valeurs différentes)
7.  valeur extrème = **atypique** (pas fausse) ou **aberrante** (fausse)
    
Comment résoudre les erreurs (Prévoir des aller-retours entre nettoyage et analyse) : 

<details> <summary> <h3> N.1. Imputer les valeurs manquantes </h3> </summary>

Bibliotheque spécialisée : `missingno` 
1.  Trouver la bonne valeur (à la main)
    
2.  Travailler avec un gruyère (données à trou, selon le traitement statistique)
3.  Oublier la variable (si trop de trous)
    
4.  Amputer = oublier les individus (risque: les individus restants ne sont pas forcément représentatifs)
    
5.  **Imputer** = deviner, e.g. imputation par la moyenne, ou imputer intelligemment, eg selon âge pour la taille, ou méthode de hot-deck, Machine Learning / KNN, régressions)

```python
myDF.isnull().sum() #somme par colonne le nb de manquant

data.loc[data['taille'].isnull(), 'taille'] = data['taille'].mean()
```

</details>

<details> <summary> <h3> N.2.Eliminer les doublons... si on peut </h3> </summary>

* Identifier les doublons : pas de règles, à identifier en fonction du contexte.

*   Regrouper en gérant les contradictions
    * methodes `myDF.duplicated() myDF.duplicate() myDF.unique()`
    * contradiction : à ignorer, ou prendre la moyenne
    * parfois regroupement (information 1 individu répartie sur plusieurs lignes)

```python
data.loc[data.duplicated(keep=False),:] # duplicated returns Booleans


data['Dept'].value_counts()
# ou
data['Dept'].unique()
```

</details>

<details><summary><h3> N.3.Traiter les outliers (= valeur aberrantes)  </h3> </summary>

- Trouvées par Z-score ou écart interquartile IQR (outliers are defined as mild above Q3 + 1.5 IQR and extreme above Q3 + 3 IQR.)
    * midspread ou Z-score :  $z = (x – μ)/σ$ 
    * boite a moustache (boxplot)
- Trouver la bonne valeur (à  la main)
- Supprimer la valeur ou conserver la valeur ... en fonction des études (e.g. moyenne vs médiane)
-  ... les valeurs **atypiques** sont intéressantes, et à mentionner

![1024px-Boxplot_vs_PDF svg](https://user-images.githubusercontent.com/7408762/197854536-b36e92b2-3057-4bbe-a9d7-d12d7600148a.png)

</details>

<details><summary> <h3>N.4.Autres erreurs </h3></summary>

* On peut **supprimer** les individus avec erreur ... si ceux qui restent sont suffisants / non biaisés.

* Erreur lexicale = souvent pas de correction possible
* Irrégularité, formatage  = parfois correction à la main possible 

```python
data['nom_colonne'] = nouvelle_colonne
mask = # condition à vérifier pour cibler spécifiquement certaines lignes
data.loc[mask, 'ma_colonne'] = nouvelles_valeurs
VALID_COUNTRIES = ['France', 'Côte d\'ivoire', 'Madagascar', 'Bénin', 'Allemagne'
, 'USA']
mask = ~data['pays'].isin(VALID_COUNTRIES)
data.loc[mask, 'pays'] = np.NaN

data['email'] = data['email'].str.split(',', n=1, expand=True)[0]

data['taille'] = data['taille'].str[:-1] # supprimer le dernier caractere
data['taille'] = pd.to_numeric(data['taille'], errors='coerce')
```

</details>    

</details>

* * *

<details><summary> <h2>Représenter une variable par analyse monovariée</h2></summary>


<details><summary> <h3> R.1.Variables qualitatives :  </h3> </summary>


```python
# PIE CHART Diagramme en secteurs
data["categ"].value_counts(normalize=True).plot(kind='pie')
# Cette ligne assure que le pie chart est un cercle plutôt qu'une éllipse
plt.axis('equal')
plt.show() # Affiche le graphique

# BAR CHART Diagramme en tuyaux d'orgues
data["categ"].value_counts(normalize=True).plot(kind='bar')
plt.show()

# TABLEAU avec effectifs, freq, freq cumulée
effectifs = myData["Modalité"].value_counts()
modalites = effectifs.index # l'index de effectifs (= de myData) contient les modalités
myTable = pd.DataFrame(modalites, columns = ["Modalité"]) # création du tableau à partir des modalités
myTable["effectif n"] = effectifs.values
myTable["frequency f"] = myTable["effectif n"]] / len(myData) # Rappel len(myDataFrame) renvoie la taille de l'échantillon = le nb de lignes
myTable = myTable.sort_values("Modalité") # tri des valeurs de la variable X (croissant)
myTable["Cumulated Freq F"] = myTable["frequency f"].cumsum() # cumsum calcule la somme cumulée
``` 

</details>

<details><summary> <h3> R.2.Variables quantitatives et moments  </h3> </summary>

```python
# BAR CHART pour var discretes # Diagramme en bâtons
data['quart_mois'] = [int((jour-1)*4/31)+1 for jour in data["date_operation"].dt.day]
data["quart_mois"].value_counts(normalize=True).plot(kind='bar',width=0.1)
plt.show()

# BAR CHART pour var continues = Histogramme
data["montant"].hist(density=True)
plt.show()
# Histogramme plus beau
data[data.montant.abs() < 100]["montant"].hist(density=True,bins=20)
plt.show()

#Fonction de répartition empirique (= histogramme cumulé)

# Mesures de tendance centrale : 
data['montant'].mode() #renvoie un Series, car il peut y avoir plusieurs modes
data['montant'].mean()
data['montant'].median()

# Moments d'ordre 2
data['montant'].var() # variance empirique (avec biais)
data['montant'].var(ddof=0) #variance empirique (sans biais)
data['montant'].std()   # s ecart type 
data['montant'].std/data['montant'].mean() # coeff variation

data.boxplot(column="montant", vert=False) 
plt.show()

#Moments d'ordre 3 et 4
data['montant'].skew()
data['montant'].kurtosis()

```
**Règle de Sturges** = le nb de classes optimales est $(1+log2(n))$

**Variance empirique** = second moment = 
$v = \frac{1}{n} \sum_{i=1}^{n}(x_i-\overline{x})^2 = s^2$

**Variance empirique sans biais** = 
$s'^2 = \frac{1}{n-1} \sum_{i=1}^{n}(x_i-\overline{x})^2$
(pour un grand échantillon, pas de différence avec/sans biais)

**Coeff de variation** = $CV = \frac{s}{overline{x}}$ où $s$ l'écart type empirique ($\sigma$ écart type population) 

**Ecart Moyen Absolu** = variance avec norme 1 = $EMA = \frac{1}{n}\sum_{i=1}^{n}{|x_i - \overline{x}|}$ 

EMA peut aussi se calculer par écart à la médiane. 

**Skewness =assymétrie** = third standardised moment = $\tilde{\mu}_3 = \frac{\mu_3}{s^3} = \frac{1}{s^3}\frac{1}{n}\sum_{i=1}^{n}(x_i-\overline{x})^3$

Skweness > 0 = positive skew = long right tail = generally, mean>median

**Kurtosis =aplatissement** = fourth standardised moment = $\tilde{\mu}_4 = \frac{\mu_4}{s^4} = \frac{1}{s^4}\frac{1}{n}\sum_{i=1}^{n}(x_i-\overline{x})^4$

Kurtosis > 0 = positive curtosis = pointier than gaussian curve

(Note : first and second standardised moment are always 0 and 1)

</details>

<details><summary> <h3> R.3.Variables quantitatives et concentration  </h3> </summary>

Les 3 approches (avec l'exemple de la concentration des richesses): 
- Courbe de Lorenz = pauvres à gauche, riches à droite = escaliers de hauteur total 1=100%
    - la personne à 50% de l'axe horizontal a le salaire médian.
    - le salaire médial est est celui de la personne correspondant à la hauteur 50%.
- Indice de Bini = calculé sur la courbe de Lorenz 
    - Igini = 2 x aire entre Lorenz et la première bissectrice
    - Gini = 0 => égalité parfaite
    - Gini = 1 => inégalité parfaite (1 personne cumule tout)
- Pareto : "les X% les + riches possèdent Y% de la richesses"    

```python
depenses = data[data['montant'] < 0] dep = -depenses['montant'].values n = len(dep) lorenz = np.cumsum(np.sort(dep)) / dep.sum() lorenz = np.append([0],lorenz) # La courbe de Lorenz commence à 0 
xaxis = np.linspace(0-1/n,1+1/n,n+1) #Il y a un segment de taille n pour chaque individu, plus 1 segment supplémentaire d'ordonnée 0. Le premier segment commence à 0-1/n, et le dernier termine à 1+1/n. 
plt.plot(xaxis,lorenz,drawstyle='steps-post') 
plt.show()

AUC = (lorenz.sum() -lorenz[-1]/2 -lorenz[0]/2)/n # Surface sous la courbe de Lorenz. Le premier segment (lorenz[0]) est à moitié en dessous de 0, on le coupe donc en 2, on fait de même pour le dernier segment lorenz[-1] qui est à moitié au dessus de 1. 
S = 0.5 - AUC # surface entre la première bissectrice et le courbe de Lorenz 
gini = 2*S gini
```

</details>

</details>

* * *

<details>
<summary> <h2> Représenter des analyses bivariées </h2> </summary>

<details><summary><h3>B1. Corrélation et covariance</h3></summary>

**Scatterplot** diagramme de dispersion

**tableau de contingence** (`pivot_table`) contient les effectifs conjoints $n_{ij}$
- la distribution conjointe empirique sont $n_{ij}$
- la distribution marginale empirique de $X_i$ ou de $Y_j$ sont les sous-totaux $n_i$ ou $n_j$
- la distribution conditionnelle empirique de $X_i$ sachant $Y_0$ est une ligne/colonne d'effectifs conjoints

**Covariance** $cov(X,Y) = s_{X,Y} = \frac{1}{n}\sum_{i=1}^{n} (x_i-\overline{x}) (y_i−\overline{y})$

**Correlation (linéaire) (de Pearson)** = covariance / les 2 écarts-types
- entre -1 et 1
- n'est pas causalité (cf multiples exemples, dont le paradoxe de Simpson)
- ne détecte que les relations linéaires (Cf exemple du cercle)

**Regression simple** $Y = a.X + b + \epsilon$ 
- où les estimateurs selon les Ordinary Least Squares (Moindres Carrés Ordinaires) sont : 
- $\hat{a} = \frac{cov(X,Y)}{s_X^2}$ et $\hat{b} = \overline{y} - \hat{a}.\overline{x}$
- $R^2$ = carré de la corrélation entre X et Y = % explicatif de la régression (somme des carrés expliqués / somme des carrés totaux)
- cette régression linéaire est peu robustes aux valeurs aberrantes

```python
import scipy.stats as st # corrélation par scipy.stats.pearsonr
st.pearsonr(depenses["solde_avt_ope"],-depenses["montant"])[0]
# Par numpy = matrice de covariance, corr = valeur en [1,0]
np.cov(depenses["solde_avt_ope"],-depenses["montant"],ddof=0)[1,0]

import statsmodels.api as sm
Y = courses['montant']
X = courses[['attente']]
X = X.copy() # On modifiera X, on en crée donc une copie
X['intercept'] = 1.
result = sm.OLS(Y, X).fit() # OLS = Ordinary Least Square (Moindres Carrés Ordinaire)
a,b = result.params['attente'],result.params['intercept']

plt.plot(courses.attente,courses.montant, "o")
plt.plot(np.arange(15),[a*x+b for x in np.arange(15)])
plt.xlabel("attente")
plt.ylabel("montant")
plt.show()
```

</details>

<details><summary> <h3>B.2. ANOVA = corrélation entre quanti et quali</h3></summary>

On décompose en 3 : 
- Total Sum of Squares = variation totale $= \sum\limits_{i=1}^{k} \sum\limits_{j=1}^{n_i} (y_{ij} − \overline{y})^2$
- Sum of Squares of the Model = variation interclasse (= somme des carrés expliqués) $= \sum\limits_{i=1}^{k} n_i (\hat{y_{i}} − \overline{y})^2$
- Sum of Squares of the Error = variation intraclasse $= \sum\limits_{i=1}^{k}  \sum\limits_{j=1}^{n_i} (y_{ij} − \hat{y_i})^2 = \sum\limits_{i=1}^{k} n_i s_i^2$
- le rapport de corrélation est maintenant eta squared : 

$$η_{Y,X}^2 = \frac{V_{interclasses}}{V_{totale}}$$

```python
modalites = sous_echantillon[X].unique()
groupes = []

for m in modalites:
    groupes.append(sous_echantillon[sous_echantillon[X]==m][Y])
# Graphiques = points rouges pour la moyenne  
medianprops = {'color':"black"}
meanprops = {'marker':'o', 'markeredgecolor':'black',
            'markerfacecolor':'firebrick'} 
plt.boxplot(groupes, labels=modalites, showfliers=False, medianprops=medianprops, 
            vert=False, patch_artist=True, showmeans=True, meanprops=meanprops)
plt.show()
```

</details>


<details><summary><h3> B.3. Chi-2 entre 2 variables quali</h3></summary>


Le tableau de contingence compare les effectifs conjoints $n_{ij}$ aux effectifs prévus en cas d'indépendance $n_j . frequency(i) = n_j . n_i / n$

Cette comparaison donne une corrélation $\xi_{ij}$ qu'on represente avec une carte de chaleur _heatmap_. Valeur entre 0 et 1 = % de contribution à la dépendance (non-indépendance). 
La somme des contribution = 100%. 



```python
import seaborn as sns
tx = cont.loc[:,["Total"]]
ty = cont.loc[["Total"],:]
n = len(data)
indep = tx.dot(ty) / n
c = cont.fillna(0) # On remplace les valeurs nulles par 0
measure = (c-indep)**2/indep
xi_n = measure.sum().sum()
table = measure/xi_n
sns.heatmap(table.iloc[:-1,:-1],annot=c.iloc[:-1,:-1])
plt.show()
```

</details>

</details>

* * *

<details>
<summary> <h2> Composantes et clustering </h2> </summary>

Supervise => j'ai déja des tag d'apprentissage. On parle de **classement**\= classification supervisée (en EN = "classification").  

Non supervisé = **clustering** 

![](ðŸ“ŒCheatsheet  Statistics_files/Image.png)


* * *

Distance (erreur = risque = eloignement des données vs prediction modele)

Attention : erreur = risque empirique != performance du modele

*   erreur quadratique (le + utilisé)
    

*   distance euclidienne = sqr(x^2 + y^2)
    

*   Distance manhattan = x + y
    
*   Pour chaines de caracteres = distance de Levenshtein = nbre mini d'operation (substitution, insertion, suppression) pour passer de l'une a l'autre.Â 
    

*   a connaitre = algo de Wagner et Fischer pour le calcul de la distance de Levenshtein.
    

algo paramétriques (eg regression = droite) => on cherche le parametreÂ Î¸ (qui peut etre multidimensionel)

algos non parametriques (+ complexité) => egg k-means qui est 'memory based' (garde toutes les données en memoire)

  

  

  

fuction loss = perte d'information

vraisemblance d'un jeu d'observations (x1...xN) par rapport Ã  un modèle en statistiques est la fonction suivante :Â Â L(Î¸)=p(x1...xN|Î¸)Â Â .= proba d'avoir x1...xN sachant \\Theta

Â Î¸^Â avec un accent circonflexeÂ lorsqu'on parle d'unÂ estimateur (eet non de la valeur reelle, intrinseque)

  

* * *

<details>
<summary> <h3> C.1.  Méthode factorielle = la + connue ACP </h3> </summary>

Factorielle :Â 

ACPÂ  ( = EN PCA) = Principal component analysis

*   Â  Â  recehche d'un (hyperplan) avec moment d'inertie max (étalement des points) = axe orthogonal Ã  l'hyperplan = donne indication sur la variabilité =
    

*   espace Rp de dimension p variables, contient Ni le nuage des individus
    

*   Rechreche des corrélations entre variablesÂ 
    

*   espace Rn de dimension n individus, contient Np le nuage des variables
    

De préférence ACP normée (centrée réduite)

3 graphiques :Â 

1.  1\. Pour l'objectif 1, ce sera la projection du nuage des individus NI sur les 2 premiers axes dâ€™inertie, câ€™est-Ã -dire sur le premier plan factoriel.
    
2.  Le second sâ€™appelle le cercle des corrélations.
    
3.  2\. Pour l'objectif 2, ce sera la projection du nuage des variables NK sur le premier plan factoriel.
    

  

combien de composantes = min (p nbr de varialbes et n-1 nombre individus)

\=> eboulis des valeurs propres (classées en valeur décroissante)

\=> frequent de n'analyser que le 1er plan (2 composantes). Critere du coude - reperer le # oÃ¹ le % inertie diminue + lentement. Criter de Kaiser (~contribution moyenen 100% / p)

</details>
  
<details>
<summary> <h3> C.2. Clulstering = Classification non supervisée = la + connue k-means (K-moyennes)
 </h3> </summary>

k-meansÂ 

k est un **hyperparamètre** (c'est Ã  nous de l'optimiser, ce n'est pas l'algo qui va le proposer).Â 

  

  

Trainig set vs testing set = 80% / 20% des données fournies

  

  

* * *

Conversion de timestamp unix =Â Â [www.epochconverter.com](http://www.epochconverter.com/)Â !

  

Erreur lexicale => Technique du dictionnaire.

Date => Format normalisé ISO8601Â 1977-04-22T06:00:00Z.

</details>

</details>

* * *

<details>  
<summary> <h2> Préparer ses données en sets <h2></summary>


**Centrer-reduire:**
- centrer réduire est nécessaire dans presque tous les cas. 
    - l'exception : la régression simple 
- Autres transformations possibles ? 
```python 
from sklearn.preprocessing import StandardScaler
X = np.asarray(X) # Version numpy :
X = pd.DataFrame(X) # Version pandas :
X.describe().round( 2) # fonctionne slt avec la version pandas DF

# On instancie notre scaler :
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # on peut séparer le fit (calcul des mean et std) du transform (centrage et réduction)

pd.DataFrame(X_scaled).describe().round(2)
```

**Training sets:**: 
- Base : "Don't train on the testing set !" : le split tout simple : 
```python
# * Training split*(https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
# NB Le 42 est un seed du random pour que ce soit toujours le même split 
```
- ⁉️ Oui mais si j'ai besoin de tester pour choisir un hyperparamètre? ⇒ découper le training set en "folds" pour choisir un hyperparamètre = `GridSearch`
- ⁉️ Oui mais si j'ai plusieurs modèles possibles ? ⇒ découper le training set en "folds" pour voir comment chaque modèle se comporte "en moyenne" avec une **valildation croisée**
    * validation croisée = couper le jeu d'apprentissage en k parties (_k folds_) Chaque partie est utilisée comme jeu de test à son tour. 
    * On peut faire LOO = Leave One Out = (k = N-1), mais préférable k= 5 ou 10 (pour temps de calcul)
    * stratifier la validation croisée = maintenir proportion de chaque classe 
    de la population dans les folds. 
    * PEut être utiliser pour Grid-search ou Line-Search = pour tester différentes valeurs d'un hyperparamètre.

![image](https://scikit-learn.org/stable/_images/grid_search_workflow.png)



</details>

***

<details> <summary><h2>Prédire </h2> </summary>

<details>  
<summary> <h3> M.1 Modeles predictifs linéaires = approximations supervisées </h3> </summary>

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

![image](https://user.oc-static.com/upload/2019/06/10/15601584374984_LassoRidge.png)

- Evaluer la performance d'une régression  
    - Avec ordre de grandeur : MSE et RMSE = mean squared error (mean of RSS = residual sum of squares = somme des carrés des résidus)
    - Sans ordre de grandeur : RMSLE et R^2
        - RMSLE = squared log error, si on veut une comparer sur des données à ordre de grandeur différents (erreur en % écart de la prédiction)
        - coef de détermination R^2 = 1- RSE (Relative Squared Error = erreur en % écart à la moyenne) = corrélation de Pearson entre valeurs vraies et prédites. See `sklearn.metrics.r2_score`


</details>
<details>
<summary> <h3> M.2 Modèles prédictifs linéaires pour classification </h3> </summary>

- la matrice de confusion `metric.confusion_matrix(y_true, y_predict)` : 

| Confusion Matrix  |   True -      | True +          | 
|:----------------- |:-------------:|:---------------:|
| Predicted -       | True negative | False negative (type II)  |
| Predicted +       | False positive (type I) | True positive  |

- mesures à connaitre: 
    - rappel (_recall_) = sensibilité (_sensitiviy_) = vrais positifs prédits en %des vrais positifs 
    - precision = vrais positifs prédits en %des positifs prédits
    - F-score = moyenne harmonique rappel & precision
    - spécificité = vrais négatifs prédits en % des vrais négatifs = sensibilité sur les négatifs
    - AUROC = area under ROC curve (1 pour classifieur parfait, 0.5 pour classifieur aléatoire)

ici mettre image courbe ROC sensibilité vs 1-speçificité


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
<summary> **Bagging** = Bootstrap aggregation </summary>
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

<details>
<summary> <h3> P.5 Mesurer la performance de la prévision </h3> </summary>

* Il y a de multiples scores de performance (si 'score", alors + haut = mieux, si "error", l'inverse), [liste ici](https://scikit-learn.org/stable/modules/model_evaluation.html), les classiques: 
    * Pour les régressions : mean_squared_error, mean_absolute_percentage_error, r2_score
    * Pour les classifications : accuracy, precision (no false positive), recall (no missing positive), F1 (une moyenne entre precision et recall)
    * Pour les regroupements (clustering): completeness, homogeinity, mutual information


Le compromis biais-variance : 
![image](https://user-images.githubusercontent.com/7408762/200091906-6977561e-4cdf-4097-b45a-7775aebf0a5e.png)

* biais d'induction = inductive bias = hypothèse à ajouter pour arriver à un "bon" modèle. Typique des "ill-posed problems" (problèmes mal posés). 
 
* Pour une prévision quanti : on calcule un MSE / RMSE ou un `r2_score` 
* Pour une classification : `accuracy` =  % des classes correctes


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
