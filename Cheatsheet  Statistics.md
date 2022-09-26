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


<!-- details-->

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
    

*   Regrouper en gÃ©rant les contradictions
    

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

  

  

  

* * *

  

**Comment centrer reduire :**

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

  

  

  

  

  

* * *

<div><span><div><a href="evernote:///view/6367254/s57/f1dae14f-b0c0-4024-a6f5-7b2535f53308/67117fc9-036c-4028-b61e-04a2b3349d73/" rel="noopener noreferrer" rev="en_rl_none"> Python (Anthony Sardelliti) programmation-python</a></div><div><br/></div><div><a href="https://asardell.github.io/statistique-python/">https://asardell.github.io/statistique-python/</a></div><div><br/></div><div><br/></div><div>Méthode : </div><ul><li><div>Aller retour nettoyage et analyse</div></li><li><div>Valeurs manquantes : </div></li><ol><li><div>Trouver la bonne valeur (à la main)</div></li><li><div>Travailler avec un gruyère</div></li><li><div>Oublier la variable</div></li><li><div>Oublier les individus (mais les individus restants ne sont pas forcément représentatifs)</div></li><li><div>Imputer (= deviner, e.g. imputation par la moyenne, ou imputer intelligemment, eg selon âge pour la taille)</div></li></ol><li><div>Traiter les outliers (= valeur aberrantes)</div><div>trouvées par Z-score ou écart interquartile</div></li><ol><li><div>Trouver la bonne valeur (à la main)</div></li><li><div>Supprimer la valeur ou conserver la valeur ... en fonction des études (e.g. moyenne vs médiane)</div></li><li><div>... les valeurs atypiques sont intéressantes, et à mentionner</div></li></ol><li><div>Eliminer les doublons... si on peut</div></li><ul><li><div>Regrouper en gérant les contradictions</div></li></ul><li><div><br/></div></li></ul><div><br/></div><div><span style="font-family: -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, Roboto, Helvetica, Arial, sans-serif, &quot;Apple Color Emoji&quot;, &quot;Segoe UI Emoji&quot;, &quot;Segoe UI Symbol&quot;;"><span style="font-size: 16px;"><span style="color:rgb(42, 49, 53);">Méthode :</span></span></span></div><ul><li><div><span style="font-family: &quot;Helvetica Neue&quot;, Arial, sans-serif;"><span style="font-size: 13px;"><span style="color:rgb(51, 51, 51);">Méthode :</span></span></span></div></li><ul><li><div><span style="font-family: &quot;Helvetica Neue&quot;, Arial, sans-serif;"><span style="font-size: 13px;"><span style="color:rgb(51, 51, 51);">Aller retour nettoyage et analyse</span></span></span></div></li><li><div><span style="font-family: &quot;Helvetica Neue&quot;, Arial, sans-serif;"><span style="font-size: 13px;"><span style="color:rgb(51, 51, 51);">Valeurs manquantes :</span></span></span></div></li><ul><li><div><span style="font-family: &quot;Helvetica Neue&quot;, Arial, sans-serif;"><span style="font-size: 13px;"><span style="color:rgb(51, 51, 51);">Trouver la bonne valeur (à la main)</span></span></span></div></li><li><div><span style="font-family: &quot;Helvetica Neue&quot;, Arial, sans-serif;"><span style="font-size: 13px;"><span style="color:rgb(51, 51, 51);">Travailler avec un gruyère</span></span></span></div></li><li><div><span style="font-family: &quot;Helvetica Neue&quot;, Arial, sans-serif;"><span style="font-size: 13px;"><span style="color:rgb(51, 51, 51);">Oublier la variable</span></span></span></div></li><li><div><span style="font-family: &quot;Helvetica Neue&quot;, Arial, sans-serif;"><span style="font-size: 13px;"><span style="color:rgb(51, 51, 51);">Oublier les individus (mais les individus restants ne sont pas forcément représentatifs)</span></span></span></div></li><li><div><span style="font-family: &quot;Helvetica Neue&quot;, Arial, sans-serif;"><span style="font-size: 13px;"><span style="color:rgb(51, 51, 51);">Imputer (= deviner, e.g. imputation par la moyenne, ou imputer intelligemment, eg selon âge pour la taille)</span></span></span></div></li></ul><li><div><span style="font-family: &quot;Helvetica Neue&quot;, Arial, sans-serif;"><span style="font-size: 13px;"><span style="color:rgb(51, 51, 51);">Traiter les outliers (= valeur aberrantes)</span></span></span></div></li><li><div><span style="font-family: &quot;Helvetica Neue&quot;, Arial, sans-serif;"><span style="font-size: 13px;"><span style="color:rgb(51, 51, 51);">trouvées par Z-score ou écart interquartile</span></span></span></div></li><ul><li><div><span style="font-family: &quot;Helvetica Neue&quot;, Arial, sans-serif;"><span style="font-size: 13px;"><span style="color:rgb(51, 51, 51);">Trouver la bonne valeur (à la main)</span></span></span></div></li><li><div><span style="font-family: &quot;Helvetica Neue&quot;, Arial, sans-serif;"><span style="font-size: 13px;"><span style="color:rgb(51, 51, 51);">Supprimer la valeur ou conserver la valeur ... en fonction des études (e.g. moyenne vs médiane)</span></span></span></div></li><li><div><span style="font-family: &quot;Helvetica Neue&quot;, Arial, sans-serif;"><span style="font-size: 13px;"><span style="color:rgb(51, 51, 51);">... les valeurs atypiques sont intéressantes, et à mentionner</span></span></span></div></li></ul><li><div><span style="font-family: &quot;Helvetica Neue&quot;, Arial, sans-serif;"><span style="font-size: 13px;"><span style="color:rgb(51, 51, 51);">Eliminer les doublons... si on peut</span></span></span></div></li><ul><li><div><span style="font-family: &quot;Helvetica Neue&quot;, Arial, sans-serif;"><span style="font-size: 13px;"><span style="color:rgb(51, 51, 51);">Regrouper en gérant les contradictions</span></span></span></div></li></ul></ul></ul><div><span style="font-family: &quot;Helvetica Neue&quot;, Arial, sans-serif;"><span style="font-size: 16px;"><span style="color:rgb(51, 51, 51);">Méthode :</span></span></span></div><ul><li><div><span style="font-family: &quot;Helvetica Neue&quot;, Arial, sans-serif;"><span style="font-size: 13px;"><span style="color:rgb(51, 51, 51);">Aller retour nettoyage et analyse</span></span></span></div></li><li><div><span style="font-family: &quot;Helvetica Neue&quot;, Arial, sans-serif;"><span style="font-size: 13px;"><span style="color:rgb(51, 51, 51);">Valeurs manquantes :</span></span></span></div></li><ul><li><div><span style="font-family: &quot;Helvetica Neue&quot;, Arial, sans-serif;"><span style="font-size: 13px;"><span style="color:rgb(51, 51, 51);">Trouver la bonne valeur (à la main)</span></span></span></div></li><li><div><span style="font-family: &quot;Helvetica Neue&quot;, Arial, sans-serif;"><span style="font-size: 13px;"><span style="color:rgb(51, 51, 51);">Travailler avec un gruyère</span></span></span></div></li><li><div><span style="font-family: &quot;Helvetica Neue&quot;, Arial, sans-serif;"><span style="font-size: 13px;"><span style="color:rgb(51, 51, 51);">Oublier la variable</span></span></span></div></li><li><div><span style="font-family: &quot;Helvetica Neue&quot;, Arial, sans-serif;"><span style="font-size: 13px;"><span style="color:rgb(51, 51, 51);">Oublier les individus (mais les individus restants ne sont pas forcément représentatifs)</span></span></span></div></li><li><div><span style="font-family: &quot;Helvetica Neue&quot;, Arial, sans-serif;"><span style="font-size: 13px;"><span style="color:rgb(51, 51, 51);">Imputer (= deviner, e.g. imputation par la moyenne, ou imputer intelligemment, eg selon âge pour la taille)</span></span></span></div></li></ul><li><div><span style="font-family: &quot;Helvetica Neue&quot;, Arial, sans-serif;"><span style="font-size: 13px;"><span style="color:rgb(51, 51, 51);">Traiter les outliers (= valeur aberrantes)</span></span></span></div></li><li><div><span style="font-family: &quot;Helvetica Neue&quot;, Arial, sans-serif;"><span style="font-size: 13px;"><span style="color:rgb(51, 51, 51);">trouvées par Z-score ou écart interquartile</span></span></span></div></li><ul><li><div><span style="font-family: &quot;Helvetica Neue&quot;, Arial, sans-serif;"><span style="font-size: 13px;"><span style="color:rgb(51, 51, 51);">Trouver la bonne valeur (à la main)</span></span></span></div></li><li><div><span style="font-family: &quot;Helvetica Neue&quot;, Arial, sans-serif;"><span style="font-size: 13px;"><span style="color:rgb(51, 51, 51);">Supprimer la valeur ou conserver la valeur ... en fonction des études (e.g. moyenne vs médiane)</span></span></span></div></li><li><div><span style="font-family: &quot;Helvetica Neue&quot;, Arial, sans-serif;"><span style="font-size: 13px;"><span style="color:rgb(51, 51, 51);">... les valeurs atypiques sont intéressantes, et à mentionner</span></span></span></div></li></ul><li><div><span style="font-family: &quot;Helvetica Neue&quot;, Arial, sans-serif;"><span style="font-size: 13px;"><span style="color:rgb(51, 51, 51);">Eliminer les doublons... si on peut</span></span></span></div></li><ul><li><div><span style="font-family: &quot;Helvetica Neue&quot;, Arial, sans-serif;"><span style="font-size: 13px;"><span style="color:rgb(51, 51, 51);">Regrouper en gérant les contradictions</span></span></span></div></li></ul></ul><div>Vocabulaire : </div><ul><li><div>Stat descriptives</div></li><li><div>probabilités = statistique inférentielle</div></li><ul><li><div>tests satistiques, estimateurs</div></li></ul><li><div>individus =&gt; observation, reálisation</div></li><li><div>variables, population, échantillon, jeu de données</div></li></ul><img src="📌Cheatsheet  Statistics_files/Image.jpg" type="image/jpeg" data-filename="Image.jpg" style="--en-naturalWidth:843; --en-naturalHeight:329;" width="513px"/><ul><li><div>variables discrètes, continues</div></li><ul><li><div>Noir = nominal, ordinal, interval (cardinal), ratio</div></li></ul><li><div>variables qualitatives =&gt; modalités</div></li><ul><li><div>nominale</div></li><li><div>ordinale ( e.g. dates)</div></li><li><div>booléenne</div></li></ul><li><div>7 types d'erreurs : </div></li><ol><li><div>Valeurs manquantes</div></li><li><div>erreur lexicale (e.g. nombre là où chiffre attendu, ou liste limitative de pays possibles,,)</div></li><li><div>Irrégularité (e.g. unite de mesure différente)</div></li><li><div>??? supposition clef unique (e.g. 2 emails pour 1 personne)</div></li><li><div>formatage</div></li><li><div>doublon (+ parfois contradiction)</div></li><li><div>valeur extrême = <b>atypique ou aberrante</b> (outlier)</div></li></ol></ul><div><br/></div><ol><li><div>Deviner une valeur manquante = <b>imputation</b></div></li></ol><div>e.g. imputation par la moyenne (simple) -&gt; méthode de hot-deck, Machine Learning, régressions</div><ol start="6"><li><div>Doublon =&gt; méthode duplicated()</div></li><li><div>Valeur extrême  = midspread ou Z-score, ou boite a moustache</div></li></ol><hr/><div>Représenter des variables</div><div><br/></div><div><br/></div><div><br/></div><div><br/></div><div><br/></div><div><br/></div><div><br/></div><hr/><div><br/></div><div>Supervisé =&gt; j'ai déja des tag d'apprentissage. On parle de <b>classement</b>= classification supervisée (en EN = &quot;classification&quot;). </div><div>Non supervisé =  <b>clustering </b></div><img src="📌Cheatsheet  Statistics_files/Image.png" type="image/png" data-filename="Image.png" style="--en-naturalWidth:413; --en-naturalHeight:150;"/><div><br/></div><div>D</div><hr/><div>Distance (erreur = risque = eloignement des données vs prediction modele)</div><div>Attention : erreur = risque empirique != performance du modele</div><ul><li><div>erreur quadratique (le + utilisé)</div></li><ul><li><div>distance euclidienne = sqr(x^2 + y^2)</div></li></ul><li><div>Distance manhattan = x + y</div></li><li><div>Pour chaines de caracteres = distance de <span style="--en-highlight:yellow;background-color: #ffef9e;">Levenshtein</span> = nbre mini d'operation (substitution, insertion, suppression) pour passer de l'une a l'autre. </div></li><ul style="--en-todo:true;"><li style="--en-checked:false;"><div>a connaitre = algo de Wagner et Fischer pour le calcul de la distance de Levenshtein.</div></li></ul></ul><div>algo paramétriques (eg regression = droite) =&gt; on cherche le parametre <span style="font-size: 16px;"><span style="color:rgba(0, 0, 0, 0.92);">θ (qui peut etre multidimensionel)</span></span></div><div>algos non parametriques (+ complexité) =&gt; egg k-means qui est 'memory based' (garde toutes les données en memoire)</div><div><br/></div><div><br/></div><div><br/></div><div>fuction loss = perte d'information</div><div>vraisemblance d'un jeu d'observations (x1...xN) par rapport à un modèle en statistiques est la fonction suivante :  L(θ)=p(x1...xN|θ)  .= proba d'avoir x1...xN sachant \Theta</div><div><span style="font-size: 1rem;"><span style="color:rgba(0, 0, 0, 0.92);"> </span></span><span style="font-size: 16px;"><span style="color:rgba(0, 0, 0, 0.92);">θ^</span></span><span style="font-size: 1rem;"><span style="color:rgba(0, 0, 0, 0.92);"> avec un accent circonflexe lorsqu'on parle d'un estimateur (eet non de la valeur reelle, intrinseque)</span></span></div><div><br/></div><hr/><ol><li><div><span style="font-size: 1rem;"><span style="color:rgba(0, 0, 0, 0.92);">Méthode factorielle = la + connue ACP</span></span></div></li><li><div><span style="font-size: 1rem;"><span style="color:rgba(0, 0, 0, 0.92);">Clulstering = Classification non supervisée = la + connue k-means (K-moyennes)</span></span></div></li></ol><div><br/></div><div><span style="font-size: 1rem;"><span style="color:rgba(0, 0, 0, 0.92);">Factorielle : </span></span></div><div><span style="font-size: 1rem;"><span style="color:rgba(0, 0, 0, 0.92);">ACP  ( = EN PCA) = Principal component analysis</span></span></div><ul><li><div><span style="font-size: 1rem;"><span style="color:rgba(0, 0, 0, 0.92);">    recehche d'un (hyperplan) avec moment d'inertie max (étalement des points) = axe orthogonal à l'hyperplan = donne indication sur la variabilité =</span></span></div></li><ul><li><div><span style="font-size: 1rem;"><span style="color:rgba(0, 0, 0, 0.92);">espace Rp de dimension p variables, contient Ni le nuage des individus</span></span></div></li></ul><li><div><span style="color:rgba(0, 0, 0, 0.92);">Rechreche des corrélations entre variables </span></div></li><ul><li><div><span style="color:rgba(0, 0, 0, 0.92);">espace Rn de dimension n individus, contient Np le nuage des variables</span></div></li></ul></ul><div><span style="color:rgba(0, 0, 0, 0.92);">De préférence ACP normée (centrée réduite)</span></div><div><span style="color:rgba(0, 0, 0, 0.92);">3 graphiques : </span></div><ol><li><div><span style="color:rgba(0, 0, 0, 0.92);">1. Pour l'objectif 1, ce sera la projection du nuage des individus NI sur les 2 premiers axes d’inertie, c’est-à-dire sur le premier plan factoriel.</span></div></li><li><div>Le second s’appelle le cercle des corrélations.</div></li><li><div><span style="color:rgba(0, 0, 0, 0.92);">2. Pour l'objectif 2, ce sera la projection du nuage des variables NK sur le premier plan factoriel.</span></div></li></ol><div><br/></div><div><span style="font-size: 1rem;">combien de composantes = min (p nbr de varialbes et n-1 nombre individus)</span></div><div><span style="font-size: 1rem;">=&gt; eboulis des valeurs propres (classées en valeur décroissante)</span></div><div><span style="font-size: 1rem;">=&gt; frequent de n'analyser que le 1er plan (2 composantes). Critere du coude - reperer le # où le % inertie diminue + lentement. Criter de Kaiser (~contribution moyenen 100% / p)</span></div><div><br/></div><div><span style="color:rgba(0, 0, 0, 0.92);">k-means </span></div><div><span style="color:rgba(0, 0, 0, 0.92);">k est un</span> <b><span style="color:rgba(0, 0, 0, 0.92);">hyperparamètre</span></b> <span style="color:rgba(0, 0, 0, 0.92);">(c'est à nous de l'optimiser, ce n'est pas l'algo qui va le proposer). </span></div><div><br/></div><div><br/></div><div>Trainig set vs testing set = 80% / 20% des données fournies</div><div><br/></div><div><br/></div><hr/><div><span style="font-size: 1rem;"><span style="color:rgba(0, 0, 0, 0.92);">Conversion de timestamp unix =  </span><a href="http://www.epochconverter.com/" rev="en_rl_none"><span style="color:rgb(116, 81, 235);"><u>www.epochconverter.com</u></span></a><span style="color:rgba(0, 0, 0, 0.92);"> !</span></span></div><div><br/></div><div><span style="font-size: 1rem;"><span style="color:rgba(0, 0, 0, 0.92);">Erreur lexicale =&gt; Technique du dictionnaire.</span></span></div><div><span style="font-size: 1rem;"><span style="color:rgba(0, 0, 0, 0.92);">Date =&gt; Format normalisé ISO8601 </span></span><span style="font-family: monospace, monospace;"><span style="font-size: 1rem;"><span style="color:rgba(0, 0, 0, 0.92);">1977-04-22T06:00:00Z</span></span></span><span style="font-size: 1rem;"><span style="color:rgba(0, 0, 0, 0.92);">.</span></span></div><div><br/></div><div><br/></div><div><br/></div><hr/><div><br/></div><div><b><span style="font-size: 20px;">Comment centrer reduire : </span></b></div><div style="--en-codeblock:true;box-sizing: border-box; padding: 8px; font-family: Monaco, Menlo, Consolas, &quot;Courier New&quot;, monospace; font-size: 12px; color: rgb(51, 51, 51); border-top-left-radius: 4px; border-top-right-radius: 4px; border-bottom-right-radius: 4px; border-bottom-left-radius: 4px; background-color: rgb(251, 250, 248); border: 1px solid rgba(0, 0, 0, 0.14902); background-position: initial initial; background-repeat: initial initial;"><div>import pandas as pd</div><div>import numpy as np</div><div>from sklearn.preprocessing import StandardScaler</div></div><div>Définissons nos données :</div><div><br/></div><div style="--en-codeblock:true;box-sizing: border-box; padding: 8px; font-family: Monaco, Menlo, Consolas, &quot;Courier New&quot;, monospace; font-size: 12px; color: rgb(51, 51, 51); border-top-left-radius: 4px; border-top-right-radius: 4px; border-bottom-right-radius: 4px; border-bottom-left-radius: 4px; background-color: rgb(251, 250, 248); border: 1px solid rgba(0, 0, 0, 0.14902); background-position: initial initial; background-repeat: initial initial;"><div># Notre matrice de base : </div><div>X = [[12,    30,    80,  -100],     [-1000, 12,    -23,  10],     [14,    1000,  0,    0]]</div><div><br/></div><div># Version numpy : </div><div>X = np.asarray(X)</div><div># Version pandas : </div><div>X = pd.DataFrame(X)</div></div><div>Avec  pandas  , on peut calculer la moyenne et l'écart-type de chaque dimension :</div><div><br/></div><div style="--en-codeblock:true;box-sizing: border-box; padding: 8px; font-family: Monaco, Menlo, Consolas, &quot;Courier New&quot;, monospace; font-size: 12px; color: rgb(51, 51, 51); border-top-left-radius: 4px; border-top-right-radius: 4px; border-bottom-right-radius: 4px; border-bottom-left-radius: 4px; background-color: rgb(251, 250, 248); border: 1px solid rgba(0, 0, 0, 0.14902); background-position: initial initial; background-repeat: initial initial;"><div># On applique la methode .describe() pour avoir la moyenne et la .std(), et la méthode .round(2) pour arrondir à 2 décimales après la virgule : </div><div>X.describe()</div></div><div>On peut ensuite « scaler » nos données :</div><div><br/></div><div style="--en-codeblock:true;box-sizing: border-box; padding: 8px; font-family: Monaco, Menlo, Consolas, &quot;Courier New&quot;, monospace; font-size: 12px; color: rgb(51, 51, 51); border-top-left-radius: 4px; border-top-right-radius: 4px; border-bottom-right-radius: 4px; border-bottom-left-radius: 4px; background-color: rgb(251, 250, 248); border: 1px solid rgba(0, 0, 0, 0.14902); background-position: initial initial; background-repeat: initial initial;"><div># On instancie notre scaler : </div><div>scaler = StandardScaler()</div><div># On le fit : </div><div>scaler.fit(X)</div><div># On l'entraine : </div><div>X_scaled = scaler.transform(X)</div><div># On peut faire les 2 opérations en une ligne : </div><div>X_scaled = scaler.fit_transform(X)</div><div># On le transforme en DataFrame : </div><div>X_scaled = pd.DataFrame(X_scaled)</div><div># On peut appliquer la méthode .describe() et .round()</div><div>X_scaled.describe().round(2)</div></div><div><br/></div><div><br/></div><div><br/></div><div><br/></div><div><br/></div><hr/><div><br/></div><div><br/></div></span>
</div></body></html> 