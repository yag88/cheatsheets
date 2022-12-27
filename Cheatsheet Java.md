# Cheatsheet Java

# 0. Environnement

IDE = 
- essentiel, syntaxe stricte. 
- Eviter XCode sur Mac 
- utiliser Eclipse (pour projets complexes avec plusieurs sous-projets + systèmes de construction de code (build) comme Maven ou Gradle +  assistants, ou wizards, afin de ne pas partir d’un fichier vide à chaque fois)
    - Window/perspective/Java pour la perspective (arrangement de fenêtres) le + simple
- (voir aussi Intellij IDEA, NetBeans)
- IDE en ligne = https://www.jdoodle.com/online-java-compiler/ 
(voir aussi compilejava.net, codiva.io)
- JDK = kit de dévt Java (compileur + JVM virtual machine ). On parle aussi de JRE (Runtime Environment), en fait le JRE est une partie de la JVM. 
    - Oracle JDK est maintenant payant
    - OpenJDK sources
    - [AdoptOpenJDK](https://adoptopenjdk.net/) open-sources compilés _pre-built_ (actuellement choisir JDK8 avec Hotspot). 
- Java SE “Standard Edition” != Java EE “Entreprise Edition”.
- On partage son code via des bibliothèques (package)
    - clic-droit sur le fichier.java New/Package. 
    - on donne le nom de domaine de l'éditeur (inversé) pour hierarchiser les packages
    - fichiers sources dans _src_. 1 fichier par classe (m^ nom que la classes)

-     

# 1. Variables 



# A. Libraries classiques

## A.1 Time 
```(Java)
import java.time.LocalDateTime;

System.out.println("Date et heure du jour : " + LocalDateTime.now());
```