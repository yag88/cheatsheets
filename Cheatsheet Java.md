# Cheatsheet Java

# 0. Environnement

IDE = 
- essentiel, syntaxe stricte. 
- Eviter XCode sur Mac 
- utiliser Eclipse (pour projets complexes avec plusieurs sous-projets + systèmes de construction de code (build) comme Maven ou Gradle +  assistants, ou wizards, afin de ne pas partir d’un fichier vide à chaque fois)
- (voir aussi Intellij IDEA, NetBeans)
- IDE en ligne = https://www.jdoodle.com/online-java-compiler/ 
(voir aussi compilejava.net, codiva.io)
- JDK = kit de dévt Java (compileur + JVM virtual machine )
    - Oracle JDK est maintenant payant
    - OpenJDK sources
    - [AdoptOpenJDK](https://adoptopenjdk.net/) open-sources compilés _pre-built_. 



# 1. Variables 



# A. Libraries classiques

## A.1 Time 
```(Java)
import java.time.LocalDateTime;

System.out.println("Date et heure du jour : " + LocalDateTime.now());
```