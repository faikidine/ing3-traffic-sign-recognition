# Traffic Sign Recognition

Projet Deep Learning

## Groupe

| Avatar | Membre | GitHub |
|:-:|---|---|
| <img src="https://github.com/faikidine.png?size=80" width="80" alt="faikidine"> | faikidine | [![GitHub](https://img.shields.io/badge/GitHub-faikidine-181717?logo=github&logoColor=white)](https://github.com/faikidine) |
| <img src="https://github.com/ArnoGrl.png?size=80" width="80" alt="ArnoGrl"> | Arno | [![GitHub](https://img.shields.io/badge/GitHub-ArnoGrl-181717?logo=github&logoColor=white)](https://github.com/ArnoGrl) |
| <img src="https://github.com/Clemino.png?size=80" width="80" alt="Clemino"> | Clément | [![GitHub](https://img.shields.io/badge/GitHub-Clemino-181717?logo=github&logoColor=white)](https://github.com/Clemino) |

## Stack

- Python
- Keras / TensorFlow
- YOLO


## Notebook

[panneaux_signalisation.ipynb](panneaux_signalisation.ipynb) contient la préparation des données, les premiers modèles, les trois expériences sur le CNN, la comparaison des détecteurs YOLO et les évaluations finales. Chaque essai garde ses courbes, ses résultats et une explication, même quand il fait moins bien.

### Classification des panneaux

On entraîne sur 31 350 images GTSRB et on compare les modèles sur 7 859 images de validation. Les photos d'un même panneau restent dans le même ensemble.

| Modèle | Époques | Accuracy validation | Loss validation | F1 macro |
|---|---:|---:|---:|---:|
| Dense 128 | 3 | 73,13 % | 1,0243 | — |
| CNN initial, dense 128 | 3 | 91,68 % | 0,3354 | 0,8915 |
| Même CNN, entraînement plus long | 6 | 92,38 % | 0,3940 | 0,9073 |
| CNN, dense 64 | 3 | 89,27 % | 0,4446 | 0,8692 |
| CNN, Dropout 0,3 | 3 | **92,71 %** | **0,2507** | 0,9062 |

On ne change qu'un réglage à la fois. Le CNN à six époques commence à surapprendre après la quatrième : la loss du train baisse encore, mais celle de validation remonte. Le modèle à 64 neurones économise environ 47 % des paramètres, mais reconnaît moins bien les images de validation. Le Dropout donne la meilleure accuracy et la loss la plus faible. Le modèle à six époques garde toutefois un F1 macro légèrement meilleur.

Le choix se fait sur l'accuracy de validation, avec la loss pour départager une égalité, avant d'ouvrir le test officiel. Le **CNN avec Dropout** atteint ensuite **91,69 % sur les 12 630 images de test**, avec un **F1 macro de 0,8761**, une loss de 0,3549 et **1 049 erreurs**. Les matrices de confusion, les scores par classe et les exemples d'erreurs sont dans le notebook.

Résultats : [comparaison CNN](resultats/comparaison_cnn.csv), [test final](resultats/test_cnn.csv), [scores par classe](resultats/test_cnn_par_classe.csv). Les courbes de chaque nouvel essai sont aussi enregistrées dans `resultats/`.

### Vérification des données

Après discussion dans le groupe, on a complété les contrôles de préparation. La séparation par séquences était déjà en place. La liste des 1 307 séquences est enregistrée dans [sequences_train_validation.csv](resultats/sequences_train_validation.csv), puis rechargée pour vérifier qu'on retrouve les mêmes images dans le même ordre.

Le [détail par classe](resultats/repartition_train_validation.csv) montre que la validation contient entre 14,29 % et 25 % des images selon la classe. Les classes 0, 19 et 37 n'ont chacune qu'une séquence de validation : leurs 30 photos représentent un seul panneau, ce qui limite l'interprétation de leurs scores.

Les dimensions, les valeurs des pixels, les catégories et leur encodage ont été vérifiés sur les 39 209 images préparées : [résultats des contrôles](resultats/controle_preparation.csv). Tous passent. Le traitement reste RGB, 32 × 32 pixels et division par 255, y compris pour les zones extraites par YOLO.

Les auteurs de GTSRB indiquent que les données sont libres d'utilisation et demandent de citer leur publication, sans nommer de licence standard dans cette rubrique : [site officiel](https://benchmark.ini.rub.de/gtsrb_dataset.html). Le fichier ReadMe de GTSDB autorise son utilisation pour tout usage : [archive officielle](https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/published-archive.html).

### Détection avec YOLO

YOLO11n apprend à localiser une seule classe, « panneau », sur GTSDB : 480 scènes d'entraînement et 120 de validation. Le CNN choisit ensuite la catégorie de chaque zone extraite.

Le premier essai avec les poids COCO sans adaptation n'avait trouvé qu'un panneau sur les six scènes affichées, en annonçant un Stop sur un Sens interdit. On conserve ce résultat, puis on adapte le détecteur aux panneaux.

| Essai | Taille à la prédiction | mAP50 validation | mAP50-95 validation |
|---|---:|---:|---:|
| 5 époques | 640 | 0,843 | 0,637 |
| 10 époques | 640 | 0,885 | 0,677 |
| Mêmes poids de 10 époques | 960 | **0,932** | **0,765** |

Le détecteur retenu a appris pendant dix époques à 640 pixels ; on lui donne ensuite des images de 960 pixels pour conserver plus de détails. Ce choix repose sur la meilleure mAP50-95 de validation. Les trois configurations sont réévaluées avec les mêmes options : [comparaison YOLO](resultats/comparaison_yolo.csv).

Au seuil fixe de 0,25, la nouvelle chaîne retrouve **91 panneaux sur 98** en validation, dont **89 bien classés**, contre 75 retrouvés et 71 bien classés auparavant. En revanche, les fausses détections passent de **11 à 38**. Le gain de rappel s'accompagne donc d'une baisse de précision. Les premiers résultats restent dans [yolo_validation.csv](resultats/yolo_validation.csv), et les nouveaux dans [yolo_validation_finale.csv](resultats/yolo_validation_finale.csv).

### Test final sur les scènes

Les réglages sont fixés avant d'utiliser les 300 scènes du test GTSDB. On prend les images 00600 à 00899 et leurs annotations dans `FullIJCNN2013.zip`. Le fichier `gt.txt` fourni séparément sur le site contenait les annotations du train ; le contrôle a détecté ce problème. Le notebook garde cette vérification et explique le changement de source.

| Mesure sur le test | Résultat |
|---|---:|
| mAP50 YOLO | 0,975 |
| mAP50-95 YOLO | 0,821 |
| Panneaux retrouvés | 351 / 361 |
| Panneaux manqués | 10 |
| Panneaux retrouvés et bien classés | 347 / 361 |
| Mauvaise catégorie après une bonne détection | 4 |
| Fausses détections | 80 |
| Précision détection | 81,44 % |
| Rappel détection | 97,23 % |

Les **96,12 % de panneaux localisés et bien classés** doivent être lus avec les **80 fausses détections** : ces dernières ne figurent pas dans le dénominateur de ce pourcentage. Ces scènes contiennent 38 catégories, contre 43 dans le test GTSRB ; les deux scores ne mesurent donc pas la même chose.

Le seuil de détection reste à 0,25 et l'IoU de correspondance à 0,5. La mAP utilise plusieurs seuils de confiance, contrairement aux comptes ci-dessus. Résultats : [test YOLO](resultats/test_yolo.csv), [comptes par scène](resultats/test_pipeline.csv). Trois scènes avec les annotations et les prédictions montrent le fonctionnement des modèles retenus dans le notebook.

### Refaire les essais

Installer les bibliothèques du projet :

```bash
python -m pip install -r requirements.txt
```

Exécuter le notebook dans l'ordre. Il télécharge les archives officielles dans `data/`, prépare les images, entraîne les modèles et enregistre les résultats. La graine est fixée à 42 et la séparation enregistrée est vérifiée avant les essais. Le test sert uniquement à l'évaluation finale.

Les fichiers retenus sont `models/cnn_final.keras` et `models/yolo_final.pt`. Les choix et les réglages sont enregistrés dans [choix_cnn.json](resultats/choix_cnn.json) et [choix_yolo.json](resultats/choix_yolo.json). Les poids des premiers essais restent disponibles séparément. Les données et les poids ne sont pas inclus dans Git ; les cellules du notebook permettent de les recréer. Un nouvel entraînement peut donner de petits écarts de score.

### Limites et suites possibles

On a réalisé un entraînement par configuration. Certains scores par classe reposent sur très peu de panneaux distincts. Les exemples montrent encore des difficultés avec les panneaux sombres, flous, petits ou visuellement proches. Une probabilité élevée ne garantit pas une bonne réponse.

L'arrêt anticipé du CNN et le choix d'un autre seuil YOLO seraient des essais à faire sur la validation. Ils n'ont pas été réalisés et aucun réglage n'a été modifié à partir des tests finaux.

### Démo vidéo

[demo_video/](demo_video/README.md) permet de choisir le CNN et YOLO dans **Réglages**, ainsi que la taille des images, le seuil YOLO, l'IoU, le seuil d'affichage des catégories et la fréquence d'analyse. Les modèles retenus sont proposés au départ ; les premiers essais restent sélectionnables. Les calculs se font pendant la lecture, avec pause, déplacements, rectangles et scores. Chaque changement de configuration vide les résultats précédents pour comparer les essais sans les mélanger.
