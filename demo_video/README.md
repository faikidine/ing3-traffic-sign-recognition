# Détection de panneaux dans une vidéo

Cette démo reprend les premiers essais du [notebook](../panneaux_signalisation.ipynb), avec le CNN sans Dropout à trois époques et YOLO à cinq époques. Les comparaisons ajoutées ensuite et les modèles retenus sont présentés dans le notebook. Dans le notebook, nous donnons une photo de route à YOLO, puis nous classons les panneaux trouvés avec notre CNN. Ici, nous faisons la même chose sur des images prises pendant la lecture d'une vidéo.

Les poids sont les mêmes. La démo n'entraîne aucun nouveau réseau et ne charge pas une vidéo déjà annotée : les rectangles, les catégories et les scores sont calculés pendant son utilisation.

## Lancer et reproduire la démo

Les commandes suivantes se lancent depuis la racine du projet.

1. Installer les dépendances :

   ```bash
   python -m pip install -r demo_video/requirements.txt
   ```

2. Avoir ces deux fichiers, produits par le notebook :

   ```text
   models/cnn_3epoques.keras
   models/yolo_panneaux_5epoques.pt
   ```

   Ils sont déjà présents sur la machine utilisée pour les essais. Ils ne sont pas inclus dans Git. Sur une autre machine, copier ces deux fichiers dans `models/` permet de reprendre exactement les poids utilisés ici. Pour refaire aussi l'apprentissage, exécuter le notebook dans l'ordre : préparation des données, entraînement du CNN, puis partie YOLO. Le notebook sauvegarde le CNN et copie le meilleur modèle YOLO de l'essai de cinq époques vers ces chemins. Un nouvel entraînement peut donner des valeurs légèrement différentes ; les fichiers sauvegardés permettent de rejouer le même modèle.

3. Démarrer le lecteur :

   ```bash
   python demo_video/lancer.py
   ```

4. Ouvrir [http://127.0.0.1:8501](http://127.0.0.1:8501), cliquer sur **Ouvrir une vidéo**, choisir son fichier puis cliquer sur **Lire**. Un MP4 contenant du H.264 convient. Pour d'autres formats, la lecture dépend des codecs pris en charge par le navigateur.

La vidéo s'ouvre directement depuis le fichier choisi : il n'est pas nécessaire d'attendre l'envoi de tout le fichier à Python. On peut mettre en pause, déplacer la barre de lecture, utiliser **−5 s**, **+5 s**, **Début** ou passer en plein écran. **Réglages** donne accès au seuil de détection, à la taille d'entrée YOLO et à la lecture en boucle. Après un déplacement, une nouvelle analyse est faite à l'instant choisi, même en pause.

Pour utiliser notre exemple de route belge :

```bash
python demo_video/preparer_video.py
```

Le script télécharge la séquence 03, caméra 00, de BelgiumTS (environ 605 Mo), trie les images et assemble les 600 premières dans `demo_video/data/belgique_0000.mp4`. La vidéo dure une minute à 10 images/s. Elle apparaît dans la liste du lecteur après actualisation de la page. `--debut 600 --images 600` permet de prendre la portion suivante.

## Ce que nous reprenons du notebook

| Partie du projet | Dans la démo |
|---|---|
| Réseau dense sur GTSRB | Conservé comme point de comparaison dans le notebook ; il n'intervient pas dans la démo. |
| CNN entraîné pendant 3 époques sur GTSRB | Même modèle `cnn_3epoques.keras`, avec les mêmes 43 catégories, dans le même ordre. |
| YOLO11n adapté pendant 5 époques sur GTSDB | Même fichier `yolo_panneaux_5epoques.pt`, qui correspond au meilleur poids sauvegardé pendant cet essai. |
| Détection puis classification sur une scène | Même enchaînement : YOLO → boîte → découpe avec marge → redimensionnement → CNN. |
| Réglages de détection de l'évaluation | Par défaut : confiance YOLO 0,25, IoU de suppression des doublons 0,5 et taille YOLO 640. |

Le CNN a appris sur des images de panneaux déjà cadrés. Son entrée est une image RGB de **32 × 32 pixels**, avec les valeurs des pixels divisées par 255. Il contient deux convolutions 3 × 3 de 32 puis 64 filtres, chacune suivie d'un max pooling 2 × 2, puis `Flatten`, une couche dense de 128 neurones et une sortie Softmax de 43 neurones. Les convolutions et la couche dense intermédiaire utilisent ReLU. L'apprentissage utilise Adam à 0,001, la cross-entropie catégorielle, des lots de 64 et trois époques.

Ce CNN atteint **91,68 % d'accuracy sur la validation GTSRB** du notebook. Cela mesure la reconnaissance de panneaux cadrés dans ce jeu de données. Ce n'est pas le taux de réussite de la vidéo.

YOLO a un autre rôle : trouver **où se trouvent les panneaux** dans toute la scène. Nous sommes partis de YOLO11n pré-entraîné sur COCO, puis avons adapté ses poids à une seule classe, « panneau », sur 480 scènes GTSDB, avec 120 scènes pour la validation. L'essai dure cinq époques, à 640 pixels, avec des lots de 8, AdamW et un taux initial de 0,001. Le notebook contient les réglages d'augmentation et les courbes de cet essai. La démo utilise le modèle adapté, pas le modèle COCO de départ.

Sur les 120 scènes de validation du notebook, la chaîne retrouve 75 panneaux sur 98, dont 71 sont bien classés. Il reste 23 panneaux manqués et 11 fausses détections. Ces résultats servent de référence pour comprendre les limites observées ensuite dans la vidéo.

## Ce qui se passe pendant la lecture

Le lecteur utilise HTML et JavaScript pour afficher la vidéo, et un petit serveur Python avec Starlette/Uvicorn pour exécuter les modèles. Il n'utilise plus Streamlit. L'adresse garde le port 8501 pour continuer à ouvrir la démo au même endroit.

### 1. Lire la vidéo et prélever une image

Le navigateur décode la vidéo et la lit à sa cadence normale, avec son son et ses commandes habituelles. Lorsqu'une nouvelle image est affichée, le lecteur peut en prendre une copie pour l'analyse.

Une seule demande de calcul peut être en cours par lecteur. Si Python est occupé, la vidéo continue et les images intermédiaires ne sont pas mises en attente. Une fois le calcul terminé, la prochaine demande porte sur une image récente. Les demandes sont espacées d'au moins 80 ms, soit au maximum 12,5 analyses/s ; la cadence réelle peut être inférieure, selon la vidéo et le temps de calcul.

Ainsi, une vidéo à 30 images/s reste lue à 30 images/s sans exiger 30 passages dans les réseaux chaque seconde. En contrepartie, un panneau visible très brièvement peut passer entre deux analyses. Il n'y a pas de suivi d'objet entre les images.

### 2. Envoyer l'image à Python

La copie envoyée est réduite, si nécessaire, pour que son plus grand côté ne dépasse pas **1 280 pixels**, puis encodée en JPEG avec une qualité de 0,88. La vidéo affichée garde sa résolution d'origine. Seule cette image est envoyée au serveur local, pas la vidéo complète importée.

C'est une différence avec les photos fixes du notebook : la réduction et la compression peuvent modifier les pixels, donc aussi les prédictions, surtout sur les petits panneaux. Nous réutilisons les mêmes réseaux et la même préparation des zones, mais une image vidéo compressée n'est pas identique à une photo d'origine.

### 3. Localiser les panneaux avec YOLO

Python décode le JPEG avec OpenCV. La fonction `reconnaitre()` de [panneaux.py](panneaux.py) appelle le modèle YOLO avec `conf=0.25`, `iou=0.5` et `imgsz=640` par défaut. Ultralytics adapte l'image à cette taille en conservant ses proportions et en ajoutant les bordures nécessaires.

YOLO renvoie des coordonnées `(x1, y1, x2, y2)` et un score pour chaque boîte retenue. Le seuil de 0,25 élimine les détections moins confiantes. La suppression des doublons utilise l'IoU à 0,5 : elle évite de conserver plusieurs boîtes très superposées pour le même panneau. Ce réglage est distinct de l'IoU utilisé dans le notebook pour comparer une prédiction à une annotation lors de l'évaluation.

### 4. Classer chaque zone avec le CNN

Pour chaque boîte, nous reprenons les opérations du notebook :

1. Convertir l'image OpenCV, qui est en BGR, vers RGB.
2. Ajouter autour du panneau une marge de `max(5, round(0.1 × max(largeur, hauteur)))` pixels, sans sortir de l'image.
3. Découper cette zone, puis la redimensionner en **32 × 32** avec l'interpolation bilinéaire de Pillow.
4. Convertir les pixels en `float32` et les diviser par **255**.
5. Passer les zones ensemble dans le CNN, avec `training=False`.
6. Prendre la catégorie ayant la plus grande probabilité Softmax parmi les 43 sorties.

Il n'y a ni calcul de gradients, ni mise à jour des poids. Si YOLO ne trouve aucune boîte, le CNN n'est pas appelé. Les modèles sont chargés une seule fois au démarrage ; Python utilise CUDA si disponible, sinon MPS sur Mac compatible, sinon le CPU.

### 5. Afficher les résultats

Python renvoie les coordonnées, la catégorie proposée, le score YOLO et la probabilité CNN. Les coordonnées sont divisées par la largeur et la hauteur de l'image analysée. Le navigateur peut ainsi replacer les rectangles sur la vidéo, même quand la fenêtre change de taille ou quand la vidéo a des bandes autour d'elle.

Les résultats arrivent après l'image qui a été envoyée : il reste donc un petit retard dû au calcul et à l'échange. À vitesse normale, une réponse ayant plus de 0,45 seconde de décalage est écartée. En pause, la tolérance est de 0,1 seconde. Un déplacement dans la vidéo invalide les réponses encore attendues de l'ancien instant. Les rectangles ne sont pas des positions prédites pour les images suivantes.

Le plein écran porte sur le lecteur entier, pour conserver la vidéo et ses rectangles ensemble.

## Lire les scores, la courbe et les images

- **YOLO** : score associé à la localisation d'un panneau.
- **CNN** : probabilité attribuée à la catégorie choisie. En dessous de 65 %, le texte devient « Catégorie incertaine » et le rectangle est orange. Au-dessus, la catégorie est affichée et le rectangle est vert. Ce seuil sert à l'affichage ; il ne supprime pas la détection et ne garantit pas qu'une catégorie est correcte.
- **ms** : temps entre le début de la préparation de l'image et la réception du résultat. Il comprend la copie, le JPEG, l'échange local et le calcul. Ce n'est pas uniquement le temps de YOLO.
- **analyses/s** : fréquence des résultats récents acceptés par le lecteur, calculée sur les deux dernières secondes. Elle ne correspond pas à la fréquence de lecture de la vidéo.
- **Panneaux par analyse** : nombre de boîtes retenues pour chacune des 30 dernières analyses affichées, positionnées selon leur instant dans la vidéo. La courbe repart après un déplacement ou un changement de réglage. Une baisse peut signifier qu'un panneau est sorti du champ ou que le modèle ne le retrouve plus ; elle ne mesure pas un apprentissage.
- **Derniers panneaux repérés** : deux découpes provenant des images réellement envoyées à Python, avec leur instant et la catégorie proposée. Elles sont mises à jour au plus une fois par seconde quand des panneaux sont trouvés. Ces aperçus montrent la boîte YOLO sans la marge ajoutée pour le CNN. Ils restent visibles pour pouvoir examiner les dernières détections, même si le panneau n'est plus à l'écran.
- **Historique** : les 40 dernières détections acceptées, avec leurs scores. Le même panneau peut y apparaître plusieurs fois. Ce n'est pas un comptage de panneaux distincts.

L'historique et les aperçus sont conservés pendant les déplacements dans une même vidéo et sont vidés quand on change de vidéo. Ils restent en mémoire dans la page et disparaissent à son actualisation.

## Ce que les essais montrent

| Essai | Observation et décision |
|---|---|
| Python/OpenCV, 100 premières images de la séquence belge | 6 détections cumulées, environ 42 images/s pour YOLO + CNN seuls. [Prédictions](resultats/essai_640.csv). Ce débit de calcul ne suffisait pas à garantir la fluidité de l'interface. |
| Premier lecteur Streamlit | Les tests des boutons et du calcul passaient, mais l'affichage réel sur une longue vidéo était lent et pouvait montrer une image cassée. Le rafraîchissement image par image n'était pas un bon lecteur vidéo. Le code est conservé dans [essai_streamlit.py](essai_streamlit.py), avec [son test](verifier_streamlit.py) et [ses anciens résultats](resultats/interface.json). |
| Lecteur vidéo dans le navigateur | Ouverture directe du fichier H.264 de 269 Mo, 1 920 × 1 080 et 7 min 47 s, puis analyse réelle. Pendant une mesure, la vidéo avance de 15,17 s en 15,16 s réelles. Avance et retour de cinq secondes vérifiés en pause. [Mesures](resultats/lecteur_navigateur.json). |
| Vidéo et résultats dans la fenêtre | Répartition 60 % / 40 % sur ordinateur, courbe et deux aperçus réels. Absence de défilement de la page vérifiée à 1 280 × 720, 681 × 645 et 390 × 700. Sur une petite fenêtre, les résultats défilent à l'intérieur de leur panneau. [Capture](resultats/lecteur_web.png). |
| Requêtes vidéo partielles et inférence réelle | Accès à un morceau du fichier et calcul sur cinq images, dont un retour au même instant, vérifiés. Temps médian aller-retour mesuré par le script : 23,5 ms. Ce test ne comprend pas la copie et l'encodage dans le navigateur. [Résultats](resultats/lecteur_api.json). |
| Envoi UDP avec `-c:v copy` | Échec au décodage en rejoignant le flux en cours : images clés trop espacées. [Trace](resultats/flux_copie.log). |
| Envoi UDP avec `libx264`, `-g 10` et `-tune zerolatency` | 100 images reçues et calculées, 25 détections cumulées. Le flux est rejoint en cours : il ne s'agit pas du même extrait que le premier essai. [Prédictions](resultats/flux_h264.csv) et [trace](resultats/flux_h264.log). |

Un exemple concret apparaît autour de 9,5 à 10 secondes dans la vidéo belge : un panneau triangulaire portant une croix d'intersection est classé « Travaux », puis « Double virage ». Dans le test actuel, à 10 secondes, le CNN donne environ **96 %** à « Double virage », alors que cette catégorie est incorrecte. YOLO a trouvé une zone pertinente, mais la classification échoue. Les aperçus permettent de voir cette erreur au lieu de se fier uniquement au score.

Les routes, les panneaux et la qualité d'image diffèrent des données d'entraînement. Nous n'avons pas d'annotations de référence pour ces vidéos : leur lecture ne permet donc pas d'annoncer une accuracy ou une mAP. Améliorer le lecteur ne corrige pas les limites des modèles. Les tests officiels du projet restent réservés à l'évaluation finale.

Pour refaire le test du serveur, laisser le lecteur lancé et exécuter :

```bash
python demo_video/verifier_interface.py
```

Il utilise les vrais modèles et des images brutes de la séquence belge, vérifie l'accès partiel au MP4 et sauvegarde les prédictions dans `resultats/lecteur_api.json`. La lecture, les déplacements et la disposition se vérifient en plus dans le navigateur.

## Lire un flux H.264 avec Python

Le lecteur web accepte les fichiers vidéo lisibles par le navigateur. Pour recevoir directement un flux UDP ou RTSP, nous gardons la version OpenCV, qui utilise le même [panneaux.py](panneaux.py).

Avec FFmpeg installé, envoyer la séquence belge en boucle dans un premier terminal :

```bash
ffmpeg -re -stream_loop -1 -i demo_video/data/belgique_0000.mp4 -an -c:v libx264 -preset ultrafast -tune zerolatency -g 10 -pix_fmt yuv420p -f mpegts 'udp://127.0.0.1:1234?pkt_size=1316'
```

Puis lancer la réception dans un deuxième terminal :

```bash
python demo_video/demo.py 'udp://127.0.0.1:1234?fifo_size=10000&overrun_nonfatal=1'
```

Le H.264 est transporté dans du MPEG-TS sur UDP. Avec cette source à 10 images/s, `-g 10` introduit une image clé chaque seconde, ce qui permet de rejoindre le flux en cours. **Espace** met en pause, **N** passe à la source suivante et **Q / Échap** ferme la fenêtre. Cette version traite les images successivement : si le calcul est trop lent, elle peut prendre du retard sur le flux.

Elle accepte aussi un ou plusieurs fichiers :

```bash
python demo_video/demo.py route1.mp4 route2.mp4 --boucle
```

Pour garder un essai avec ses scores :

```bash
python demo_video/demo.py demo_video/data/belgique_0000.mp4 --sans-affichage --max-images 100 --sortie demo_video/resultats/essai.mp4 --journal demo_video/resultats/essai.csv
```

Le CSV conserve les boîtes, les catégories et les scores pour chaque image traitée. Le MP4 annoté sert à revoir cet essai ; il n'est pas utilisé comme source des détections du lecteur web.

## Vidéos de conduite consultées

| Dataset | Contenu et choix |
|---|---|
| [BelgiumTS](https://btsd.ethz.ch/shareddata/) | Quatre séquences belges, huit caméras, images ordonnées en JPEG 2000. Retenu pour les panneaux européens et le téléchargement direct ; nous assemblons les images de la caméra avant gauche en MP4 H.264. |
| [CURE-TSD](https://github.com/olivesgatech/CURE-TSD) | Vidéos réelles et synthétiques de 300 images, avec notamment pluie, flou et obscurité. L'accès passe par un formulaire ; non utilisé dans les essais. |
| [BDD100K](https://bair.berkeley.edu/blog/2018/05/30/bdd/) | 100 000 vidéos de conduite aux États-Unis. Intéressant pour varier les scènes, mais les panneaux américains diffèrent de ceux appris par le CNN ; non utilisé ici. |

L'[article BelgiumTS](https://btsd.ethz.ch/shareddata/publications/Timofte-MVA-2011-preprint.pdf), de Radu Timofte, Karel Zimmermann et Luc Van Gool, décrit des prises de vue espacées d'environ un mètre. Les 10 images/s de notre assemblage sont donc une cadence de lecture choisie, pas la vitesse d'origine de la voiture.
