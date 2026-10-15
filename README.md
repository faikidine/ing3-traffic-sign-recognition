# Traffic Sign Recognition

Deep Learning project — Master, Semester 1, 2026-2027.

Build a computer vision system that recognizes traffic sign categories, starting from a CNN classifier on cropped signs and extending to sign detection in road scenes and videos.

## Team

| Avatar | Member | GitHub |
|:-:|---|---|
| <img src="https://github.com/faikidine.png?size=80" width="80" alt="faikidine"> | faikidine | [![GitHub](https://img.shields.io/badge/GitHub-faikidine-181717?logo=github&logoColor=white)](https://github.com/faikidine) |
| <img src="https://github.com/ArnoGrl.png?size=80" width="80" alt="ArnoGrl"> | Arno | [![GitHub](https://img.shields.io/badge/GitHub-ArnoGrl-181717?logo=github&logoColor=white)](https://github.com/ArnoGrl) |
| <img src="https://github.com/Clemino.png?size=80" width="80" alt="Clemino"> | Clément | [![GitHub](https://img.shields.io/badge/GitHub-Clemino-181717?logo=github&logoColor=white)](https://github.com/Clemino) |

Group of 3: parts 1 to 6 are required. Our extension is traffic sign detection with YOLO (part 8), followed by classification with our CNN.

## Stack

- Python
- Keras / TensorFlow
- YOLO (extensions)

## Defense

Friday, October 16, 2026.

## Roadmap

| Part | Topic | Status |
|:-:|---|:-:|
| 1 | Data research, understanding and preparation | Required |
| 2 | Baseline model (dense network) | Required |
| 3 | CNN with Keras | Required |
| 4 | Training | Required |
| 5 | Evaluation and error analysis | Required |
| 6 | Experiments and improvements (≥ 3 experiments) | Required |
| 7 | Enrichment with new course concepts | Extension |
| 8 | Traffic sign detection in a scene (YOLO) | Required for our project |
| 9 | Video extension | Extension |

1. **Data** — Choose a public traffic sign dataset (e.g. GTSRB, 43 classes) and describe its source, license, size, classes, images per class, image size and format, and class imbalance, with sample images. Resize, normalize, encode labels and split into train / validation / test (test set used only for the final evaluation).
2. **Baseline** — Simple dense network (Flatten → Dense → Softmax) used as a reference. Explain input flattening, weights and biases, forward pass, activations, loss, backpropagation, gradient descent and multiclass output.
3. **CNN** — Own justified architecture with several Conv2D + pooling blocks and a K-neuron Softmax output. Explain filters, convolution, feature maps, kernel size, stride, padding, ReLU, pooling, Flatten and Dense layers, with output shapes and parameter counts per layer.
4. **Training** — Justify the loss, optimizer, batch size, number of epochs and metrics. Plot and interpret training / validation loss and accuracy, and identify overfitting.
5. **Evaluation** — Confusion matrix and per-class performance: well-recognized signs, frequent confusions, hardest classes and likely causes (lighting, angle, size, blur, similar classes). Show image + true class + predicted class + probability, and analyze wrong predictions.
6. **Experiments** — At least 3 experiments, changing few parameters at a time (layers, filters, kernels, learning rate, optimizer, Dropout, data augmentation, early stopping, regularization). Comparison table and justified final model; only use augmentations that keep the sign's meaning (no horizontal flip).
7. **Enrichment** — Apply new course concepts: data augmentation, transfer learning and pre-trained networks, regularization, object detection, YOLO.
8. **Detection** — Process full road scenes: image → detection → bounding boxes → crop → CNN → predicted classes. Use a YOLO model trained or fine-tuned on traffic signs, apply the same preprocessing as for training, and display a box, class and score for each sign.
9. **Video** — Run the pipeline on a short video or image sequence: frames → detection → crop → CNN → predictions → display.

## Deliverables

1. Structured, commented notebook that runs end to end.
2. Concise presentation of the approach, architectures, experiments and results in the notebook's Markdown cells.
3. Demo of the final model.

## Notebook

[panneaux_signalisation.ipynb](panneaux_signalisation.ipynb) contient la préparation des données GTSRB, un réseau dense, un CNN à deux blocs, puis la détection des panneaux avec YOLO et leur classification par le CNN. Les courbes et les erreurs sont analysées sur la **validation**.

Les deux modèles sont essayés pendant 3 époques avec les mêmes données. On conserve le code, les réglages, les sorties et l'explication de chaque essai, y compris s'il est moins bon. Le test officiel est réservé à l'évaluation finale.

Résultats du 15 octobre 2026, sur 31 350 images d'entraînement et 7 859 images de validation, séparées par séquence :

| Modèle | Accuracy validation | Loss validation |
|---|---:|---:|
| Dense 128 | 73,13 % | 1,0243 |
| CNN 32-64, dense 128 | 91,68 % | 0,3354 |

Les courbes et l'analyse des erreurs sont dans le notebook. Ces scores sont calculés sur la validation.

### Premiers essais YOLO

YOLO11n apprend à localiser une seule classe, « panneau », sur GTSDB : 480 scènes pour l'entraînement, 120 pour la validation. Le CNN reconnaît ensuite la catégorie de chaque zone extraite.

| Essai | Résultat |
|---|---|
| YOLO pré-entraîné sur COCO, sans adaptation | Sur les six scènes affichées : une détection, un Sens interdit annoncé comme Stop |
| YOLO adapté pendant 5 époques, images de 640 pixels | mAP50 : 0,843 ; mAP50-95 : 0,637 |
| YOLO + CNN sur les 120 scènes de validation | 75 panneaux retrouvés sur 98, dont 71 bien classés ; 23 panneaux manqués et 11 fausses détections |

La chaîne utilise un seuil de détection de 0,25. Une détection correspond à une annotation si leur IoU atteint 0,5. Les petits panneaux restent difficiles à repérer. Les jeux de test officiels n'ont pas encore été utilisés.

Les résultats par époque sont dans [yolo_5epoques.csv](resultats/yolo_5epoques.csv), et les comptes par scène dans [yolo_validation.csv](resultats/yolo_validation.csv). Les poids entraînés sont sauvegardés localement dans `models/`.

### Ce qui reste à faire

- Compléter au moins trois expériences sur le CNN, en gardant chaque tentative et son interprétation.
- Choisir le modèle sur la validation, puis évaluer une fois le modèle retenu sur le test officiel.
- Vérifier et documenter les conditions d'utilisation du dataset.
- Comparer les améliorations de YOLO, notamment la durée d'entraînement et la taille des images, puis effectuer son évaluation finale.
- Finaliser la synthèse et la démonstration dans le notebook.

### Démo vidéo

[demo_video/](demo_video/README.md) reprend les mêmes poids YOLO et CNN dans un lecteur vidéo avec pause, déplacements, rectangles, scores, courbe des détections et aperçus des panneaux. La lecture continue pendant les calculs. Le README détaille le lien avec le notebook, chaque étape du traitement, les essais et les commandes pour les reproduire. Une version Python/OpenCV permet aussi de recevoir un flux H.264 et de conserver un essai en vidéo et CSV.
