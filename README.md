# Traffic Sign Recognition

Deep Learning project — Master, Semester 1, 2026-2027.

Build a computer vision system that recognizes traffic sign categories, starting from a CNN classifier on cropped signs and extending to sign detection in road scenes and videos.

## Team

| Avatar | Member | GitHub |
|:-:|---|---|
| <img src="https://github.com/faikidine.png?size=80" width="80" alt="faikidine"> | faikidine | [![GitHub](https://img.shields.io/badge/GitHub-faikidine-181717?logo=github&logoColor=white)](https://github.com/faikidine) |
| <img src="https://github.com/ArnoGrl.png?size=80" width="80" alt="ArnoGrl"> | Arno | [![GitHub](https://img.shields.io/badge/GitHub-ArnoGrl-181717?logo=github&logoColor=white)](https://github.com/ArnoGrl) |
| <img src="https://github.com/Clemino.png?size=80" width="80" alt="Clemino"> | Clément | [![GitHub](https://img.shields.io/badge/GitHub-Clemino-181717?logo=github&logoColor=white)](https://github.com/Clemino) |

Group of 3: parts 1 to 6 are required, plus at least one substantial extension (e.g. traffic sign detection with YOLO, part 8).

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
| 8 | Traffic sign detection in a scene (YOLO) | Extension |
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
2. Presentation of the approach, architectures, experiments and results.
3. Demo of the final model.

## Grading

| Criterion | Parts | Points |
|---|:-:|:-:|
| Data and preparation | 1 | 3 |
| Baseline model + CNN | 2–3 | 5 |
| Training and evaluation | 4–5 | 3 |
| Experiments and critical analysis | 6 | 4 |
| Enrichment (improvements and extensions) | 7–9 | 2 |
| Defense, demo and mastery of the work | Defense | 3 |
| **Total** | | **20** |
