"""Détection des panneaux, puis classification de chaque zone par le CNN."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import torch
from ultralytics import YOLO


MODELES = Path(__file__).resolve().parents[1] / "models"
NOMS = [
    "Limitation 20 km/h", "Limitation 30 km/h", "Limitation 50 km/h",
    "Limitation 60 km/h", "Limitation 70 km/h", "Limitation 80 km/h",
    "Fin de limitation 80 km/h", "Limitation 100 km/h", "Limitation 120 km/h",
    "Dépassement interdit", "Dépassement interdit aux poids lourds",
    "Priorité à la prochaine intersection", "Route prioritaire", "Cédez le passage",
    "Stop", "Circulation interdite", "Poids lourds interdits", "Sens interdit",
    "Danger", "Virage à gauche", "Virage à droite", "Double virage",
    "Chaussée déformée", "Chaussée glissante", "Chaussée rétrécie à droite",
    "Travaux", "Feux tricolores", "Passage piétons", "Enfants", "Cyclistes",
    "Neige ou verglas", "Animaux sauvages", "Fin des interdictions",
    "Tourner à droite", "Tourner à gauche", "Tout droit", "Tout droit ou à droite",
    "Tout droit ou à gauche", "Contournement par la droite", "Contournement par la gauche",
    "Rond-point", "Fin d'interdiction de dépasser", "Fin d'interdiction de dépasser pour poids lourds",
]


def charger_modeles():
    device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    detecteur = YOLO(str(MODELES / "yolo_panneaux_5epoques.pt"))
    cnn = tf.keras.models.load_model(MODELES / "cnn_3epoques.keras", compile=False)
    return detecteur, cnn, device


def reconnaitre(image_bgr, detecteur, cnn, device, seuil=0.25, taille=640):
    resultat = detecteur.predict(image_bgr, conf=seuil, iou=0.5, imgsz=taille, device=device, verbose=False)[0]
    boites = resultat.boxes.xyxy.cpu().numpy()
    scores = resultat.boxes.conf.cpu().numpy()
    if len(boites) == 0:
        return []

    # OpenCV lit en BGR ; le CNN a appris avec des images RGB entre 0 et 1.
    image_rgb = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    zones = []
    for x1, y1, x2, y2 in boites:
        marge = max(5, round(0.1 * max(x2 - x1, y2 - y1)))
        zone = image_rgb.crop((max(0, int(x1) - marge), max(0, int(y1) - marge),
                               min(image_rgb.width, int(np.ceil(x2)) + marge),
                               min(image_rgb.height, int(np.ceil(y2)) + marge)))
        zone = zone.resize((32, 32), Image.Resampling.BILINEAR)
        zones.append(np.asarray(zone, dtype="float32") / 255.0)

    # Un seul passage du CNN pour tous les panneaux de l'image.
    probabilites = cnn(np.array(zones), training=False).numpy()
    detections = []
    for boite, score, proba in zip(boites, scores, probabilites):
        classe = int(np.argmax(proba))
        detections.append({"boite": boite.tolist(), "classe": classe,
                           "nom": NOMS[classe], "yolo": float(score), "cnn": float(proba[classe])})
    return detections
