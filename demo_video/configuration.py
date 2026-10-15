"""Modèles disponibles et réglages du lecteur."""

import json
import math
from pathlib import Path


RACINE = Path(__file__).resolve().parents[1]
MODELES = RACINE / "models"
NOMS = {
    "cnn_final.keras": "CNN retenu (Dropout 0,3)",
    "cnn_3epoques.keras": "CNN initial · 3 époques",
    "cnn_6epoques.keras": "CNN · 6 époques",
    "cnn_dense64.keras": "CNN dense 64 · 3 époques",
    "cnn_dropout03.keras": "CNN Dropout 0,3 · 3 époques",
    "yolo_final.pt": "YOLO retenu (10 époques)",
    "yolo_panneaux_5epoques.pt": "YOLO panneaux · 5 époques",
    "yolo_panneaux_10epoques.pt": "YOLO panneaux · 10 époques",
    "yolo11n.pt": "YOLO COCO · Stop seulement",
}
TAILLES = (320, 640, 960, 1280)


def fichiers_modeles():
    fichiers = {p.name: p for motif in ("cnn_*.keras", "yolo_*.pt")
                for p in MODELES.glob(motif) if p.is_file()}
    if (RACINE / "yolo11n.pt").is_file():
        fichiers["yolo11n.pt"] = RACINE / "yolo11n.pt"
    return fichiers


def catalogue():
    fichiers = fichiers_modeles()
    return {genre: [{"id": nom, "nom": NOMS.get(nom, Path(nom).stem)}
                    for nom in sorted(fichiers, key=lambda n: ("final" not in n, n == "yolo11n.pt", n))
                    if nom.endswith(extension)]
            for genre, extension in [("cnn", ".keras"), ("yolo", ".pt")]}


def reglages_defaut():
    disponibles = catalogue()
    def premier(genre):
        return disponibles[genre][0]["id"] if disponibles[genre] else ""
    resultat = {"cnn": premier("cnn"), "yolo": premier("yolo"), "taille": 640,
                "seuil": 0.25, "iou": 0.5, "seuil_cnn": 0.65, "cadence": 12.5, "boucle": False}
    choix = RACINE / "resultats/choix_yolo.json"
    if resultat["yolo"] == "yolo_final.pt" and choix.is_file():
        configuration = json.loads(choix.read_text())
        resultat.update(taille=configuration["taille"], seuil=configuration["seuil"],
                        iou=configuration["iou_nms"])
    return resultat


def valider_reglages(parametres):
    defaut = reglages_defaut()
    resultat = {cle: parametres.get(cle, defaut[cle]) for cle in ("cnn", "yolo")}
    disponibles = catalogue()
    for genre in ("cnn", "yolo"):
        if resultat[genre] not in {m["id"] for m in disponibles[genre]}:
            raise ValueError(f"Modèle {genre.upper()} indisponible.")
    resultat["taille"] = int(parametres.get("taille", defaut["taille"]))
    if resultat["taille"] not in TAILLES:
        raise ValueError("Taille YOLO invalide.")
    for cle, minimum, maximum in [("seuil", 0.05, 0.95), ("iou", 0.1, 0.9)]:
        valeur = float(parametres.get(cle, defaut[cle]))
        if not math.isfinite(valeur) or not minimum <= valeur <= maximum:
            raise ValueError(f"Réglage {cle} invalide.")
        resultat[cle] = valeur
    return resultat
