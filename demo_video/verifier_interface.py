"""Essai du lecteur : accès vidéo partiel et calcul réel sur des images de route."""

import json
from pathlib import Path
import statistics
import time

import cv2
import requests


DOSSIER = Path(__file__).resolve().parent
URL = "http://127.0.0.1:8501"


def envoyer(image):
    hauteur, largeur = image.shape[:2]
    ratio = min(1, 1280 / max(largeur, hauteur))
    image = cv2.resize(image, (round(largeur * ratio), round(hauteur * ratio)))
    ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    assert ok
    for _ in range(50):
        debut = time.perf_counter()
        reponse = requests.post(URL + "/detecter?seuil=0.25&taille=640", data=jpeg.tobytes(),
                                headers={"Content-Type": "image/jpeg"}, timeout=10)
        if reponse.status_code != 429:
            reponse.raise_for_status()
            return reponse.json(), (time.perf_counter() - debut) * 1000
        time.sleep(.05)
    raise RuntimeError("Le modèle est occupé par un autre lecteur.")


assert requests.get(URL + "/etat", timeout=5).json()["pret"], "Attendre le chargement des modèles."
liste = requests.get(URL + "/videos", timeout=5).json()
source = next(v for v in liste if v["nom"] == "belgique_0000.mp4")
reponse = requests.get(URL + source["url"], headers={"Range": "bytes=4096-5119"}, timeout=5)
assert reponse.status_code == 206 and len(reponse.content) == 1024
assert reponse.content == (DOSSIER / "data/belgique_0000.mp4").read_bytes()[4096:5120]

capture = cv2.VideoCapture(str(DOSSIER / "data/belgique_0000.mp4"))
essais = []
try:
    # On vérifie aussi un retour en arrière, en redemandant un calcul.
    for seconde in [9.5, 9.8, 10.0, 0.0, 9.5]:
        capture.set(cv2.CAP_PROP_POS_MSEC, seconde * 1000)
        ok, image = capture.read()
        assert ok
        resultat, duree = envoyer(image)
        for d in resultat["detections"]:
            assert 0 <= d["classe"] < 43
            assert all(0 <= v <= 1 for v in d["boite"])
            assert 0 <= d["cnn"] <= 1 and 0 <= d["yolo"] <= 1
        essais.append({"seconde": seconde, "aller_retour_ms": round(duree, 1), **resultat})
finally:
    capture.release()
assert any(e["detections"] for e in essais), "Aucun panneau sur les images d'essai."
resultats = {"acces_partiel_video": "OK", "inference_reelle": "OK", "boites_normalisees": "OK",
             "retour_sur_une_image": "OK", "mediane_aller_retour_ms": round(statistics.median(e["aller_retour_ms"] for e in essais), 1),
             "essais": essais}
(DOSSIER / "resultats/lecteur_api.json").write_text(json.dumps(resultats, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(resultats, ensure_ascii=False, indent=2))
