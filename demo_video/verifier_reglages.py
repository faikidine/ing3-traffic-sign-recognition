"""Compare les choix du lecteur à une inférence directe sur une vraie image."""

import asyncio
import json
from pathlib import Path
import time

import cv2
import numpy as np
from starlette.testclient import TestClient

from app import app


DOSSIER = Path(__file__).resolve().parent
image = cv2.imread(str(DOSSIER.parent / "data/gtsdb_yolo/images/val/00501.jpg"))
assert image is not None, "Exécuter la préparation GTSDB du notebook."
ok, jpeg = cv2.imencode(".jpg", image)
assert ok
image_envoyee = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
resultats = []

with TestClient(app) as client:
    for _ in range(300):
        etat = client.get("/etat").json()
        assert not etat["erreur"], etat["erreur"]
        if etat["pret"]:
            break
        time.sleep(.1)
    assert etat["pret"], "Les modèles ne sont pas prêts."
    catalogue = client.get("/modeles").json()
    assert catalogue["defaut"]["cnn"] == "cnn_final.keras"
    assert catalogue["defaut"]["yolo"] == "yolo_final.pt"
    assert catalogue["defaut"]["taille"] == 960

    def envoyer(configuration):
        reponse = client.post("/detecter", params=configuration, content=jpeg.tobytes())
        assert reponse.status_code == 200, reponse.text
        sortie = reponse.json()
        for cle in ("cnn", "yolo", "taille", "seuil", "iou"):
            assert sortie["configuration"][cle] == configuration[cle]
        for d in sortie["detections"]:
            assert 0 <= d["classe"] < 43 and all(0 <= v <= 1 for v in d["boite"])
        resultats.append({"configuration": sortie["configuration"],
                          "panneaux": len(sortie["detections"]), "calcul_ms": sortie["calcul_ms"]})
        return sortie

    # Tous les couples disponibles passent réellement dans les deux modèles.
    for yolo in catalogue["yolo"]:
        for cnn in catalogue["cnn"]:
            envoyer({"yolo": yolo["id"], "cnn": cnn["id"], "taille": 640, "seuil": .25, "iou": .5})

    # Le calcul du lecteur doit donner les mêmes scores et rectangles que l'appel direct.
    from panneaux import charger_modeles, reconnaitre, charger_cnn, charger_yolo
    reglages = {"yolo": "yolo_panneaux_10epoques.pt", "cnn": "cnn_dense64.keras",
                "taille": 960, "seuil": .4, "iou": .3}
    sortie = envoyer(reglages)
    direct = reconnaitre(image_envoyee, *charger_modeles(reglages["yolo"], reglages["cnn"]),
                          seuil=.4, taille=960, iou=.3)
    assert len(direct) == len(sortie["detections"]) > 0
    h, w = image.shape[:2]
    for attendu, obtenu in zip(direct, sortie["detections"]):
        assert attendu["classe"] == obtenu["classe"]
        np.testing.assert_allclose(attendu["boite"] / np.array([w, h, w, h]), obtenu["boite"], atol=1e-5)
        np.testing.assert_allclose([attendu["cnn"], attendu["yolo"]], [obtenu["cnn"], obtenu["yolo"]], atol=1e-5)

    avant = (charger_cnn.cache_info().misses, charger_yolo.cache_info().misses)
    envoyer({**reglages, "cnn": "cnn_3epoques.keras", "yolo": "yolo_panneaux_5epoques.pt"})
    envoyer(reglages)
    assert avant == (charger_cnn.cache_info().misses, charger_yolo.cache_info().misses)
    for taille in (320, 1280):
        envoyer({**reglages, "taille": taille})

    invalides = [{"cnn": "../cnn_final.keras"}, {"yolo": "absent.pt"}, {"taille": "abc"},
                 {"taille": 777}, {"seuil": "nan"}, {"iou": 1.5}, {"iou": "nan"}]
    for parametres in invalides:
        assert client.post("/detecter", params=parametres, content=jpeg.tobytes()).status_code == 400

    # Une seconde demande n'attend pas derrière le calcul déjà en cours.
    client.portal.call(app.state.verrou.acquire)
    try:
        assert client.post("/detecter", content=jpeg.tobytes()).status_code == 429
    finally:
        async def liberer():
            app.state.verrou.release()
        client.portal.call(liberer)
    assert client.post("/detecter", content=b"image invalide").status_code == 400
    assert client.get("/video/0", headers={"Range": "bytes=0-1023"}).status_code == 206

bilan = {"couples_testes": len(catalogue["cnn"]) * len(catalogue["yolo"]),
         "inference_identique_appel_direct": True, "modeles_reutilises_sans_rechargement": True,
         "reglages_invalides_refuses": len(invalides), "calcul_concurrent_refuse": True,
         "lecture_partielle_video": True, "essais": resultats}
(DOSSIER / "resultats/reglages_api.json").write_text(json.dumps(bilan, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({k:v for k,v in bilan.items() if k != "essais"}, ensure_ascii=False, indent=2))
