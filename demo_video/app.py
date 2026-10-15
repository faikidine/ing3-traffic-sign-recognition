"""Le navigateur lit la vidéo ; Python calcule les panneaux à la demande."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
import time

from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route
from starlette.staticfiles import StaticFiles
from configuration import catalogue, reglages_defaut, valider_reglages, TAILLES


DOSSIER = Path(__file__).resolve().parent


def charger():
    # Les imports et le premier calcul ne bloquent pas l'ouverture du lecteur.
    import numpy as np
    from panneaux import charger_modeles, reconnaitre

    modeles = charger_modeles()
    reglages = reglages_defaut()
    reconnaitre(np.zeros((640, 640, 3), dtype="uint8"), *modeles,
                taille=reglages["taille"], seuil=reglages["seuil"], iou=reglages["iou"])
    modeles[1](np.zeros((1, 32, 32, 3), dtype="float32"), training=False)
    return modeles


async def preparer(app):
    try:
        app.state.modeles = await asyncio.get_running_loop().run_in_executor(app.state.worker, charger)
    except Exception as erreur:
        app.state.erreur = str(erreur)


@asynccontextmanager
async def duree_application(app):
    app.state.modeles = None
    app.state.erreur = None
    app.state.verrou = asyncio.Lock()
    app.state.worker = ThreadPoolExecutor(max_workers=1)
    preparation = asyncio.create_task(preparer(app))
    yield
    await preparation
    app.state.worker.shutdown(wait=True)


async def accueil(request):
    return FileResponse(DOSSIER / "web/index.html", headers={"Cache-Control": "no-store"})


def videos_disponibles():
    return sorted(p for p in (DOSSIER / "data").rglob("*")
                  if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".webm", ".m4v"})


async def videos(request):
    return JSONResponse([{"nom": p.name, "url": f"/video/{i}"}
                         for i, p in enumerate(videos_disponibles())])


async def video(request):
    fichiers = videos_disponibles()
    i = request.path_params["numero"]
    if i >= len(fichiers):
        return JSONResponse({"erreur": "Vidéo introuvable."}, status_code=404)
    # Les requêtes Range permettent d'avancer sans télécharger tout le fichier.
    return FileResponse(fichiers[i])


async def etat(request):
    return JSONResponse({"pret": request.app.state.modeles is not None,
                         "erreur": request.app.state.erreur})


async def modeles_disponibles(request):
    return JSONResponse({**catalogue(), "defaut": reglages_defaut(), "tailles": TAILLES})


def analyser(contenu, reglages):
    import cv2
    import numpy as np
    from panneaux import charger_modeles, reconnaitre

    debut = time.perf_counter()
    if not contenu:
        raise ValueError("Image vide.")
    image = cv2.imdecode(np.frombuffer(contenu, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or max(image.shape[:2]) > 2560:
        raise ValueError("Image illisible ou trop grande.")
    modeles = charger_modeles(reglages["yolo"], reglages["cnn"])
    detections = reconnaitre(image, *modeles, seuil=reglages["seuil"], taille=reglages["taille"],
                             iou=reglages["iou"])
    hauteur, largeur = image.shape[:2]
    for d in detections:
        d["boite"] = [v / (largeur if i % 2 == 0 else hauteur) for i, v in enumerate(d["boite"])]
    return {"detections": detections, "calcul_ms": round((time.perf_counter() - debut) * 1000, 1),
            "configuration": reglages}


async def detection(request):
    etat = request.app.state
    if etat.modeles is None:
        return JSONResponse({"erreur": etat.erreur or "Chargement des modèles…"}, status_code=503)
    # Un deuxième lecteur réessaiera sur une image récente, sans file d'attente.
    if etat.verrou.locked():
        return JSONResponse({"erreur": "Calcul en cours."}, status_code=429)
    try:
        reglages = valider_reglages(request.query_params)
        contenu = bytearray()
        async for morceau in request.stream():
            contenu.extend(morceau)
            if len(contenu) > 3_000_000:
                return JSONResponse({"erreur": "Image trop volumineuse."}, status_code=413)
        if etat.verrou.locked():
            return JSONResponse({"erreur": "Calcul en cours."}, status_code=429)
        async with etat.verrou:
            resultat = await asyncio.get_running_loop().run_in_executor(
                etat.worker, analyser, contenu, reglages)
        return JSONResponse(resultat)
    except (ValueError, TypeError) as erreur:
        return JSONResponse({"erreur": str(erreur)}, status_code=400)
    except (OSError, KeyError) as erreur:
        return JSONResponse({"erreur": "Impossible de charger le modèle choisi : " + str(erreur)}, status_code=400)


app = Starlette(lifespan=duree_application, routes=[
    Route("/", accueil), Route("/etat", etat), Route("/modeles", modeles_disponibles), Route("/videos", videos),
    Route("/video/{numero:int}", video), Route("/detecter", detection, methods=["POST"]),
])
app.mount("/web", StaticFiles(directory=DOSSIER / "web"), name="web")
