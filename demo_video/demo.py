"""Affiche les panneaux détectés dans une vidéo ou un flux H.264."""

import argparse
import csv
import json
import math
from pathlib import Path
import time
import unicodedata

import cv2
from configuration import catalogue, reglages_defaut


DOSSIER = Path(__file__).resolve().parent
FENETRE = "Vision routiere - YOLO + CNN"


def texte(image, contenu, position, couleur=(255, 255, 255), taille=0.65):
    # La police OpenCV classique n'affiche pas les accents.
    contenu = unicodedata.normalize("NFKD", contenu).encode("ascii", "ignore").decode()
    cv2.putText(image, contenu, position, cv2.FONT_HERSHEY_SIMPLEX, taille, (15, 15, 15), 4, cv2.LINE_AA)
    cv2.putText(image, contenu, position, cv2.FONT_HERSHEY_SIMPLEX, taille, couleur, 1, cv2.LINE_AA)


def annoter(image, detections):
    affichage = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = map(round, detection["boite"])
        couleur = (90, 230, 110) if detection["cnn"] >= 0.65 else (0, 180, 255)
        cv2.rectangle(affichage, (x1, y1), (x2, y2), couleur, 2)
        nom = detection["nom"] if detection["cnn"] >= 0.65 else "Panneau : categorie incertaine"
        etiquette = f"{nom} | Y {detection['yolo']:.0%} / C {detection['cnn']:.0%}"
        largeur = cv2.getTextSize(etiquette, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)[0][0]
        x = max(5, min(x1, image.shape[1] - largeur - 5))
        y = max(25, y1 - 10)
        texte(affichage, etiquette, (x, y), couleur)
    return affichage


def main():
    defaut = reglages_defaut()
    modeles = catalogue()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="*", help="Vidéos locales ou URL de flux (UDP, RTSP, HTTP)")
    parser.add_argument("--boucle", action="store_true", help="Rejouer les vidéos à la fin")
    parser.add_argument("--yolo", choices=[m["id"] for m in modeles["yolo"]], default=defaut["yolo"])
    parser.add_argument("--cnn", choices=[m["id"] for m in modeles["cnn"]], default=defaut["cnn"])
    parser.add_argument("--seuil", type=float, default=defaut["seuil"], help="Seuil YOLO")
    parser.add_argument("--iou", type=float, default=defaut["iou"], help="IoU de suppression des doublons")
    parser.add_argument("--taille", type=int, default=defaut["taille"], help="Taille d'entrée de YOLO")
    parser.add_argument("--sortie", type=Path, help="Enregistrer la vidéo annotée en MP4 H.264")
    parser.add_argument("--journal", type=Path, help="Conserver les prédictions image par image en CSV")
    parser.add_argument("--sans-affichage", action="store_true")
    parser.add_argument("--max-images", type=int, default=0, help="Arrêter après N images (0 : tout)")
    args = parser.parse_args()
    if not 0 < args.seuil <= 1 or not 0 < args.iou <= 1 or args.taille < 32 or args.max_images < 0:
        parser.error("Seuil et IoU attendus entre 0 et 1, taille >= 32 et max-images >= 0.")
    sources = args.sources or [str(p) for p in sorted((DOSSIER / "data").glob("*.mp4"))]
    if not sources:
        parser.error("Aucune vidéo. Lancez preparer_video.py ou indiquez un fichier vidéo.")
    if args.sortie and (len(sources) != 1 or args.boucle):
        parser.error("L'enregistrement se fait sur une seule source, sans --boucle.")
    if args.sortie and "://" not in sources[0] and args.sortie.resolve() == Path(sources[0]).resolve():
        parser.error("La sortie doit être différente de la vidéo source.")

    from panneaux import charger_modeles, reconnaitre

    detecteur, cnn, device = charger_modeles(args.yolo, args.cnn)
    journal = None
    capture = None
    sortie = None
    total_images = total_detections = 0
    duree_traitement = 0.0
    arreter = False
    if args.journal:
        args.journal.parent.mkdir(parents=True, exist_ok=True)
        journal = args.journal.open("w", newline="", encoding="utf-8")
        tableau = csv.writer(journal)
        tableau.writerow(["source", "image", "temps_video_s", "traitement_ms", "nombre_panneaux", "detections"])

    try:
        while not arreter:
            for source in sources:
                flux = "://" in source
                capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG, [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC, 4000,
                ])
                if not capture.isOpened():
                    raise RuntimeError(f"Impossible de lire : {source}")
                fps_source = capture.get(cv2.CAP_PROP_FPS)
                if not math.isfinite(fps_source) or fps_source <= 0:
                    fps_source = 10.0
                indice = 0
                fps_lisse = None
                debut_source = time.perf_counter()
                print(f"Lecture : {source}", flush=True)
                if not args.sans_affichage:
                    cv2.namedWindow(FENETRE, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(FENETRE, 1100, 800)

                while True:
                    ok, image = capture.read()
                    if not ok:
                        if indice == 0:
                            raise RuntimeError(f"Aucune image décodée : {source}")
                        if flux:
                            print("Flux terminé ou interrompu.", flush=True)
                        break
                    debut = time.perf_counter()
                    detections = reconnaitre(image, detecteur, cnn, device, args.seuil, args.taille, args.iou)
                    duree = time.perf_counter() - debut
                    duree_traitement += duree
                    fps_lisse = 1 / duree if fps_lisse is None else 0.9 * fps_lisse + 0.1 / duree
                    seconde = time.perf_counter() - debut_source if flux else indice / fps_source
                    annotee = annoter(image, detections)
                    if args.sortie and sortie is None:
                        args.sortie.parent.mkdir(parents=True, exist_ok=True)
                        hauteur, largeur = image.shape[:2]
                        sortie = cv2.VideoWriter(str(args.sortie), cv2.VideoWriter_fourcc(*"avc1"), fps_source, (largeur, hauteur))
                        if not sortie.isOpened():
                            raise RuntimeError("Impossible de créer la vidéo H.264.")
                    if sortie is not None:
                        sortie.write(annotee)
                    if journal:
                        tableau.writerow([source, indice, round(seconde, 3), round(duree * 1000, 2), len(detections),
                                          json.dumps(detections, ensure_ascii=False)])
                    total_images += 1
                    total_detections += len(detections)
                    indice += 1
                    if args.max_images and total_images >= args.max_images:
                        arreter = True
                        break
                    if not args.sans_affichage:
                        cv2.imshow(FENETRE, annotee)
                        # Une vidéo locale suit sa cadence ; un flux arrive déjà au fil de l'eau.
                        attente = 1 if flux else max(1, round(1000 * (1 / fps_source - (time.perf_counter() - debut))))
                        touche = cv2.waitKey(attente) & 0xFF
                        if touche == ord(" "):
                            touche = cv2.waitKey(0) & 0xFF
                        if touche in (ord("q"), 27):
                            arreter = True
                            break
                        if touche == ord("n"):
                            break
                capture.release()
                if arreter:
                    break
            if not args.boucle:
                break
    finally:
        if capture is not None:
            capture.release()
        if sortie is not None:
            sortie.release()
        if journal:
            journal.close()
        if not args.sans_affichage:
            cv2.destroyAllWindows()

    fps_moyen = total_images / duree_traitement if duree_traitement else 0
    print(f"{total_images} images traitées, {total_detections} détections cumulées, {fps_moyen:.1f} FPS de traitement.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except (RuntimeError, FileNotFoundError) as erreur:
        raise SystemExit(str(erreur))
