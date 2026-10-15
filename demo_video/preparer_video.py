"""Télécharge une séquence de conduite et en fait une vidéo H.264."""

import argparse
from pathlib import Path
import tarfile
from urllib.request import urlretrieve

import cv2
import numpy as np


DOSSIER = Path(__file__).resolve().parent / "data"
URL = "https://btsd.ethz.ch/shareddata/BelgiumTS/Seqs/Seq03_00.tar"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debut", type=int, default=0, help="Première image dans la séquence triée")
    parser.add_argument("--images", type=int, default=600)
    parser.add_argument("--fps", type=float, default=10, help="Cadence de lecture choisie")
    args = parser.parse_args()
    if args.debut < 0 or args.images < 1 or args.fps <= 0:
        parser.error("Le début doit être positif ou nul, le nombre d'images et les FPS strictement positifs.")

    DOSSIER.mkdir(exist_ok=True)
    archive = DOSSIER / "Seq03_00.tar"
    if not archive.exists():
        print("Téléchargement de BelgiumTS, séquence 03, caméra avant gauche (605 Mo)...", flush=True)
        temporaire = archive.with_suffix(".part")
        urlretrieve(URL, temporaire)
        temporaire.rename(archive)

    sortie = DOSSIER / f"belgique_{args.debut:04d}.mp4"
    video = None
    try:
        with tarfile.open(archive) as contenu:
            # L'archive n'est pas rangée chronologiquement : on trie les noms.
            images = sorted((m for m in contenu if m.isfile() and m.name.endswith(".jp2")), key=lambda m: m.name)
            selection = images[args.debut:args.debut + args.images]
            if not selection:
                parser.error("Aucune image à cette position.")
            for i, membre in enumerate(selection):
                with contenu.extractfile(membre) as fichier:
                    image = cv2.imdecode(np.frombuffer(fichier.read(), np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f"Image illisible : {membre.name}")
                if video is None:
                    hauteur, largeur = image.shape[:2]
                    video = cv2.VideoWriter(str(sortie), cv2.VideoWriter_fourcc(*"avc1"), args.fps, (largeur, hauteur))
                    if not video.isOpened():
                        raise RuntimeError("L'encodeur H.264 n'est pas disponible dans OpenCV.")
                video.write(image)
                if (i + 1) % 100 == 0:
                    print(f"{i + 1}/{len(selection)} images", flush=True)
        print(f"{sortie} — {len(selection)} images, {len(selection) / args.fps:.1f} s à {args.fps:g} FPS")
    finally:
        if video is not None:
            video.release()


if __name__ == "__main__":
    main()
