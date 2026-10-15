"""Essai de l'import, du calcul et des commandes avec une vraie vidéo."""

import json
from pathlib import Path
import subprocess

from streamlit.testing.v1 import AppTest


DOSSIER = Path(__file__).resolve().parent
source = DOSSIER / "data/belgique_0000.mp4"
extrait = DOSSIER / "resultats/entree_test.mp4"
extrait.parent.mkdir(exist_ok=True)
subprocess.run([
    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", "9.5", "-i", str(source),
    "-t", "1", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(extrait),
], check=True)

app = AppTest.from_file(str(DOSSIER / "essai_streamlit.py"), default_timeout=30).run()


def cliquer(nom):
    next(b for b in app.button if b.label == nom).click().run()


assert not app.exception, app.exception
app.get("file_uploader")[0].upload("entree_test.mp4", extrait.read_bytes(), "video/mp4").run()
assert Path(app.session_state["import_path"]).read_bytes() == extrait.read_bytes()

try:
    cliquer("Lire")  # Les vrais modèles calculent une première image.
    assert not app.exception and app.session_state["lecture"]
    assert app.session_state["numero"] == 1
    assert app.session_state["image"] is not None
    detections = list(app.session_state["detections"])
    cliquer("Pause")  # Le numéro d'image reste inchangé.
    numero = app.session_state["numero"]
    app.run()
    assert not app.session_state["lecture"] and app.session_state["numero"] == numero
    cliquer("Image +1")
    assert app.session_state["numero"] == numero + 1
    cliquer("Image −1")
    assert app.session_state["numero"] == numero
    cliquer("+5 s")
    assert app.session_state["numero"] == app.session_state["nombre_images"]
    cliquer("−5 s")
    assert app.session_state["numero"] == 1
    app.slider(key="position_video").set_value(0.4).run()
    assert app.session_state["numero"] == round(0.4 * app.session_state["fps_source"]) + 1
    assert not app.session_state["lecture"]
    numero = app.session_state["numero"]
    cliquer("Lire")
    assert app.session_state["numero"] == numero + 1
    cliquer("Recommencer")  # Revient au début de la vidéo importée.
    assert app.session_state["numero"] == 1
    assert not app.exception, app.exception
finally:
    cliquer("Arrêter")
    assert app.session_state["capture"] is None
    assert not app.session_state["lecture"]

resultat = {"import_video": "OK", "inference_reelle": "OK", "pause": "OK",
            "reprise": "OK", "recommencer": "OK", "arret": "OK", "image_par_image": "OK",
            "avancer_reculer": "OK", "barre_de_lecture": "OK", "premiere_image": detections}
(DOSSIER / "resultats/interface.json").write_text(json.dumps(resultat, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(resultat, ensure_ascii=False, indent=2))
