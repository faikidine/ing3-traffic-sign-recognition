"""Lecture vidéo et inférence image par image dans Streamlit."""

from collections import deque
from pathlib import Path
import time

import cv2
import pandas as pd
import streamlit as st

from demo import annoter
from panneaux import charger_modeles, reconnaitre


DOSSIER = Path(__file__).resolve().parent
st.set_page_config(page_title="Détection de panneaux", layout="wide")
# La vidéo suit la hauteur de la fenêtre, sans déformer ses proportions.
st.markdown("""<style>
    [data-testid="stMain"] { overflow-x: hidden; }
    [data-testid="stMainBlockContainer"] {
        padding: 3rem 1.25rem 0.5rem;
        max-width: none;
    }
    [data-testid="stSidebarUserContent"] { padding: 0.5rem 1rem; }
    [data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] { gap: 0.6rem; }
    [data-testid="stSidebarHeader"] { padding: 0.5rem 1rem; }
    .st-key-video [data-testid="stImage"] { width: 100%; }
    .st-key-video [data-testid="stImageContainer"] {
        height: max(60px, calc(100dvh - 310px));
        width: 100%;
    }
    .st-key-video img {
        width: 100% !important;
        height: 100%;
        object-fit: contain;
        object-position: top left;
    }
    .st-key-resultats [data-testid="stVerticalBlock"] { gap: 0.4rem; }
    .st-key-video [data-testid="stVerticalBlock"] { gap: 0.3rem; }
    .st-key-commandes [data-testid="stHorizontalBlock"],
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] { flex-wrap: nowrap; gap: 0.3rem; }
    .st-key-commandes [data-testid="stColumn"],
    [data-testid="stSidebar"] [data-testid="stColumn"] { min-width: 0; flex: 1 1 0 !important; }
    .st-key-commandes button { padding: 0.1rem 0.25rem; min-height: 2rem; }
    .st-key-resultats [data-testid="stDataFrame"] {
        max-height: max(100px, calc(100dvh - 300px));
        overflow: auto;
    }
    .st-key-resultats [data-testid="stElementContainer"]:has([data-testid="stDataFrame"]),
    .st-key-resultats [data-testid="stFullScreenFrame"]:has([data-testid="stDataFrame"]) {
        height: auto !important;
        min-height: 0;
        flex: 0 1 auto !important;
    }
    @media (min-width: 501px) {
        [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-video) { flex-wrap: nowrap; }
        [data-testid="stColumn"]:has(> div .st-key-video) { min-width: 0; flex: 2 1 0 !important; }
        [data-testid="stColumn"]:has(> div .st-key-resultats) { min-width: 0; flex: 1 1 0 !important; }
    }
    @media (max-width: 500px) {
        [data-testid="stMainBlockContainer"] { padding: 3rem 0.6rem 0.3rem; }
        .st-key-video [data-testid="stImageContainer"] { height: max(60px, calc((100dvh - 390px) * 0.52)); }
        .st-key-resultats [data-testid="stDataFrame"] { max-height: max(70px, calc((100dvh - 340px) * 0.48 - 105px)); }
    }
</style>""", unsafe_allow_html=True)


def arreter():
    if st.session_state.get("capture") is not None:
        st.session_state.capture.release()
    st.session_state.capture = None
    st.session_state.lecture = False


def reinitialiser():
    arreter()
    st.session_state.numero = 0
    st.session_state.image = None
    st.session_state.detections = []
    st.session_state.historique = deque(maxlen=30)
    st.session_state.durees = deque(maxlen=30)
    st.session_state.total = 0
    st.session_state.fps = 0.0
    st.session_state.dernier_ms = 0.0
    st.session_state.nombre_images = 0
    st.session_state.destination = None
    st.session_state.message = "PRÊT"


def choisir_position():
    etat = st.session_state
    etat.destination = round(etat.position_video * etat.fps_source)


def deplacer(nombre_images):
    etat = st.session_state
    etat.destination = etat.numero - 1 + nombre_images


if "lecture" not in st.session_state:
    reinitialiser()

st.subheader("Détection de panneaux")

with st.sidebar:
    mode = st.radio("Entrée", ["Vidéo", "Flux H.264"], horizontal=True)
    if mode == "Vidéo":
        fichier = st.file_uploader("Importer une vidéo", type=["mp4", "mov", "avi", "mkv", "h264", "ts"])
        if fichier is not None:
            if st.session_state.get("import_id") != fichier.file_id:
                imports = DOSSIER / "data" / "imports"
                imports.mkdir(parents=True, exist_ok=True)
                chemin = imports / Path(fichier.name).name
                arreter()
                chemin.write_bytes(fichier.getbuffer())
                st.session_state.import_id = fichier.file_id
                st.session_state.import_path = str(chemin)
            source = st.session_state.import_path
        else:
            videos = sorted((DOSSIER / "data").glob("*.mp4"))
            choix = st.selectbox("Séquence disponible", videos, format_func=lambda p: p.name) if videos else None
            source = str(choix) if choix else ""
        boucle = st.checkbox("Rejouer à la fin", value=True)
    else:
        source = st.text_input("URL du flux", "udp://127.0.0.1:1234?fifo_size=10000&overrun_nonfatal=1")
        boucle = False

    signature = (source, st.session_state.get("import_id") if mode == "Vidéo" and fichier is not None else None)
    if st.session_state.get("source_signature") != signature:
        reinitialiser()
        st.session_state.source_signature = signature

    a, b = st.columns(2)
    lire = a.button("Lire", use_container_width=True, disabled=not source)
    if b.button("Pause", use_container_width=True):
        st.session_state.lecture = False
        st.session_state.message = "PAUSE"
    a, b = st.columns(2)
    recommencer = a.button("Recommencer", use_container_width=True, disabled=not source)
    if b.button("Arrêter", use_container_width=True):
        arreter()
        st.session_state.message = "ARRÊT"
    with st.popover("Réglages"):
        seuil = st.slider("Seuil YOLO", 0.05, 0.95, 0.25, 0.05)
        taille = st.select_slider("Taille YOLO", options=[320, 640, 960], value=640)
        st.caption("Y : score de détection / C : probabilité de la catégorie")

if lire or recommencer:
    if recommencer:
        reinitialiser()
    try:
        if "modeles" not in st.session_state:
            with st.spinner("Chargement de YOLO et du CNN..."):
                st.session_state.modeles = charger_modeles()
        if st.session_state.capture is None:
            capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG, [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000,
            ])
            if not capture.isOpened():
                capture.release()
                raise RuntimeError("Impossible de lire cette source.")
            st.session_state.capture = capture
            fps = capture.get(cv2.CAP_PROP_FPS)
            st.session_state.fps_source = fps if 0 < fps <= 240 else 10.0
            st.session_state.nombre_images = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT))) if mode == "Vidéo" else 0
            st.session_state.numero = 0
        st.session_state.lecture = True
        st.session_state.message = "LECTURE"
    except Exception as erreur:
        arreter()
        st.session_state.message = f"ERREUR : {erreur}"


# Une seule image par rafraîchissement : les boutons restent disponibles.
intervalle = max(1 / st.session_state.get("fps_source", 10), 0.04)


@st.fragment(run_every=intervalle if st.session_state.lecture else None)
def lecture_video():
    etat = st.session_state
    if etat.lecture or etat.destination is not None:
        debut = time.perf_counter()
        # Après un déplacement, on recalcule l'image choisie, y compris en pause.
        if etat.destination is not None:
            if etat.capture is None:
                etat.capture = cv2.VideoCapture(source)
            cible = max(0, min(etat.nombre_images - 1, etat.destination))
            etat.capture.set(cv2.CAP_PROP_POS_FRAMES, cible)
            etat.numero = cible
            etat.destination = None
            if not etat.lecture:
                etat.message = "PAUSE"
        ok, image = etat.capture.read()
        if not ok and boucle and mode == "Vidéo":
            etat.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            etat.numero = 0
            ok, image = etat.capture.read()
        if not ok:
            arreter()
            etat.message = "FIN DU FLUX" if mode != "Vidéo" else "FIN DE LA VIDÉO"
            st.rerun()
        try:
            avant_modele = time.perf_counter()
            detections = reconnaitre(image, *etat.modeles, seuil=seuil, taille=taille)
            duree = time.perf_counter() - avant_modele
            etat.durees.append(duree)
            etat.fps = len(etat.durees) / sum(etat.durees)
            etat.seconde = etat.numero / etat.fps_source
            etat.image = annoter(image, detections)
            etat.detections = detections
            etat.total += len(detections)
            for d in detections:
                etat.historique.append({"Image": etat.numero, "Panneau": d["nom"], "Y": f"{d['yolo']:.0%}", "C": f"{d['cnn']:.0%}"})
            etat.numero += 1
            etat.dernier_ms = (time.perf_counter() - debut) * 1000
        except Exception as erreur:
            arreter()
            etat.message = f"ERREUR : {erreur}"
            st.rerun()

    st.caption(f"{etat.message.capitalize()} · Image {etat.numero}")
    gauche, droite = st.columns([2.4, 1])
    with gauche, st.container(key="video"):
        if etat.image is not None:
            # On réduit seulement l'affichage, après l'inférence sur l'image source.
            h, w = etat.image.shape[:2]
            apercu = cv2.resize(etat.image, (1000, round(h * 1000 / w))) if w > 1000 else etat.image
            st.image(apercu, channels="BGR", output_format="JPEG", width="stretch")
        else:
            st.write("Choisir une vidéo puis cliquer sur Lire.")
        if mode == "Vidéo" and etat.nombre_images > 1:
            fin = (etat.nombre_images - 1) / etat.fps_source
            etat.position_video = min(fin, max(0.0, (etat.numero - 1) / etat.fps_source))
            st.slider("Position dans la vidéo (secondes)", 0.0, float(fin),
                      step=float(1 / etat.fps_source), format="%.1f s",
                      key="position_video", on_change=choisir_position, label_visibility="collapsed")
            with st.container(key="commandes"):
                retour, precedente, suivante, avance = st.columns(4)
                retour.button("−5 s", on_click=deplacer, args=(-round(5 * etat.fps_source),), use_container_width=True)
                precedente.button("Image −1", on_click=deplacer, args=(-1,), use_container_width=True)
                suivante.button("Image +1", on_click=deplacer, args=(1,), use_container_width=True)
                avance.button("+5 s", on_click=deplacer, args=(round(5 * etat.fps_source),), use_container_width=True)
    with droite, st.container(key="resultats"):
        courant, historique = st.tabs(["Résultats", "Historique"])
        with courant:
            st.write(f"Panneaux dans l'image : {len(etat.detections)}")
            st.caption(f"Calcul : {etat.get('fps', 0):.1f} images/s · {etat.get('dernier_ms', 0):.0f} ms/image")
            if etat.detections:
                lignes = [{"Panneau": d["nom"] if d["cnn"] >= 0.65 else "Catégorie incertaine",
                           "YOLO": f"{d['yolo']:.1%}", "CNN": f"{d['cnn']:.1%}"} for d in etat.detections]
                st.dataframe(pd.DataFrame(lignes), hide_index=True, height=180, width="stretch")
            else:
                st.text("Aucun panneau détecté.")
        with historique:
            if etat.historique:
                lignes = pd.DataFrame(list(reversed(etat.historique)))
                st.dataframe(lignes, hide_index=True, height=210, width="stretch")
            st.caption(f"{etat.total} détections cumulées ; un panneau peut apparaître plusieurs fois.")


lecture_video()
