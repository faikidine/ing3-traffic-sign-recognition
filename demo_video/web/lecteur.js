const $ = (id) => document.getElementById(id);
const video = $("video");
const canvas = $("boites");
const dessin = canvas.getContext("2d");
const capture = document.createElement("canvas");
const pixels = capture.getContext("2d");
let pret = false, occupe = false, version = 0, derniereAnalyse = -1;
let dernierEnvoi = 0, instantResultat = -1, detections = [], historique = [], terminees = [];
let vueHistorique = false, fichierLocal = null;
let mesures = [], apercus = [], dernierApercu = -Infinity;
let catalogue = null, config = null, bloque = false;
const cleReglages = "panneaux-reglages-v1";

function afficherConfiguration() {
  if (!catalogue || !config) return;
  const nom = genre => catalogue[genre].find(m => m.id === config[genre])?.nom || config[genre];
  $("etat").textContent = `${nom("yolo")} / ${nom("cnn")} · ${config.taille} px`;
}

function remplirReglages(valeurs) {
  for (const cle of ["cnn", "yolo", "taille", "seuil", "iou", "seuil_cnn", "cadence"]) $(cle).value = valeurs[cle];
  for (const cle of ["seuil", "iou", "seuil_cnn"]) $("valeur-" + cle).textContent = Number($(cle).value).toFixed(2);
  $("boucle").checked = valeurs.boucle;
}

function appliquerReglages(valeurs) {
  config = {...valeurs};
  video.loop = config.boucle;
  bloque = false;
  historique = []; apercus = [];
  $("apercus").hidden = true;
  $("vignettes").replaceChildren();
  invalider();
  erreur();
  afficherConfiguration();
  try { localStorage.setItem(cleReglages, JSON.stringify(config)); } catch {}
  analyser();
}

async function chargerReglages() {
  const reponse = await fetch("/modeles");
  if (!reponse.ok) throw new Error("Impossible de charger les modèles disponibles.");
  catalogue = await reponse.json();
  for (const genre of ["cnn", "yolo"]) {
    for (const m of catalogue[genre]) $(genre).append(new Option(m.nom, m.id));
  }
  for (const taille of catalogue.tailles) $("taille").append(new Option(taille, taille));
  if (!catalogue.cnn.length || !catalogue.yolo.length) throw new Error("Il manque un modèle CNN ou YOLO dans models/.");
  let sauvegarde = {};
  try { sauvegarde = JSON.parse(localStorage.getItem(cleReglages)) || {}; } catch {}
  // Les choix enregistrés ne doivent pas pointer vers un fichier retiré depuis.
  remplirReglages(catalogue.defaut);
  for (const cle of ["cnn", "yolo", "taille", "cadence"]) {
    if ([...$(cle).options].some(o => o.value === String(sauvegarde[cle]))) $(cle).value = sauvegarde[cle];
  }
  for (const cle of ["seuil", "iou", "seuil_cnn"]) {
    if (typeof sauvegarde[cle] === "number" && Number.isFinite(sauvegarde[cle])) $(cle).value = sauvegarde[cle];
  }
  if (typeof sauvegarde.boucle === "boolean") $("boucle").checked = sauvegarde.boucle;
  appliquerReglages(lireReglages());
  $("reglages").disabled = false;
}

function lireReglages() {
  return {cnn: $("cnn").value, yolo: $("yolo").value, taille: Number($("taille").value),
    seuil: Number($("seuil").value), iou: Number($("iou").value), seuil_cnn: Number($("seuil_cnn").value),
    cadence: Number($("cadence").value), boucle: $("boucle").checked};
}

function temps(secondes) {
  if (!Number.isFinite(secondes)) return "0:00";
  return `${Math.floor(secondes / 60)}:${String(Math.floor(secondes % 60)).padStart(2, "0")}`;
}

function erreur(message = "") {
  $("erreur").textContent = message;
  $("erreur").hidden = !message;
}

function tableau() {
  const lignes = vueHistorique ? historique : detections;
  const cible = $("resultats");
  cible.replaceChildren();
  if (!lignes.length) {
    const texte = document.createElement("p");
    texte.textContent = vueHistorique ? "Pas encore de détection." : "Aucun panneau détecté.";
    cible.append(texte);
    return;
  }
  const table = document.createElement("table");
  const entete = table.createTHead().insertRow();
  for (const titre of (vueHistorique ? ["Temps", "Panneau", "YOLO", "CNN"] : ["Panneau", "YOLO", "CNN"])) {
    const cellule = document.createElement("th");
    cellule.textContent = titre;
    entete.append(cellule);
  }
  const corps = table.createTBody();
  for (const d of lignes) {
    const valeurs = [d.cnn >= config.seuil_cnn ? d.nom : "Catégorie incertaine", `${(d.yolo * 100).toFixed(0)} %`, `${(d.cnn * 100).toFixed(0)} %`];
    if (vueHistorique) valeurs.unshift(temps(d.instant));
    const ligne = corps.insertRow();
    for (const valeur of valeurs) ligne.insertCell().textContent = valeur;
  }
  cible.append(table);
}

function completerResultats(instant) {
  mesures.push({instant, nombre: detections.length});
  mesures = mesures.slice(-30);
  const maximum = Math.max(1, ...mesures.map(m => m.nombre));
  const premier = mesures[0].instant, dernier = mesures.at(-1).instant;
  const points = mesures.map(m => `${24 + 286 * (m.instant - premier) / (dernier - premier || 1)},${56 - 48 * m.nombre / maximum}`);
  $("courbe").setAttribute("points", points.join(" "));
  $("maximum").textContent = maximum;
  $("temps-debut").textContent = temps(premier);
  $("temps-fin").textContent = temps(dernier);
  $("graphe").hidden = mesures.length < 2;

  // La capture est toujours celle envoyée au modèle, pas une image lue plus tard.
  if (!detections.length || instant - dernierApercu < 1) return;
  dernierApercu = instant;
  for (const d of detections.slice(0, 2)) {
    const [x1, y1, x2, y2] = d.boite;
    const zone = document.createElement("canvas");
    const largeur = (x2 - x1) * capture.width, hauteur = (y2 - y1) * capture.height;
    if (largeur < 1 || hauteur < 1) continue;
    const ratio = Math.min(1, 120 / Math.max(largeur, hauteur));
    zone.width = Math.max(1, Math.round(largeur * ratio));
    zone.height = Math.max(1, Math.round(hauteur * ratio));
    zone.getContext("2d").drawImage(capture, x1 * capture.width, y1 * capture.height, largeur, hauteur, 0, 0, zone.width, zone.height);
    apercus.unshift({...d, instant, image: zone.toDataURL("image/jpeg", .9)});
  }
  apercus = apercus.slice(0, 2);
  $("vignettes").replaceChildren();
  for (const d of apercus) {
    const figure = document.createElement("figure");
    const image = document.createElement("img");
    image.src = d.image;
    image.alt = `Zone détectée à ${temps(d.instant)}`;
    const legende = document.createElement("figcaption");
    const nom = document.createElement("strong");
    nom.textContent = d.cnn >= config.seuil_cnn ? d.nom : "Catégorie incertaine";
    nom.title = nom.textContent;
    legende.append(nom, `${temps(d.instant)} · CNN ${(d.cnn * 100).toFixed(0)} %`);
    figure.append(image, legende);
    $("vignettes").append(figure);
  }
  $("apercus").hidden = !apercus.length;
}

function dessiner() {
  const largeur = canvas.clientWidth, hauteur = canvas.clientHeight;
  dessin.clearRect(0, 0, largeur, hauteur);
  dessin.lineWidth = 2;
  dessin.font = "12px system-ui";
  for (const d of detections) {
    const [x1, y1, x2, y2] = d.boite;
    const x = x1 * largeur, y = y1 * hauteur;
    dessin.strokeStyle = dessin.fillStyle = d.cnn >= config.seuil_cnn ? "#227c3a" : "#9c5c00";
    dessin.strokeRect(x, y, (x2 - x1) * largeur, (y2 - y1) * hauteur);
    const texte = `${d.cnn >= config.seuil_cnn ? d.nom : "Catégorie incertaine"} · ${(d.cnn * 100).toFixed(0)} %`;
    const w = Math.min(largeur, dessin.measureText(texte).width + 10);
    const tx = Math.max(0, Math.min(x, largeur - w));
    const ty = y >= 22 ? y - 22 : y;
    dessin.fillRect(tx, ty, w, 20);
    dessin.fillStyle = "white";
    dessin.fillText(texte, tx + 5, ty + 14, w - 10);
  }
}

function redimensionner() {
  if (!video.videoWidth) return;
  const scene = $("scene");
  const ratio = Math.min(scene.clientWidth / video.videoWidth, scene.clientHeight / video.videoHeight);
  const w = video.videoWidth * ratio, h = video.videoHeight * ratio;
  Object.assign(canvas.style, {width: `${w}px`, height: `${h}px`, left: `${(scene.clientWidth - w) / 2}px`, top: `${(scene.clientHeight - h) / 2}px`});
  const densite = window.devicePixelRatio || 1;
  canvas.width = Math.round(w * densite);
  canvas.height = Math.round(h * densite);
  dessin.setTransform(densite, 0, 0, densite, 0, 0);
  dessiner();
}
new ResizeObserver(redimensionner).observe($("scene"));

function invalider() {
  // Une réponse de l'ancien instant ne doit pas apparaître après un déplacement.
  version++;
  derniereAnalyse = -1;
  detections = [];
  terminees = [];
  mesures = [];
  $("graphe").hidden = true;
  dernierApercu = -Infinity;
  dessiner();
  tableau();
  $("mesures").textContent = "En attente d’analyse.";
}

async function analyser() {
  if (!pret || !config || bloque || occupe || video.seeking || video.readyState < 2 || document.hidden) return;
  const instant = video.currentTime;
  if (Math.abs(instant - derniereAnalyse) < .001) return;
  const maintenant = performance.now();
  if (!video.paused && maintenant - dernierEnvoi < 1000 / config.cadence) return;
  occupe = true;
  dernierEnvoi = maintenant;
  const demande = version;
  // Un seul calcul en cours. Les images intermédiaires continuent à être lues.
  const ratio = Math.min(1, 1280 / Math.max(video.videoWidth, video.videoHeight));
  const w = Math.round(video.videoWidth * ratio), h = Math.round(video.videoHeight * ratio);
  if (capture.width !== w || capture.height !== h) { capture.width = w; capture.height = h; }
  try {
    pixels.drawImage(video, 0, 0, w, h);
    const jpeg = await new Promise((resolve) => capture.toBlob(resolve, "image/jpeg", .88));
    if (demande !== version || !jpeg) return;
    const parametres = new URLSearchParams({cnn: config.cnn, yolo: config.yolo,
      seuil: config.seuil, taille: config.taille, iou: config.iou});
    const reponse = await fetch(`/detecter?${parametres}`, {
      method: "POST", headers: {"Content-Type": "image/jpeg"}, body: jpeg,
      signal: AbortSignal.timeout(30000),
    });
    if (reponse.status === 429) return;
    const resultat = await reponse.json();
    if (demande !== version) return;
    if (!reponse.ok) {
      if (reponse.status === 400) bloque = true;
      throw new Error(resultat.erreur || "Calcul indisponible.");
    }
    derniereAnalyse = instant;
    // On écarte une réponse devenue trop ancienne au lieu de décaler les boîtes.
    if (Math.abs(video.currentTime - instant) > (video.paused ? .1 : .45 * video.playbackRate)) return;
    erreur();
    instantResultat = instant;
    detections = resultat.detections;
    for (const d of detections) historique.unshift({...d, instant});
    historique = historique.slice(0, 40);
    const fin = performance.now();
    terminees.push(fin);
    terminees = terminees.filter(t => fin - t < 2000);
    const cadence = terminees.length > 1 ? ((terminees.length - 1) * 1000 / (fin - terminees[0])).toFixed(1) : "—";
    $("mesures").textContent = `${detections.length} panneau${detections.length === 1 ? "" : "x"} · ${Math.round(fin - maintenant)} ms · ${video.paused ? "Pause" : `${cadence} analyses/s`}`;
    dessiner();
    tableau();
    completerResultats(instant);
  } catch (e) {
    if (demande === version) erreur(`Détection : ${e.message}`);
  } finally {
    occupe = false;
    // Un déplacement pendant un calcul doit aussi être analysé lorsque la vidéo est en pause.
    if (demande !== version || (video.paused && Math.abs(video.currentTime - derniereAnalyse) > .001)) {
      setTimeout(analyser, 100);
    }
  }
}

function actualiser() {
  $("temps").textContent = `${temps(video.currentTime)} / ${temps(video.duration)}`;
  if (!video.paused && detections.length && Math.abs(video.currentTime - instantResultat) > .45 * video.playbackRate) {
    detections = []; dessiner(); tableau();
  }
  analyser();
}
if ("requestVideoFrameCallback" in video) {
  const imageDisponible = () => { actualiser(); video.requestVideoFrameCallback(imageDisponible); };
  video.requestVideoFrameCallback(imageDisponible);
} else {
  const imageDisponible = () => { if (!video.paused) actualiser(); requestAnimationFrame(imageDisponible); };
  requestAnimationFrame(imageDisponible);
}

function choisir(url) {
  video.pause();
  invalider();
  historique = [];
  apercus = [];
  $("apercus").hidden = true;
  $("vignettes").replaceChildren();
  tableau();
  erreur();
  for (const id of ["lire", "reculer", "avancer", "debut", "plein-ecran"]) $(id).disabled = true;
  if (url) video.src = url;
  else video.removeAttribute("src");
  video.load();
  $("temps").textContent = "0:00 / 0:00";
  $("attente").hidden = !!url;
}

$("source").addEventListener("change", () => choisir($("source").value));
$("fichier").addEventListener("change", () => {
  const fichier = $("fichier").files[0];
  if (!fichier) return;
  if (fichierLocal) URL.revokeObjectURL(fichierLocal);
  fichierLocal = URL.createObjectURL(fichier);
  $("source").querySelector('[data-local]')?.remove();
  const option = new Option(fichier.name, fichierLocal, true, true);
  option.dataset.local = "true";
  $("source").append(option);
  choisir(fichierLocal);
  $("fichier").value = "";
});
$("lire").onclick = async () => {
  if (video.paused) { try { await video.play(); } catch (e) { erreur(e.message); } }
  else video.pause();
};
function deplacer(delta) {
  if (Number.isFinite(video.duration)) video.currentTime = Math.max(0, Math.min(video.duration, video.currentTime + delta));
}
$("reculer").onclick = () => deplacer(-5);
$("avancer").onclick = () => deplacer(5);
$("debut").onclick = () => { video.currentTime = 0; };
$("plein-ecran").onclick = async () => {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await document.querySelector(".lecteur").requestFullscreen();
  } catch (e) { erreur(e.message); }
};
video.addEventListener("play", () => { $("lire").textContent = "Pause"; terminees = []; });
video.addEventListener("pause", () => { $("lire").textContent = "Lire"; actualiser(); });
video.addEventListener("seeking", invalider);
video.addEventListener("seeked", actualiser);
video.addEventListener("timeupdate", () => { $("temps").textContent = `${temps(video.currentTime)} / ${temps(video.duration)}`; });
video.addEventListener("loadedmetadata", () => {
  for (const id of ["lire", "reculer", "avancer", "debut", "plein-ecran"]) $(id).disabled = false;
  redimensionner();
  $("temps").textContent = `0:00 / ${temps(video.duration)}`;
});
video.addEventListener("loadeddata", actualiser);
video.addEventListener("error", () => erreur("Cette vidéo ne peut pas être lue. Essayer un MP4 H.264."));
$("reglages").onclick = () => {
  remplirReglages(config);
  $("options").showModal();
};
for (const cle of ["seuil", "iou", "seuil_cnn"]) $(cle).oninput = () => {
  $("valeur-" + cle).textContent = Number($(cle).value).toFixed(2);
};
$("annuler").onclick = () => $("options").close();
$("defaut").onclick = () => remplirReglages(catalogue.defaut);
$("form-reglages").onsubmit = (evenement) => {
  evenement.preventDefault();
  appliquerReglages(lireReglages());
  $("options").close();
};
for (const id of ["actuel", "historique"]) $(id).onclick = () => {
  vueHistorique = id === "historique";
  $("actuel").setAttribute("aria-selected", String(!vueHistorique));
  $("historique").setAttribute("aria-selected", String(vueHistorique));
  $("resultats").setAttribute("aria-label", vueHistorique ? "Historique" : "Résultats");
  tableau();
};
document.addEventListener("visibilitychange", () => { if (!document.hidden) { invalider(); analyser(); } });

async function verifierModeles() {
  try {
    const reponse = await fetch("/etat");
    const etat = await reponse.json();
    pret = etat.pret;
    $("etat").textContent = etat.erreur ? "Modèles indisponibles" : pret ? "YOLO11n + CNN · prêt" : "Chargement des modèles…";
    if (etat.erreur) { erreur(etat.erreur); return; }
    if (pret) { afficherConfiguration(); analyser(); return; }
  } catch {
    $("etat").textContent = "Connexion au serveur…";
  }
  setTimeout(verifierModeles, 1000);
}
fetch("/videos").then(r => r.json()).then(videos => {
  for (const v of videos) $("source").append(new Option(v.nom, v.url));
  if (videos.length) { $("source").value = videos[0].url; choisir(videos[0].url); }
}).catch(() => erreur("Impossible de charger la liste des vidéos."));
chargerReglages().then(verifierModeles).catch(e => erreur(e.message));
