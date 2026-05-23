"""
Téléchargement automatique des fichiers depuis Google Drive.
Appelé une seule fois au démarrage de l'app.
"""
import os
import glob
import zipfile
import streamlit as st
# telcharger
# ── IDs Google Drive ──────────────────────────────────────────────────────────
GDRIVE_FILES = {
    "models/global_lstm.keras":  "1NzDUOtwSHyduaeKx0ICFFSmttQlMxKh4",
    "models/global_gru.keras":   "16As511yfxgvRZyr2aIp8WWynlCO3l6u3",
    "models/scaler.pkl":         "1khCCzNWchuQXjR6qaKENqXiqI2Ftrpqx",
}
PLOTS_ZIP_ID   = "1O4qfVUw7S1rRQ6icIjUddC3dno161qaH"
DATA_ZIP_ID    = "1dd-dtROZB6kEoZmoz0XWqHSo5GR_kLjn"
PLOTS_ZIP_PATH = "models/plots_local_models.zip"
DATA_ZIP_PATH  = "data/hsv_dataset.zip"
PLOTS_DIR      = "models/plots"
DATA_SENTINEL  = "data/.downloaded"


def _download_file(file_id: str, dest: str, label: str):
    import gdown
    os.makedirs(
        os.path.dirname(dest) if os.path.dirname(dest) else ".",
        exist_ok=True
    )
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url=url, output=dest, quiet=False)


def _unzip(zip_path: str, extract_to: str):
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)
    os.remove(zip_path)


@st.cache_resource(show_spinner=False)
def download_all():
    errors = []

    # ── 1. Modèles individuels ─────────────────────────────────────────
    for dest, file_id in GDRIVE_FILES.items():
        if not os.path.exists(dest):
            try:
                st.info(f"⬇️ Téléchargement {os.path.basename(dest)}…")
                _download_file(file_id, dest, os.path.basename(dest))
                st.success(f"✅ {os.path.basename(dest)} téléchargé")
            except Exception as e:
                errors.append(f"{dest}: {e}")

    # ── 2. Modèles locaux fine-tunés (ZIP → models/plots/) ────────────
    plots_ready = os.path.isdir(PLOTS_DIR) and len(os.listdir(PLOTS_DIR)) > 0
    if not plots_ready:
        try:
            st.info("⬇️ Téléchargement modèles fine-tunés (plots)…")
            _download_file(PLOTS_ZIP_ID, PLOTS_ZIP_PATH, "plots_local_models.zip")
            st.info("📦 Décompression modèles fine-tunés…")
            _unzip(PLOTS_ZIP_PATH, PLOTS_DIR)
            st.success("✅ Modèles fine-tunés prêts")
        except Exception as e:
            errors.append(f"plots: {e}")

    # ── 3. Dataset (ZIP → data/) ──────────────────────────────────────
    if not os.path.exists(DATA_SENTINEL):
        try:
            st.info("⬇️ Téléchargement dataset HSV (peut prendre quelques minutes)…")
            _download_file(DATA_ZIP_ID, DATA_ZIP_PATH, "hsv_dataset.zip")
            st.info("📦 Décompression dataset…")
            _unzip(DATA_ZIP_PATH, "data/")
            open(DATA_SENTINEL, "w").close()
            st.success("✅ Dataset téléchargé et extrait")
        except Exception as e:
            errors.append(f"data: {e}")

    # ── DEBUG : affiche les parquets trouvés (sidebar) ─────────────────
    parquets = glob.glob("data/**/*.parquet", recursive=True)
    st.sidebar.info(f"📁 {len(parquets)} parquets trouvés")
    if parquets:
        for p in parquets[:5]:
            st.sidebar.caption(p)
    else:
        all_files = glob.glob("data/**/*", recursive=True)
        st.sidebar.warning("⚠️ Aucun parquet ! Contenu de data/ :")
        for f in all_files[:10]:
            st.sidebar.caption(f)

    # ──────────────────────────────────────────────────────────────────
    if errors:
        for err in errors:
            st.error(f"❌ Erreur : {err}")
        return False
    return True
