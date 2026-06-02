import gdown
import os
import zipfile

def download_if_needed():

    # ══════════════════════════════
    # DATA
    # ══════════════════════════════
    if not os.path.exists("data/lstm_final_clean"):
        os.makedirs("data", exist_ok=True)
        print("⬇️ Téléchargement lstm_final_clean.zip...")
        gdown.download(
            id="1bhldEXv0WgPrr2Phw3a9dH93564Ogu6s",
            output="data/lstm_final_clean.zip",
            quiet=False
        )
        print("📦 Extraction lstm_final_clean...")
        with zipfile.ZipFile("data/lstm_final_clean.zip", "r") as z:
            z.extractall("data/")
        os.remove("data/lstm_final_clean.zip")

    if not os.path.exists("data/dataset_model2_1999_2023_clean"):
        os.makedirs("data", exist_ok=True)
        print("⬇️ Téléchargement dataset_model2.zip...")
        gdown.download(
            id="1TEqa7Jk6a5TS_cyZHTon9QPQrv4c0EvZ",
            output="data/dataset_model2.zip",
            quiet=False
        )
        print("📦 Extraction dataset_model2...")
        with zipfile.ZipFile("data/dataset_model2.zip", "r") as z:
            z.extractall("data/")
        os.remove("data/dataset_model2.zip")

    # ══════════════════════════════
    # MODELS
    # ══════════════════════════════
    if not os.path.exists("models/global_lstm.keras"):
        os.makedirs("models", exist_ok=True)
        print("⬇️ Téléchargement global_lstm.keras...")
        gdown.download(
            id="1NzDUOtwSHyduaeKx0ICFFSmttQlMxKh4",
            output="models/global_lstm.keras",
            quiet=False
        )

    if not os.path.exists("models/scaler.pkl"):
        os.makedirs("models", exist_ok=True)
        print("⬇️ Téléchargement scaler.pkl...")
        gdown.download(
            id="1khCCzNWchuQXjR6qaKENqXiqI2Ftrpqx",
            output="models/scaler.pkl",
            quiet=False
        )

    if not os.path.exists("models/plots"):
        os.makedirs("models", exist_ok=True)
        print("⬇️ Téléchargement plots_local_models.zip...")
        gdown.download(
            id="1O4qfVUw7S1rRQ6icIjUddC3dno161qaH",
            output="models/plots_local_models.zip",
            quiet=False
        )
        print("📦 Extraction plots_local_models...")
        with zipfile.ZipFile("models/plots_local_models.zip", "r") as z:
            z.extractall("models/")
        os.remove("models/plots_local_models.zip")

    # ══════════════════════════════
    # MODELS M2
    # ══════════════════════════════
    if not os.path.exists("models_m2/global_lstm_aqua.keras"):
        os.makedirs("models_m2", exist_ok=True)
        print("⬇️ Téléchargement global_lstm_aqua.keras...")
        gdown.download(
            id="1_3Zs9y0cM2shbDUWqQqd-ruU1lTp47An",
            output="models_m2/global_lstm_aqua.keras",
            quiet=False
        )

    if not os.path.exists("models_m2/global_lstm_dessal.keras"):
        os.makedirs("models_m2", exist_ok=True)
        print("⬇️ Téléchargement global_lstm_dessal.keras...")
        gdown.download(
            id="1y46nJ_AWy8uyF5O-xV2rhqQvnuVJyxhS",
            output="models_m2/global_lstm_dessal.keras",
            quiet=False
        )

    if not os.path.exists("models_m2/scaler_aqua.pkl"):
        os.makedirs("models_m2", exist_ok=True)
        print("⬇️ Téléchargement scaler_aqua.pkl...")
        gdown.download(
            id="1qGjXWAP57XATsOddE_EGL5pUQIKcp7K-",
            output="models_m2/scaler_aqua.pkl",
            quiet=False
        )

    if not os.path.exists("models_m2/scaler_dessal.pkl"):
        os.makedirs("models_m2", exist_ok=True)
        print("⬇️ Téléchargement scaler_dessal.pkl...")
        gdown.download(
            id="1WvA9trcaiNFhrTscboXzu7jsZROBO5No",
            output="models_m2/scaler_dessal.pkl",
            quiet=False
        )

    if not os.path.exists("models_m2/plots"):
        os.makedirs("models_m2", exist_ok=True)
        print("⬇️ Téléchargement plots.rar...")
        gdown.download(
            id="1bIE6k-XlZl24zGYJfs2FMaOOk1TaJ9es",
            output="models_m2/plots.rar",
            quiet=False
        )
        print("📦 Extraction plots.rar...")
        try:
            import rarfile
            with rarfile.RarFile("models_m2/plots.rar") as r:
                r.extractall("models_m2/")
            os.remove("models_m2/plots.rar")
        except Exception as e:
            print(f"⚠️ RAR non extrait : {e}")

    print("✅ Tous les fichiers sont prêts !")
