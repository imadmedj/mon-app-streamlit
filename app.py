
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  DASHBOARD HSV — CÔTES ALGÉRIENNES  (VERSION FUSIONNÉE)                     ║
║  Base    : app(12) — toutes fonctionnalités complètes                        ║
║  Ajout   : Prédiction Temps Réel — ERA5/Open-Meteo + calibration            ║
║  Modèle 1 : ERA5 seul          · 1985–2023 · MESURE, wind_speed, mwp, mwd   ║
║  Modèle 2 : ERA5 + CMEMS       · 1999–2023 · + salinity, o2, spm, sst       ║
║  Dataset  : data/lstm_final_clean   (~20 M lignes)                           ║
║  Dataset2 : data/dataset_model2_1999_2023_clean  (~12 M lignes)              ║
║  Optimisé : DuckDB (SQL sur Parquet)                                         ║
║  HORIZONS : +1h · +6h · +12h  (3 sorties LSTM)                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import duckdb
import os
import json
import tempfile
import warnings
from download_data import download_if_needed
download_if_needed()
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# TRADUCTIONS  (fr / en / ar)
# ═══════════════════════════════════════════════════════════════════════════════
TRANSLATIONS = {
    "fr": {
        "app_title": "HSV Algérie", "app_subtitle": "Côtes · ERA5 + CMEMS",
        "model_data": "MODÈLE DE DONNÉES",
        "model1_label": "🔵 M1 — ERA5 seul (1985–2023)",
        "model2_label": "🟣 M2 — ERA5 + CMEMS (1999–2023)",
        "navigation": "Navigation",
        "temporal_filters": "FILTRES TEMPORELS", "geo_filters": "FILTRES GÉOGRAPHIQUES",
        "year": "Année", "month": "Mois", "hour": "Heure",
        "wilaya": "Wilaya", "beach": "Plage", "all": "Tous...", "all_f": "Toutes...",
        "home": "🏠 Accueil", "global_analysis": "📊 Analyse Globale",
        "summer_analysis": "🏖️ Analyse Été", "activities": "🌊 Activités",
        "analysis": "📊 Analyse", "drowning_alerts": "🏊 Alertes Noyades",
        "desalination": "💧 Dessalement SWRO", "aquaculture": "🐟 Aquaculture",
        "synthesis": "📋 Synthèse & Export", "danger_map": "🗺️ Carte des Dangers",
        "realtime_pred": "🔮 Prédiction Temps Réel",
        "hero_title": "Système d'Analyse des Vagues Côtières — Algérie",
        "hero_sub": "Prévision HSV par LSTM + Transfer Learning · Deux modèles complémentaires :",
        "hero_sub2": "M1 ERA5 seul 1985–2023 (20M mesures) · M2 ERA5 + CMEMS 1999–2023 avec Salinité, O₂ dissous et Matières en suspension.",
        "drowning_alerts_pill": "Alertes noyades", "desalination_pill": "Dessalement SWRO",
        "aquaculture_pill": "Aquaculture marine", "marine_quality_pill": "Qualité marine O₂/Salinité",
        "two_models_pill": "Deux modèles LSTM",
        "global_stats": "Statistiques globales — Modèle actif",
        "annual_evolution": "Évolution annuelle de la HSV",
        "critical_thresholds": "Seuils critiques — tableau de synthèse",
        "measures": "Mesures", "avg_hsv": "HSV Moyenne", "max_hsv": "HSV Maximum",
        "p95": "Percentile 95", "std": "Écart-type",
        "months": {1:"Janvier",2:"Février",3:"Mars",4:"Avril",5:"Mai",6:"Juin",
                   7:"Juillet",8:"Août",9:"Septembre",10:"Octobre",11:"Novembre",12:"Décembre"},
        "months_short": {1:"Jan",2:"Fév",3:"Mar",4:"Avr",5:"Mai",6:"Jun",
                         7:"Jul",8:"Aoû",9:"Sep",10:"Oct",11:"Nov",12:"Déc"},
        "no_data": "Aucune donnée disponible.", "no_filtered_data": "Aucune donnée pour les filtres sélectionnés.",
        "computing": "Calcul KPIs...", "loading_map": "Chargement carte...",
        "m2_required": "⚠️ Cette page nécessite le **Modèle 2** (ERA5 + CMEMS). Veuillez sélectionner **M2** dans la sidebar.",
        "m1_drowning_only": "🏊 En Modèle 1, seule la page **Alertes Noyades** est disponible.\nPour Dessalement et Aquaculture, merci de sélectionner **🟣 M2 — ERA5 + CMEMS**.",
        "variables": "Variables", "lang_button": "🌐 Langue", "select_lang": "Sélectionner la langue",
        "time_series": "📈 Série Temporelle", "distribution": "🗂️ Distribution",
        "seasonality": "📅 Saisonnalité", "by_beach": "🏖️ Par Plage",
        "alerts": "📊 Alertes", "wind_mwd": "🌬️ Vent & MWD",
        "favorable_windows": "📊 Fenêtres Favorables", "best_sites": "🏖️ Meilleurs Sites",
        "sst_tab": "🌡️ SST", "pressure_tab": "📊 Pression MSL",
        "op_windows": "⚙️ Fenêtres Opérationnelles", "evolution": "📅 Évolution",
        "by_wilaya": "Classement par wilaya", "synth_by_beach": "📊 Synthèse par Plage",
        "monthly_synth": "📅 Synthèse Mensuelle", "export": "💾 Export",
        "download_csv": "⬇️ Télécharger CSV",
        "indicator_mapped": "Indicateur cartographié", "avg_hsv_map": "HSV Moyenne",
        "max_hsv_map": "HSV Maximum", "alert_m1_map": "Alertes Noyades M1",
        "alert_m2_map": "Alertes Noyades M2", "dessal_map": "Dessalement", "aqua_map": "Aquaculture",
        "seasons": {"Hiver":"Hiver","Printemps":"Printemps","Été":"Été","Automne":"Automne"},
        "pred_m1_label": "🔵 Prédiction M1 — ERA5",
        "pred_m2_label": "🟣 Prédiction M2 — ERA5 + CMEMS",
        "pred_submenu": "Type de prédiction",
    },
    "en": {
        "app_title": "HSV Algeria", "app_subtitle": "Coastline · ERA5 + CMEMS",
        "model_data": "DATA MODEL",
        "model1_label": "🔵 M1 — ERA5 only (1985–2023)",
        "model2_label": "🟣 M2 — ERA5 + CMEMS (1999–2023)",
        "navigation": "Navigation",
        "temporal_filters": "TEMPORAL FILTERS", "geo_filters": "GEOGRAPHIC FILTERS",
        "year": "Year", "month": "Month", "hour": "Hour",
        "wilaya": "Wilaya", "beach": "Beach", "all": "All...", "all_f": "All...",
        "home": "🏠 Home", "global_analysis": "📊 Global Analysis",
        "summer_analysis": "🏖️ Summer Analysis", "activities": "🌊 Activities",
        "analysis": "📊 Analysis", "drowning_alerts": "🏊 Drowning Alerts",
        "desalination": "💧 SWRO Desalination", "aquaculture": "🐟 Aquaculture",
        "synthesis": "📋 Summary & Export", "danger_map": "🗺️ Danger Map",
        "realtime_pred": "🔮 Real-Time Prediction",
        "hero_title": "Coastal Wave Analysis System — Algeria",
        "hero_sub": "HSV Prediction via LSTM + Transfer Learning · Two complementary models:",
        "hero_sub2": "M1 ERA5 only 1985–2023 (20M records) · M2 ERA5 + CMEMS 1999–2023 with Salinity, Dissolved O₂ and Suspended Matter.",
        "drowning_alerts_pill": "Drowning alerts", "desalination_pill": "SWRO Desalination",
        "aquaculture_pill": "Marine aquaculture", "marine_quality_pill": "Marine quality O₂/Salinity",
        "two_models_pill": "Two LSTM models",
        "global_stats": "Global statistics — Active model",
        "annual_evolution": "Annual HSV evolution",
        "critical_thresholds": "Critical thresholds — summary table",
        "measures": "Records", "avg_hsv": "Avg HSV", "max_hsv": "Max HSV",
        "p95": "Percentile 95", "std": "Std Dev",
        "months": {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",
                   7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"},
        "months_short": {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                         7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"},
        "no_data": "No data available.", "no_filtered_data": "No data for the selected filters.",
        "computing": "Computing KPIs...", "loading_map": "Loading map...",
        "m2_required": "⚠️ This page requires **Model 2** (ERA5 + CMEMS). Please select **M2** in the sidebar.",
        "m1_drowning_only": "🏊 In Model 1, only the **Drowning Alerts** page is available.\nFor Desalination and Aquaculture, please select **🟣 M2 — ERA5 + CMEMS**.",
        "variables": "Variables", "lang_button": "🌐 Language", "select_lang": "Select language",
        "time_series": "📈 Time Series", "distribution": "🗂️ Distribution",
        "seasonality": "📅 Seasonality", "by_beach": "🏖️ By Beach",
        "alerts": "📊 Alerts", "wind_mwd": "🌬️ Wind & MWD",
        "favorable_windows": "📊 Favorable Windows", "best_sites": "🏖️ Best Sites",
        "sst_tab": "🌡️ SST", "pressure_tab": "📊 Pressure MSL",
        "op_windows": "⚙️ Operational Windows", "evolution": "📅 Evolution",
        "by_wilaya": "Ranking by wilaya", "synth_by_beach": "📊 Summary by Beach",
        "monthly_synth": "📅 Monthly Summary", "export": "💾 Export",
        "download_csv": "⬇️ Download CSV",
        "indicator_mapped": "Mapped indicator", "avg_hsv_map": "Average HSV",
        "max_hsv_map": "Maximum HSV", "alert_m1_map": "Drowning Alerts M1",
        "alert_m2_map": "Drowning Alerts M2", "dessal_map": "Desalination", "aqua_map": "Aquaculture",
        "seasons": {"Hiver":"Winter","Printemps":"Spring","Été":"Summer","Automne":"Autumn"},
        "pred_m1_label": "🔵 Prediction M1 — ERA5",
        "pred_m2_label": "🟣 Prediction M2 — ERA5 + CMEMS",
        "pred_submenu": "Prediction type",
    },
    "ar": {
        "app_title": "نظام HSV الجزائر", "app_subtitle": "السواحل · ERA5 + CMEMS",
        "model_data": "نموذج البيانات",
        "model1_label": "🔵 N1 — ERA5 فقط (1985–2023)",
        "model2_label": "🟣 N2 — ERA5 + CMEMS (1999–2023)",
        "navigation": "التنقل",
        "temporal_filters": "مرشحات زمنية", "geo_filters": "مرشحات جغرافية",
        "year": "السنة", "month": "الشهر", "hour": "الساعة",
        "wilaya": "الولاية", "beach": "الشاطئ", "all": "الكل...", "all_f": "الكل...",
        "home": "🏠 الرئيسية", "global_analysis": "📊 التحليل العام",
        "summer_analysis": "🏖️ تحليل الصيف", "activities": "🌊 الأنشطة",
        "analysis": "📊 التحليل", "drowning_alerts": "🏊 تنبيهات الغرق",
        "desalination": "💧 تحلية المياه SWRO", "aquaculture": "🐟 تربية الأحياء البحرية",
        "synthesis": "📋 الملخص والتصدير", "danger_map": "🗺️ خريطة المخاطر",
        "realtime_pred": "🔮 التنبؤ الفوري",
        "hero_title": "نظام تحليل الأمواج الساحلية — الجزائر",
        "hero_sub": "توقع HSV بواسطة LSTM + Transfer Learning · نموذجان تكاملييان:",
        "hero_sub2": "N1 ERA5 فقط 1985–2023 (20 مليون قياس) · N2 ERA5 + CMEMS 1999–2023 مع الملوحة، O₂ الذائب والمواد العالقة.",
        "drowning_alerts_pill": "تنبيهات الغرق", "desalination_pill": "تحلية المياه SWRO",
        "aquaculture_pill": "تربية الأحياء البحرية", "marine_quality_pill": "جودة البيئة البحرية O₂/ملوحة",
        "two_models_pill": "نموذجان LSTM",
        "global_stats": "إحصائيات عامة — النموذج النشط",
        "annual_evolution": "التطور السنوي لـ HSV",
        "critical_thresholds": "الحدود الحرجة — جدول ملخص",
        "measures": "القياسات", "avg_hsv": "متوسط HSV", "max_hsv": "أقصى HSV",
        "p95": "المئين 95", "std": "الانحراف المعياري",
        "months": {1:"جانفي",2:"فيفري",3:"مارس",4:"أبريل",5:"ماي",6:"جوان",
                   7:"جويلية",8:"أوت",9:"سبتمبر",10:"أكتوبر",11:"نوفمبر",12:"ديسمبر"},
        "months_short": {1:"جان",2:"فيف",3:"مار",4:"أبر",5:"ماي",6:"جوا",
                         7:"جوي",8:"أوت",9:"سبت",10:"أكت",11:"نوف",12:"ديس"},
        "no_data": "لا توجد بيانات متاحة.", "no_filtered_data": "لا توجد بيانات للمرشحات المحددة.",
        "computing": "جارٍ الحساب...", "loading_map": "جارٍ تحميل الخريطة...",
        "m2_required": "⚠️ هذه الصفحة تتطلب **النموذج 2** (ERA5 + CMEMS). يرجى اختيار **N2** في الشريط الجانبي.",
        "m1_drowning_only": "🏊 في النموذج 1، صفحة **تنبيهات الغرق** فقط متاحة.\nللتحلية والأحياء البحرية، يرجى اختيار **🟣 N2 — ERA5 + CMEMS**.",
        "variables": "المتغيرات", "lang_button": "🌐 اللغة", "select_lang": "اختر اللغة",
        "time_series": "📈 السلسلة الزمنية", "distribution": "🗂️ التوزيع",
        "seasonality": "📅 الموسمية", "by_beach": "🏖️ حسب الشاطئ",
        "alerts": "📊 التنبيهات", "wind_mwd": "🌬️ الرياح والاتجاه",
        "favorable_windows": "📊 النوافذ الملائمة", "best_sites": "🏖️ أفضل المواقع",
        "sst_tab": "🌡️ درجة حرارة السطح", "pressure_tab": "📊 ضغط MSL",
        "op_windows": "⚙️ النوافذ التشغيلية", "evolution": "📅 التطور",
        "by_wilaya": "التصنيف حسب الولاية", "synth_by_beach": "📊 ملخص حسب الشاطئ",
        "monthly_synth": "📅 الملخص الشهري", "export": "💾 تصدير",
        "download_csv": "⬇️ تنزيل CSV",
        "indicator_mapped": "المؤشر المرسوم", "avg_hsv_map": "متوسط HSV",
        "max_hsv_map": "أقصى HSV", "alert_m1_map": "تنبيهات الغرق N1",
        "alert_m2_map": "تنبيهات الغرق N2", "dessal_map": "التحلية", "aqua_map": "الأحياء البحرية",
        "seasons": {"Hiver":"شتاء","Printemps":"ربيع","Été":"صيف","Automne":"خريف"},
        "pred_m1_label": "🔵 التنبؤ N1 — ERA5",
        "pred_m2_label": "🟣 التنبؤ N2 — ERA5 + CMEMS",
        "pred_submenu": "نوع التنبؤ",
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG PAGE
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="HSV · Côtes Algériennes",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "lang" not in st.session_state:
    st.session_state["lang"] = "fr"

def T(key):
    lang = st.session_state.get("lang", "fr")
    return TRANSLATIONS[lang].get(key, TRANSLATIONS["fr"].get(key, key))

# ═══════════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Syne:wght@600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
:root {
  --bg-base:#080f1a;--bg-surface:#0c1829;--bg-card:#0f2035;--bg-card-h:#132640;
  --border-s:rgba(30,90,150,.25);--border-m:rgba(30,110,180,.4);--border-b:rgba(14,157,232,.6);
  --a1:#0ea5e9;--a2:#06b6d4;--a3:#10b981;--warn:#f59e0b;--danger:#ef4444;--purple:#8b5cf6;
  --text-h:#f0f9ff;--text-p:#94b8cc;--text-m:#4a7a96;
  --fd:'Syne',sans-serif;--fb:'DM Sans',sans-serif;--fm:'JetBrains Mono',monospace;
  --r-sm:8px;--r-md:12px;--r-lg:16px;--r-xl:20px;
}
html,body,[class*="css"]{font-family:var(--fb)!important;background:var(--bg-base)!important;color:var(--text-p)!important;}
.stApp{background:var(--bg-base)!important;}
.block-container{padding:1.5rem 2rem 3rem!important;}

/* ── SIDEBAR ── */
[data-testid="stSidebar"]{background:var(--bg-surface)!important;border-right:1px solid var(--border-s)!important;}
[data-testid="stSidebar"] *{color:var(--text-p)!important;}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{font-family:var(--fd)!important;color:var(--text-h)!important;}
[data-testid="stSidebar"] label{font-size:.68rem!important;text-transform:uppercase;letter-spacing:.12em;color:var(--text-m)!important;font-weight:600!important;}

/* ── EXPANDER LANGUE ── */
[data-testid="stSidebar"] [data-testid="stExpander"]{background:var(--bg-card)!important;border:1px solid var(--border-s)!important;border-radius:var(--r-md)!important;overflow:hidden;}
[data-testid="stSidebar"] [data-testid="stExpander"] summary{background:var(--bg-card)!important;color:var(--text-p)!important;padding:8px 12px!important;}
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover{background:var(--bg-card-h)!important;}
[data-testid="stSidebar"] [data-testid="stExpander"] summary p{color:var(--text-p)!important;font-size:.82rem!important;font-weight:500!important;}
[data-testid="stSidebar"] [data-testid="stExpander"] summary svg{stroke:var(--text-m)!important;fill:none!important;}
[data-testid="stSidebar"] [data-testid="stExpanderDetails"]{background:var(--bg-surface)!important;padding:6px 8px 10px!important;}

/* ── BOUTONS LANGUE DANS L'EXPANDER ── */
[data-testid="stSidebar"] [data-testid="stExpander"] .stButton>button{
  background:var(--bg-card)!important;
  color:var(--text-p)!important;
  border:1px solid var(--border-s)!important;
  border-radius:var(--r-sm)!important;
  font-family:var(--fb)!important;
  font-size:.82rem!important;
  font-weight:500!important;
  padding:6px 12px!important;
  width:100%!important;
  text-align:left!important;
  transition:background .15s,border-color .15s,color .15s!important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] .stButton>button:hover{
  background:var(--bg-card-h)!important;
  border-color:var(--border-m)!important;
  color:var(--text-h)!important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] .stButton>button:active{
  background:rgba(14,165,233,.15)!important;
  border-color:var(--border-b)!important;
  color:var(--a1)!important;
}

/* ── MÉTRIQUES ── */
[data-testid="stMetric"]{background:var(--bg-card)!important;border:1px solid var(--border-s)!important;border-radius:var(--r-lg)!important;padding:1.1rem 1.3rem!important;position:relative;overflow:hidden;transition:border-color .2s,background .2s;}
[data-testid="stMetric"]:hover{border-color:var(--border-m)!important;background:var(--bg-card-h)!important;}
[data-testid="stMetric"]::after{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--a1),var(--a2));}
[data-testid="stMetricLabel"]{color:var(--text-m)!important;font-size:.7rem!important;text-transform:uppercase!important;letter-spacing:.12em!important;font-weight:600!important;}
[data-testid="stMetricValue"]{font-family:var(--fd)!important;color:var(--text-h)!important;font-size:1.7rem!important;font-weight:700!important;}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"]{background:var(--bg-card)!important;border:1px solid var(--border-s)!important;border-radius:var(--r-md)!important;padding:4px!important;gap:3px;}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:var(--text-m)!important;border-radius:var(--r-sm)!important;font-size:.8rem!important;font-weight:600!important;letter-spacing:.04em;padding:7px 18px!important;border:none!important;}
.stTabs [aria-selected="true"]{background:var(--a1)!important;color:white!important;}

/* ── BOUTONS GLOBAUX ── */
.stDownloadButton>button,.stButton>button{background:var(--a1)!important;color:white!important;border:none!important;border-radius:var(--r-md)!important;font-weight:600!important;}

hr{border:none;border-top:1px solid var(--border-s)!important;margin:1.5rem 0!important;}

/* ── HERO ── */
.hero{background:var(--bg-surface);border:1px solid var(--border-s);border-radius:var(--r-xl);padding:2.2rem 2rem 1.8rem;margin-bottom:1.5rem;position:relative;overflow:hidden;}
.hero::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--a1),var(--a2),var(--a3));}
.hero h1{font-family:var(--fd);font-size:1.9rem;font-weight:800;color:var(--text-h);margin:0 0 .4rem;line-height:1.2;}
.hero .sub{font-size:.85rem;color:var(--text-m);line-height:1.6;max-width:680px;}
.hero .pills{margin-top:1rem;display:flex;flex-wrap:wrap;gap:8px;}

/* ── SECTIONS & CARDS ── */
.section-title{font-family:var(--fd);font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.16em;color:var(--a1);margin:2rem 0 .8rem;display:flex;align-items:center;gap:8px;}
.section-title::after{content:'';flex:1;height:1px;background:var(--border-s);}
.stat-card{background:var(--bg-card);border:1px solid var(--border-s);border-radius:var(--r-lg);padding:1.1rem 1.3rem;position:relative;overflow:hidden;}
.info-card{background:var(--bg-card);border:1px solid var(--border-s);border-left:3px solid;border-radius:var(--r-md);padding:.9rem 1.1rem;margin-bottom:.7rem;}
.info-card .title{font-family:var(--fd);font-size:.85rem;font-weight:700;color:var(--text-h);margin-bottom:.5rem;}
.threshold-row{display:flex;align-items:center;justify-content:space-between;padding:6px 0;border-bottom:1px solid var(--border-s);font-size:.8rem;}
.threshold-row:last-child{border-bottom:none;}
.threshold-key{color:var(--text-m);}
.threshold-val{font-weight:600;font-family:var(--fm);font-size:.78rem;}

/* ── PILLS ── */
.pill{display:inline-flex;align-items:center;padding:3px 10px;border-radius:20px;font-size:.7rem;font-weight:600;letter-spacing:.04em;border:1px solid;}
.pill-blue{background:rgba(14,165,233,.12);color:#38bdf8;border-color:rgba(14,165,233,.3);}
.pill-green{background:rgba(16,185,129,.12);color:#34d399;border-color:rgba(16,185,129,.3);}
.pill-red{background:rgba(239,68,68,.12);color:#f87171;border-color:rgba(239,68,68,.3);}
.pill-amber{background:rgba(245,158,11,.12);color:#fbbf24;border-color:rgba(245,158,11,.3);}
.pill-purple{background:rgba(139,92,246,.12);color:#a78bfa;border-color:rgba(139,92,246,.3);}
.pill-cyan{background:rgba(6,182,212,.12);color:#22d3ee;border-color:rgba(6,182,212,.3);}

/* ── PAGE HEADER ── */
.page-header{display:flex;align-items:center;gap:12px;margin-bottom:1.5rem;padding-bottom:1rem;border-bottom:1px solid var(--border-s);}
.page-header-icon{width:40px;height:40px;border-radius:var(--r-md);display:flex;align-items:center;justify-content:center;font-size:1.2rem;flex-shrink:0;}
.page-header h1{font-family:var(--fd);font-size:1.4rem;font-weight:700;color:var(--text-h);margin:0;line-height:1.2;}
.page-header p{font-size:.8rem;color:var(--text-m);margin:2px 0 0;}

/* ── SIDEBAR LOGO ── */
.sidebar-logo{text-align:center;padding:1.2rem 0 1.5rem;border-bottom:1px solid var(--border-s);margin-bottom:1rem;}
.sidebar-logo .logo-icon{font-size:2rem;}
.sidebar-logo .logo-title{font-family:var(--fd);font-size:1.1rem;font-weight:800;color:var(--text-h);margin:6px 0 2px;}
.sidebar-logo .logo-sub{font-size:.68rem;text-transform:uppercase;letter-spacing:.12em;color:var(--text-m);}

/* ── SYNTHESIS TABLE ── */
.synthesis-table{width:100%;border-collapse:collapse;font-size:.8rem;}
.synthesis-table thead th{background:var(--bg-surface);color:var(--text-m);font-size:.68rem;text-transform:uppercase;letter-spacing:.1em;font-weight:600;padding:10px 14px;text-align:left;border-bottom:1px solid var(--border-s);}
.synthesis-table tbody td{padding:9px 14px;border-bottom:1px solid var(--border-s);color:var(--text-p);vertical-align:middle;}
.synthesis-table tbody tr:last-child td{border-bottom:none;}
.synthesis-table tbody tr:hover td{background:var(--bg-card-h);}

/* ── BANNERS ── */
.incompat-banner{background:rgba(239,68,68,.08);border:2px solid rgba(239,68,68,.4);border-radius:var(--r-xl);padding:2rem 2.2rem;margin-bottom:1.5rem;position:relative;overflow:hidden;}
.incompat-banner::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#ef4444,#f87171);}
.incompat-banner .ib-icon{font-size:2.5rem;margin-bottom:.6rem;}
.incompat-banner .ib-title{font-family:var(--fd);font-size:1.3rem;font-weight:800;color:#f87171;margin-bottom:.5rem;}
.incompat-banner .ib-body{font-size:.88rem;color:var(--text-p);line-height:1.7;}
.incompat-banner .ib-step{background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.25);border-radius:var(--r-md);padding:.7rem 1rem;margin-top:1rem;font-size:.82rem;color:#fca5a5;}
.wip-banner{background:rgba(139,92,246,.08);border:2px solid rgba(139,92,246,.35);border-radius:var(--r-xl);padding:2rem 2.2rem;margin-bottom:1.5rem;position:relative;overflow:hidden;}
.wip-banner::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,var(--purple),#a855f7);}
.wip-banner .wb-icon{font-size:2.5rem;margin-bottom:.6rem;}
.wip-banner .wb-title{font-family:var(--fd);font-size:1.3rem;font-weight:800;color:#a78bfa;margin-bottom:.5rem;}
.wip-banner .wb-body{font-size:.88rem;color:var(--text-p);line-height:1.7;}

/* ── FEAT TABLE ── */
.feat-table{width:100%;border-collapse:collapse;font-size:.82rem;margin-top:1rem;}
.feat-table th{background:var(--bg-card);color:var(--text-m);font-size:.68rem;text-transform:uppercase;letter-spacing:.1em;padding:8px 14px;text-align:left;border-bottom:1px solid var(--border-m);}
.feat-table td{padding:8px 14px;border-bottom:1px solid var(--border-s);color:var(--text-p);}
.feat-table tr:last-child td{border-bottom:none;}
.feat-table .check-yes{color:#34d399;font-weight:700;}
.feat-table .check-no{color:#4a7a96;}
.feat-table .feat-name{color:var(--text-h);font-family:var(--fm);font-size:.78rem;}

/* ── HORIZON BADGE ── */
.horizon-badge{display:inline-flex;align-items:center;gap:6px;background:rgba(14,165,233,.1);border:1px solid rgba(14,165,233,.3);border-radius:8px;padding:5px 12px;font-size:.78rem;color:#38bdf8;margin-bottom:1rem;}

/* ── SELECTBOX / MULTISELECT / RADIO sidebar ── */
[data-testid="stSidebar"] [data-baseweb="select"] > div{background:var(--bg-card)!important;border-color:var(--border-s)!important;color:var(--text-p)!important;}
[data-testid="stSidebar"] [data-baseweb="select"] [data-testid="stMarkdownContainer"] p{color:var(--text-p)!important;}
[data-testid="stSidebar"] [data-baseweb="radio"] label{color:var(--text-p)!important;}
[data-testid="stSidebar"] [data-baseweb="radio"] [role="radio"]{border-color:var(--border-m)!important;}
[data-testid="stSidebar"] [data-baseweb="radio"] [data-checked="true"] [role="radio"]{background:var(--a1)!important;border-color:var(--a1)!important;}

/* ── MULTISELECT TAG ── */
[data-testid="stSidebar"] [data-baseweb="tag"]{background:rgba(14,165,233,.15)!important;border-color:rgba(14,165,233,.3)!important;}
[data-testid="stSidebar"] [data-baseweb="tag"] span{color:var(--text-h)!important;}
</style>
""", unsafe_allow_html=True)
# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ═══════════════════════════════════════════════════════════════════════════════
PATH_M1     = "data/lstm_final_clean"
PATH_M2     = "data/dataset_model2_1999_2023_clean"
LSTM_PATH   = "models/global_lstm.keras"
SCALER_PATH = "models/scaler.pkl"

HORIZONS  = [1, 6, 12]
N_OUTPUTS = len(HORIZONS)

SEASON_COLORS = {'Hiver':'#818cf8','Printemps':'#34d399','Été':'#f87171','Automne':'#fb923c'}
DANGER_COLORS = {
    "Calme (<0.5m)":"#10b981","Faible (0.5–1.5m)":"#f59e0b",
    "Modéré (1.5–2.5m)":"#ef4444","Agité (2.5–4m)":"#8b5cf6","Très agité (>4m)":"#6d28d9",
}
ALERTE_COLORS = {
    'Calme (< 1 m)':'#10b981','Vigilance (1–2 m)':'#f59e0b','Danger (> 2 m)':'#ef4444',
}

PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(12,24,41,0.6)",
    font=dict(color="#94b8cc",family="DM Sans",size=12),
    title_font=dict(color="#f0f9ff",family="Syne",size=14),
    xaxis=dict(gridcolor="rgba(30,90,150,.2)",linecolor="rgba(30,90,150,.3)",tickcolor="#4a7a96",zerolinecolor="rgba(30,90,150,.2)"),
    yaxis=dict(gridcolor="rgba(30,90,150,.2)",linecolor="rgba(30,90,150,.3)",tickcolor="#4a7a96",zerolinecolor="rgba(30,90,150,.2)"),
    legend=dict(bgcolor="rgba(12,24,41,.8)",bordercolor="rgba(30,90,150,.3)",borderwidth=1),
    margin=dict(l=20,r=20,t=50,b=30),
    hoverlabel=dict(bgcolor="#0f2035",bordercolor="rgba(30,90,150,.5)",font_color="#f0f9ff"),
    colorway=["#0ea5e9","#06b6d4","#10b981","#f59e0b","#ef4444","#8b5cf6","#f97316"],
)

def apply_theme(fig): fig.update_layout(**PLOTLY_THEME); return fig
def section(icon,title): st.markdown(f'<div class="section-title">{icon} {title}</div>',unsafe_allow_html=True)
def page_header(icon_bg,icon,title,subtitle):
    st.markdown(f"""
    <div class="page-header">
        <div class="page-header-icon" style="background:{icon_bg}20;border:1px solid {icon_bg}40;">{icon}</div>
        <div><h1>{title}</h1><p>{subtitle}</p></div>
    </div>""",unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# DUCKDB
# ═══════════════════════════════════════════════════════════════════════════════
data_m1_ok = os.path.exists(PATH_M1)
data_m2_ok = os.path.exists(PATH_M2)

def _collect_parquets(path):
    files = []
    if not path: return files
    if os.path.isdir(path):
        for r,_,fs in os.walk(path):
            for f in fs:
                if f.endswith(".parquet"):
                    files.append(os.path.join(r,f))
    elif path.endswith(".parquet") and os.path.exists(path):
        files = [path]
    return files

def _build_view(con, view_name, path):
    files = _collect_parquets(path)
    if not files: return False
    files_sql = ", ".join(f"'{f}'" for f in files)
    probe = con.execute(f"SELECT * FROM read_parquet([{files_sql}]) LIMIT 1").df()
    cols  = probe.columns.tolist()
    def _col(c, typ="DOUBLE"):
        return f"CAST({c} AS {typ})" if c in cols else f"NULL::{typ}"
    dist_expr = _col("DISTANCE","INTEGER") if "DISTANCE" in cols else "1::INTEGER"
    ws_expr   = _col("wind_speed")
    u10_expr  = _col("u10"); v10_expr = _col("v10")
    mwp_expr  = _col("mwp"); mwd_expr  = _col("mwd")
    sal_expr  = _col("salinity"); o2_expr = _col("o2"); spm_expr = _col("spm")
    try:
        avg_sst = con.execute(f"SELECT AVG(CAST(sst AS DOUBLE)) FROM read_parquet([{files_sql}]) LIMIT 100000").fetchone()[0]
        sst_conv = "CAST(sst AS DOUBLE) - 273.15" if avg_sst and avg_sst>100 else "CAST(sst AS DOUBLE)"
    except: sst_conv = "NULL::DOUBLE"
    try:
        avg_msl = con.execute(f"SELECT AVG(CAST(msl AS DOUBLE)) FROM read_parquet([{files_sql}]) LIMIT 100000").fetchone()[0]
        msl_conv = "CAST(msl AS DOUBLE)/100.0" if avg_msl and avg_msl>10000 else "CAST(msl AS DOUBLE)"
    except: msl_conv = "NULL::DOUBLE"
    con.execute(f"""
        CREATE OR REPLACE VIEW {view_name} AS
        SELECT
            CAST(NOM_PLAGE  AS VARCHAR)   AS NOM_PLAGE,
            CAST(NOM_WILAYA AS VARCHAR)   AS NOM_WILAYA,
            CAST(DATETIME   AS TIMESTAMP) AS DATETIME,
            CAST(X AS DOUBLE)             AS X,
            CAST(Y AS DOUBLE)             AS Y,
            {dist_expr}                   AS DISTANCE,
            CAST(MESURE AS DOUBLE)        AS MESURE,
            {u10_expr}  AS u10, {v10_expr} AS v10,
            {ws_expr}   AS wind_speed,
            {mwp_expr}  AS mwp,
            {mwd_expr}  AS mwd,
            ({sst_conv}) AS sst,
            ({msl_conv}) AS msl,
            {sal_expr}  AS salinity,
            {o2_expr}   AS o2,
            {spm_expr}  AS spm,
            YEAR(CAST(DATETIME AS TIMESTAMP))      AS YEAR,
            MONTH(CAST(DATETIME AS TIMESTAMP))     AS MONTH,
            DAY(CAST(DATETIME AS TIMESTAMP))       AS DAY,
            HOUR(CAST(DATETIME AS TIMESTAMP))      AS HOUR,
            DAYOFWEEK(CAST(DATETIME AS TIMESTAMP)) AS WEEKDAY,
            CASE MONTH(CAST(DATETIME AS TIMESTAMP))
                WHEN 12 THEN 'Hiver' WHEN 1 THEN 'Hiver' WHEN 2 THEN 'Hiver'
                WHEN 3 THEN 'Printemps' WHEN 4 THEN 'Printemps' WHEN 5 THEN 'Printemps'
                WHEN 6 THEN 'Été' WHEN 7 THEN 'Été' WHEN 8 THEN 'Été'
                WHEN 9 THEN 'Automne' WHEN 10 THEN 'Automne' WHEN 11 THEN 'Automne'
            END AS SEASON,
            CASE
                WHEN CAST(MESURE AS DOUBLE) < 1.0 THEN 'Calme (< 1 m)'
                WHEN CAST(MESURE AS DOUBLE) < 2.0 THEN 'Vigilance (1–2 m)'
                ELSE 'Danger (> 2 m)'
            END AS ALERTE,
            CASE
                WHEN CAST(MESURE AS DOUBLE) < 0.5 THEN 'Calme (<0.5m)'
                WHEN CAST(MESURE AS DOUBLE) < 1.5 THEN 'Faible (0.5–1.5m)'
                WHEN CAST(MESURE AS DOUBLE) < 2.5 THEN 'Modéré (1.5–2.5m)'
                WHEN CAST(MESURE AS DOUBLE) < 4.0 THEN 'Agité (2.5–4m)'
                ELSE 'Très agité (>4m)'
            END AS NIVEAU,
            CASE WHEN CAST(MESURE AS DOUBLE)<1.2
                      AND ({sst_conv}) BETWEEN 16 AND 24
                      AND {mwp_expr}<8
                 THEN TRUE ELSE FALSE END AS AQUA_OK,
            CASE WHEN CAST(MESURE AS DOUBLE)<=3.0
                 THEN TRUE ELSE FALSE END AS DESSAL_OK
        FROM read_parquet([{files_sql}])
    """)
    return True

@st.cache_resource
def get_con():
    con = duckdb.connect(database=":memory:", read_only=False)
    if data_m1_ok: _build_view(con, "hsv",  PATH_M1)
    if data_m2_ok: _build_view(con, "hsv2", PATH_M2)
    return con

con = get_con()

def q(sql):
    try:
        return con.execute(sql).df()
    except Exception as e:
        st.error(f"❌ Erreur SQL : {str(e)[:300]}")
        return pd.DataFrame()

def has_col(view, col):
    try:
        r = con.execute(f"SELECT COUNT(*) FROM {view} WHERE {col} IS NOT NULL LIMIT 1").fetchone()
        return r[0] > 0
    except: return False

@st.cache_data(show_spinner=False)
def get_lists(view):
    try:
        wilayas = con.execute(f"SELECT DISTINCT NOM_WILAYA FROM {view} ORDER BY NOM_WILAYA").df()["NOM_WILAYA"].tolist()
        plages  = con.execute(f"SELECT DISTINCT NOM_PLAGE  FROM {view} ORDER BY NOM_PLAGE").df()["NOM_PLAGE"].tolist()
        years   = con.execute(f"SELECT DISTINCT YEAR FROM {view} ORDER BY YEAR").df()["YEAR"].tolist()
        return wilayas, plages, years
    except: return [], [], []

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    LANG_FLAGS = {"fr": " Français", "en": " English", "ar": " العربية"}
    current_flag = LANG_FLAGS.get(st.session_state.get("lang","fr"), " Français")
    with st.expander(f"🌐 {current_flag}", expanded=False):
        for code, label in LANG_FLAGS.items():
            if st.button(label, key=f"lang_{code}", use_container_width=True):
                st.session_state["lang"] = code
                st.rerun()

    st.markdown(f"""
    <div class="sidebar-logo">
        <div class="logo-icon">🌊</div>
        <div class="logo-title">{T("app_title")}</div>
        <div class="logo-sub">{T("app_subtitle")}</div>
    </div>""",unsafe_allow_html=True)

    st.markdown(f"**{T('model_data')}**")
    model_choice = st.radio("Modèle", [T("model1_label"), T("model2_label")], label_visibility="collapsed")
    is_m2 = "M2" in model_choice or "N2" in model_choice
    VIEW  = "hsv2" if is_m2 else "hsv"
    data_ok = data_m2_ok if is_m2 else data_m1_ok

    if is_m2:
        st.markdown("""
        <div style="background:rgba(139,92,246,.1);border:1px solid rgba(139,92,246,.3);border-radius:8px;padding:8px 12px;font-size:.75rem;color:#a78bfa;margin-bottom:.8rem;">
            🟣 <b>M2</b> · ERA5 + CMEMS<br>
            <span style="color:#94b8cc;">salinity · o2 · spm · sst · mwp · wind_speed · MESURE</span>
        </div>""",unsafe_allow_html=True)
        if not data_m2_ok:
            st.error("❌ Dataset M2 introuvable :\n`data/dataset_model2_1999_2023_clean`")
    else:
        st.markdown("""
        <div style="background:rgba(14,165,233,.1);border:1px solid rgba(14,165,233,.3);border-radius:8px;padding:8px 12px;font-size:.75rem;color:#38bdf8;margin-bottom:.8rem;">
            🔵 <b>M1</b> · ERA5 seul<br>
            <span style="color:#94b8cc;">MESURE · wind_speed · mwp · mwd</span>
        </div>""",unsafe_allow_html=True)
        if not data_m1_ok:
            st.error("❌ Dataset M1 introuvable :\n`data/lstm_final_clean`")

    st.markdown("---")
    st.markdown(f"**{T('navigation')}**")
    NAV_MAIN = [T("home"), T("analysis"), T("activities"), T("synthesis"), T("danger_map"), T("realtime_pred")]
    page = st.selectbox("Nav", NAV_MAIN, label_visibility="collapsed")

    analysis_page = None
    if page == T("analysis"):
        st.markdown("<div style='font-size:.68rem;text-transform:uppercase;letter-spacing:.12em;color:#4a7a96;font-weight:600;margin-top:.5rem;margin-bottom:.3rem;'>Sous-menu Analyse</div>", unsafe_allow_html=True)
        analysis_page = st.radio("Analyse", [T("global_analysis"), T("summer_analysis")], label_visibility="collapsed")

    activity_page = None
    if page == T("activities"):
        st.markdown(f"<div style='font-size:.68rem;text-transform:uppercase;letter-spacing:.12em;color:#4a7a96;font-weight:600;margin-top:.5rem;margin-bottom:.3rem;'>{T('activities')}</div>", unsafe_allow_html=True)
        activity_page = st.radio("Activité", [T("drowning_alerts"), T("desalination"), T("aquaculture")], label_visibility="collapsed")

    pred_model_page = None
    if page == T("realtime_pred"):
        st.markdown(
            f"<div style='font-size:.68rem;text-transform:uppercase;letter-spacing:.12em;color:#4a7a96;font-weight:600;margin-top:.5rem;margin-bottom:.3rem;'>🔮 {T('pred_submenu')}</div>",
            unsafe_allow_html=True
        )
        pred_model_page = st.radio(
            "Pred modèle",
            [T("pred_m1_label"), T("pred_m2_label")],
            label_visibility="collapsed"
        )
        if pred_model_page == T("pred_m1_label"):
            clr = "#0ea5e9" if not is_m2 else "#ef4444"
            txt = "✅ Modèle actif correspondant" if not is_m2 else "⚠️ Modèle actif : M2 — incompatible"
        else:
            clr = "#8b5cf6" if is_m2 else "#ef4444"
            txt = "✅ Modèle actif correspondant" if is_m2 else "⚠️ Modèle actif : M1 — incompatible"
        st.markdown(f"<div style='background:{clr}12;border:1px solid {clr}40;border-radius:8px;padding:6px 10px;font-size:.72rem;color:{clr};margin-top:4px;'>{txt}</div>",unsafe_allow_html=True)

    

    all_wilayas, all_plages, all_years = get_lists(VIEW) if data_ok else ([],[],[])

    st.markdown(f"**{T('temporal_filters')}**")
    year_filter  = st.multiselect(T("year"),  all_years, default=[], placeholder=T("all_f"))
    month_filter = st.multiselect(T("month"), list(range(1,13)), format_func=lambda x: T("months")[x], default=[], placeholder=T("all"))
    hour_filter  = st.multiselect(T("hour"),  list(range(0,24)), format_func=lambda x: f"{x:02d}h00", default=[], placeholder=T("all_f"))

    st.markdown(f"**{T('geo_filters')}**")
    wilaya_filter = st.multiselect(T("wilaya"), all_wilayas, default=[], placeholder=T("all_f"))
    if wilaya_filter and data_ok:
        wil_in = ",".join(f"'{w}'" for w in wilaya_filter)
        plages_dispo = q(f"SELECT DISTINCT NOM_PLAGE FROM {VIEW} WHERE NOM_WILAYA IN ({wil_in}) ORDER BY NOM_PLAGE")["NOM_PLAGE"].tolist() if data_ok else []
    else:
        plages_dispo = all_plages
    plage_filter = st.multiselect(T("beach"), plages_dispo, default=[], placeholder=T("all_f"))

    st.markdown("---")
    vars_ok = []
    for c in ["MESURE","wind_speed","mwp","mwd"]:
        if data_ok and has_col(VIEW, c.lower() if c!="MESURE" else "MESURE"): vars_ok.append(c)
    if is_m2:
        for c in ["sst","salinity","o2","spm"]:
            if data_ok and has_col(VIEW, c): vars_ok.append(c.upper())
    if vars_ok:
        badge = "🟣 M2" if is_m2 else "🔵 M1"
        st.success(f"✅ {badge} {T('variables')}\n{', '.join(vars_ok)}")

# ═══════════════════════════════════════════════════════════════════════════════
# WHERE CLAUSE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def where_clause(extra=""):
    conds = []
    if year_filter:   conds.append(f"YEAR IN ({','.join(str(y) for y in year_filter)})")
    if month_filter:  conds.append(f"MONTH IN ({','.join(str(m) for m in month_filter)})")
    if hour_filter:   conds.append(f"HOUR IN ({','.join(str(h) for h in hour_filter)})")
    if plage_filter:
        pl_in = ",".join(f"'{p}'" for p in plage_filter)
        conds.append(f"NOM_PLAGE IN ({pl_in})")
    elif wilaya_filter:
        wil_in2 = ",".join(f"'{w}'" for w in wilaya_filter)
        conds.append(f"NOM_WILAYA IN ({wil_in2})")
    if extra: conds.append(f"({extra})")
    return ("WHERE " + " AND ".join(conds)) if conds else ""

def where_clause_with_extra(additional_extra=""):
    return where_clause(additional_extra)

W = where_clause

def show_kpis(wh=""):
    if not data_ok: st.warning(T("no_data")); return
    with st.spinner(T("computing")):
        r = q(f"""
            SELECT COUNT(*) AS total, AVG(MESURE) AS avg_hsv, MAX(MESURE) AS max_hsv,
                   STDDEV(MESURE) AS std_hsv,
                   SUM(CASE WHEN MESURE>=1.5 THEN 1 ELSE 0 END) AS n15,
                   SUM(CASE WHEN MESURE>=2.5 THEN 1 ELSE 0 END) AS n25,
                   SUM(CASE WHEN MESURE>=4.0 THEN 1 ELSE 0 END) AS n40,
                   PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY MESURE) AS p95
            FROM {VIEW} {wh}
        """)
    if r.empty or r["total"].iloc[0]==0: st.warning(T("no_filtered_data")); return
    total=r["total"].iloc[0]; avg_h=r["avg_hsv"].iloc[0]; max_h=r["max_hsv"].iloc[0]
    std_h=r["std_hsv"].iloc[0]; n15=r["n15"].iloc[0]; n25=r["n25"].iloc[0]
    n40=r["n40"].iloc[0]; p95=r["p95"].iloc[0]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric(f"📦 {T('measures')}",   f"{int(total):,}")
    c2.metric(f"🌊 {T('avg_hsv')}",    f"{avg_h:.3f} m", f"± {std_h:.3f} m")
    c3.metric(f"⚠️ {T('max_hsv')}",    f"{max_h:.2f} m")
    c4.metric(f"📈 {T('p95')}",        f"{p95:.2f} m")
    c5,c6,c7,c8 = st.columns(4)
    c5.metric("🟡 ≥ 1.5 m", f"{int(n15):,}", f"{n15/total*100:.1f}%")
    c6.metric("🟠 ≥ 2.5 m", f"{int(n25):,}", f"{n25/total*100:.1f}%")
    c7.metric("🔴 ≥ 4.0 m", f"{int(n40):,}", f"{n40/total*100:.1f}%")
    c8.metric(f"📐 {T('std')}",        f"{std_h:.3f} m")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE : ACCUEIL
# ═══════════════════════════════════════════════════════════════════════════════
if page == T("home"):
    st.markdown(f"""
    <div class="hero">
        <h1>{T("hero_title")}</h1>
        <div class="sub">
            {T("hero_sub")}<br>
            <b>M1</b> {T("hero_sub2")}
        </div>
        <div class="pills">
            <span class="pill pill-red">{T("drowning_alerts_pill")}</span>
            <span class="pill pill-blue">{T("desalination_pill")}</span>
            <span class="pill pill-green">{T("aquaculture_pill")}</span>
            <span class="pill pill-cyan">{T("marine_quality_pill")}</span>
            <span class="pill pill-purple">{T("two_models_pill")}</span>
        </div>
    </div>""",unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-card" style="border-left-color:#0ea5e9;">
            <div class="title" style="color:#38bdf8;">🔵 Modèle 1 — ERA5 (1985–2023)</div>
            <div class="threshold-row"><span class="threshold-key">Période</span><span class="threshold-val" style="color:#38bdf8;">1985 → 2023</span></div>
            <div class="threshold-row"><span class="threshold-key">Lignes</span><span class="threshold-val">~20 millions</span></div>
            <div class="threshold-row"><span class="threshold-key">Variables</span><span class="threshold-val">MESURE · wind_speed · mwp · mwd</span></div>
            <div class="threshold-row"><span class="threshold-key">Application</span><span class="threshold-val" style="color:#f87171;">⚠️ Alertes Noyades + Prédiction TR M1</span></div>
        </div>""",unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="info-card" style="border-left-color:#8b5cf6;">
            <div class="title" style="color:#a78bfa;">🟣 Modèle 2 — ERA5 + CMEMS (1999–2023)</div>
            <div class="threshold-row"><span class="threshold-key">Période</span><span class="threshold-val" style="color:#a78bfa;">1999 → 2023</span></div>
            <div class="threshold-row"><span class="threshold-key">Lignes</span><span class="threshold-val">~12 millions</span></div>
            <div class="threshold-row"><span class="threshold-key">Dessalement</span><span class="threshold-val" style="color:#22d3ee;">salinity · spm · MESURE · mwp · wind_speed</span></div>
            <div class="threshold-row"><span class="threshold-key">Aquaculture</span><span class="threshold-val" style="color:#34d399;">o2 · sst · mwp · wind_speed · MESURE · spm</span></div>
        </div>""",unsafe_allow_html=True)

    section("📊", T("global_stats"))
    if data_ok: show_kpis()
    else: st.error("❌ Dataset introuvable.")

    if data_ok:
        st.markdown("---")
        section("📅", T("annual_evolution"))
        @st.cache_data(show_spinner=False)
        def _annual(view):
            return q(f"SELECT YEAR, AVG(MESURE) AS avg_hsv, MAX(MESURE) AS max_hsv, STDDEV(MESURE) AS std_hsv FROM {view} GROUP BY YEAR ORDER BY YEAR")
        df_yr = _annual(VIEW)
        if not df_yr.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_yr["YEAR"],y=df_yr["avg_hsv"]+df_yr["std_hsv"],fill=None,mode="lines",line=dict(width=0),showlegend=False,hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=df_yr["YEAR"],y=df_yr["avg_hsv"]-df_yr["std_hsv"],fill="tonexty",mode="lines",line=dict(width=0),fillcolor="rgba(14,165,233,.1)",showlegend=False,hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=df_yr["YEAR"],y=df_yr["avg_hsv"],name="HSV Moyenne",mode="lines+markers",line=dict(color="#0ea5e9",width=2.5),marker=dict(size=5,color="#06b6d4")))
            fig.add_trace(go.Scatter(x=df_yr["YEAR"],y=df_yr["max_hsv"],name="HSV Maximum",mode="lines",line=dict(color="#ef4444",width=1.5,dash="dot")))
            fig.add_hline(y=1.5,line_dash="dash",line_color="#f59e0b",annotation_text="Seuil 1.5 m",annotation_font_color="#f59e0b",annotation_font_size=10)
            apply_theme(fig)
            fig.update_layout(title="Évolution annuelle HSV — Côtes algériennes",xaxis_title="Année",yaxis_title="HSV (m)",height=380)
            st.plotly_chart(fig,use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE : ANALYSE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == T("analysis"):

    if analysis_page == T("global_analysis"):
        badge = "🟣 M2" if is_m2 else "🔵 M1"
        page_header("#0ea5e9","📊","Analyse Globale",f"Distribution et tendances HSV — {badge}")
        wh = W()
        show_kpis(wh)

        tab1,tab2,tab3,tab4 = st.tabs([T("time_series"),T("distribution"),T("seasonality"),T("by_beach")])

        with tab1:
            section("📈","Évolution annuelle")
            df_ann = q(f"SELECT YEAR,AVG(MESURE) AS avg_hsv,MAX(MESURE) AS max_hsv,STDDEV(MESURE) AS std_hsv FROM {VIEW} {wh} GROUP BY YEAR ORDER BY YEAR")
            if not df_ann.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_ann["YEAR"],y=df_ann["avg_hsv"]+df_ann["std_hsv"],fill=None,mode="lines",line=dict(width=0),showlegend=False,hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=df_ann["YEAR"],y=df_ann["avg_hsv"]-df_ann["std_hsv"],fill="tonexty",mode="lines",line=dict(width=0),fillcolor="rgba(14,165,233,.1)",showlegend=False,hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=df_ann["YEAR"],y=df_ann["avg_hsv"],name="Moyenne",mode="lines+markers",line=dict(color="#0ea5e9",width=2)))
                fig.add_trace(go.Scatter(x=df_ann["YEAR"],y=df_ann["max_hsv"],name="Maximum",mode="lines",line=dict(color="#ef4444",width=1.5,dash="dot")))
                for seuil,color,label in [(1.5,"#f59e0b","Vigilance"),(2.5,"#ef4444","Danger")]:
                    fig.add_hline(y=seuil,line_dash="dash",line_color=color,annotation_text=label,annotation_font_color=color,annotation_font_size=10)
                apply_theme(fig); fig.update_layout(title="HSV annuelle",xaxis_title="Année",yaxis_title="HSV (m)",height=380)
                st.plotly_chart(fig,use_container_width=True)

            section("📆","Évolution mensuelle")
            df_mo = q(f"SELECT MONTH,AVG(MESURE) AS avg_hsv,MAX(MESURE) AS max_hsv,STDDEV(MESURE) AS std_hsv FROM {VIEW} {wh} GROUP BY MONTH ORDER BY MONTH")
            if not df_mo.empty:
                df_mo["MOIS_LABEL"] = df_mo["MONTH"].map(T("months_short"))
                fig2 = go.Figure(go.Bar(x=df_mo["MOIS_LABEL"],y=df_mo["avg_hsv"],
                    marker_color=df_mo["avg_hsv"].apply(lambda v:"#ef4444" if v>=2 else "#f59e0b" if v>=1 else "#10b981"),
                    error_y=dict(type="data",array=df_mo["std_hsv"],visible=True,color="rgba(148,184,204,.4)")))
                apply_theme(fig2); fig2.update_layout(title="HSV moyenne par mois",xaxis_title=T("month"),yaxis_title="HSV (m)",height=340)
                st.plotly_chart(fig2,use_container_width=True)

        with tab2:
            col_a,col_b = st.columns(2)
            with col_a:
                df_dist = q(f"SELECT ROUND(MESURE,1) AS h,COUNT(*) AS n FROM {VIEW} {wh} GROUP BY h ORDER BY h")
                if not df_dist.empty:
                    fig = go.Figure(go.Bar(x=df_dist["h"],y=df_dist["n"],marker_color="#0ea5e9",marker_line_width=0))
                    for s,c in [(1.5,"#f59e0b"),(2.5,"#ef4444")]:
                        fig.add_vline(x=s,line_color=c,line_dash="dash",annotation_text=f"{s} m",annotation_font_color=c)
                    apply_theme(fig); fig.update_layout(title="Histogramme HSV",xaxis_title="HSV (m)",yaxis_title="Nombre",height=340)
                    st.plotly_chart(fig,use_container_width=True)
            with col_b:
                df_niv = q(f"SELECT NIVEAU,COUNT(*) AS n FROM {VIEW} {wh} GROUP BY NIVEAU")
                if not df_niv.empty:
                    ordre = ["Calme (<0.5m)","Faible (0.5–1.5m)","Modéré (1.5–2.5m)","Agité (2.5–4m)","Très agité (>4m)"]
                    df_niv["NIVEAU"] = pd.Categorical(df_niv["NIVEAU"],categories=ordre,ordered=True)
                    df_niv = df_niv.sort_values("NIVEAU")
                    fig = go.Figure(go.Pie(labels=df_niv["NIVEAU"],values=df_niv["n"],hole=0.55,
                        marker_colors=[DANGER_COLORS.get(n,"#666") for n in df_niv["NIVEAU"]]))
                    apply_theme(fig); fig.update_layout(title="Répartition par niveau",height=340)
                    st.plotly_chart(fig,use_container_width=True)

        with tab3:
            col1,col2 = st.columns(2)
            with col1:
                df_seas = q(f"SELECT SEASON,AVG(MESURE) AS avg_hsv,COUNT(*) AS n FROM {VIEW} {wh} GROUP BY SEASON")
                if not df_seas.empty:
                    ord_s = ['Hiver','Printemps','Été','Automne']
                    df_seas["SEASON"] = pd.Categorical(df_seas["SEASON"],categories=ord_s,ordered=True)
                    df_seas = df_seas.sort_values("SEASON")
                    fig = go.Figure(go.Bar(x=df_seas["SEASON"],y=df_seas["avg_hsv"],
                        marker_color=[SEASON_COLORS.get(s,"#0ea5e9") for s in df_seas["SEASON"]],
                        text=df_seas["avg_hsv"].map(lambda v:f"{v:.2f} m"),textposition="outside"))
                    apply_theme(fig); fig.update_layout(title="HSV par saison",yaxis_title="HSV (m)",height=340)
                    st.plotly_chart(fig,use_container_width=True)
            with col2:
                df_hr = q(f"SELECT HOUR,AVG(MESURE) AS avg_hsv FROM {VIEW} {wh} GROUP BY HOUR ORDER BY HOUR")
                if not df_hr.empty:
                    fig = go.Figure(go.Scatter(x=df_hr["HOUR"],y=df_hr["avg_hsv"],mode="lines+markers",fill="tozeroy",
                        line=dict(color="#06b6d4",width=2),fillcolor="rgba(6,182,212,.1)"))
                    apply_theme(fig); fig.update_layout(title="Cycle diurne",xaxis_title=T("hour"),yaxis_title="HSV (m)",height=340)
                    st.plotly_chart(fig,use_container_width=True)

        with tab4:
            df_pl = q(f"""
                SELECT NOM_PLAGE,NOM_WILAYA,AVG(MESURE) AS avg_hsv,MAX(MESURE) AS max_hsv,
                       STDDEV(MESURE) AS std_hsv,COUNT(*) AS n,
                       SUM(CASE WHEN MESURE>=2 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_danger
                FROM {VIEW} {wh} GROUP BY NOM_PLAGE,NOM_WILAYA ORDER BY avg_hsv DESC LIMIT 30
            """)
            if not df_pl.empty:
                fig = go.Figure(go.Bar(x=df_pl["avg_hsv"],y=df_pl["NOM_PLAGE"],orientation="h",
                    marker_color=df_pl["avg_hsv"].apply(lambda v:"#ef4444" if v>=1.5 else "#f59e0b" if v>=1 else "#10b981"),
                    text=df_pl["avg_hsv"].map(lambda v:f"{v:.2f} m"),textposition="outside"))
                apply_theme(fig); fig.update_layout(title="Top 30 plages — HSV moyenne",xaxis_title="HSV (m)",
                    height=max(400,len(df_pl)*22),yaxis=dict(autorange="reversed",**PLOTLY_THEME["yaxis"]))
                st.plotly_chart(fig,use_container_width=True)

    elif analysis_page == T("summer_analysis"):
        page_header("#f97316","🏖️","Analyse Estivale","Juin · Juillet · Août — Risques HSV, vent et direction")
        wh_ete = where_clause_with_extra("MONTH IN (6,7,8)")
        section("📊","KPIs Estivaux"); show_kpis(wh_ete)

        def _risk_color(score):
            if score>=1.0: return "#ef4444"
            elif score>=0.7: return "#f97316"
            elif score>=0.4: return "#f59e0b"
            else: return "#22c55e"

        tab1,tab2,tab3 = st.tabs([T("evolution"),T("wind_mwd"),T("by_beach")])

        with tab1:
            col1,col2 = st.columns(2)
            with col1:
                df_ye = q(f"SELECT YEAR,AVG(MESURE) AS avg_hsv,MAX(MESURE) AS max_hsv FROM {VIEW} {wh_ete} GROUP BY YEAR ORDER BY YEAR")
                if not df_ye.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=df_ye["YEAR"],y=df_ye["avg_hsv"],name="HSV moyen",marker_color="#0ea5e9"))
                    fig.add_trace(go.Scatter(x=df_ye["YEAR"],y=df_ye["max_hsv"],name="HSV max",mode="lines+markers",line=dict(color="#ef4444",width=2)))
                    fig.add_hline(y=1.5,line_dash="dash",line_color="#f59e0b")
                    apply_theme(fig); fig.update_layout(title="HSV été par année",height=360)
                    st.plotly_chart(fig,use_container_width=True)
            with col2:
                df_me = q(f"SELECT MONTH,AVG(MESURE) AS avg_hsv FROM {VIEW} {wh_ete} GROUP BY MONTH ORDER BY MONTH")
                if not df_me.empty:
                    df_me["MOIS"] = df_me["MONTH"].map({6:"Juin",7:"Juillet",8:"Août"})
                    fig = go.Figure(go.Bar(x=df_me["MOIS"],y=df_me["avg_hsv"],
                        marker_color=["#0ea5e9","#38bdf8","#7dd3fc"],
                        text=df_me["avg_hsv"].map(lambda v:f"{v:.3f} m"),textposition="outside"))
                    apply_theme(fig); fig.update_layout(title="HSV par mois d'été",height=360)
                    st.plotly_chart(fig,use_container_width=True)

        with tab2:
            mwd_wh = where_clause_with_extra("MONTH IN (6,7,8) AND mwd IS NOT NULL")
            df_mwd = q(f"SELECT FLOOR(mwd/22.5)*22.5 AS dir_bin,COUNT(*) AS n,AVG(MESURE) AS avg_hsv FROM {VIEW} {mwd_wh} GROUP BY dir_bin ORDER BY dir_bin")
            if not df_mwd.empty:
                fig3 = go.Figure(go.Barpolar(
                    r=df_mwd["n"],theta=df_mwd["dir_bin"],width=22,
                    marker=dict(color=df_mwd["avg_hsv"],colorscale="RdYlGn_r",
                        colorbar=dict(title=dict(text="HSV moy (m)",side="right"),thickness=14),showscale=True)
                ))
                apply_theme(fig3)
                fig3.update_layout(title="Rose des vagues — été",
                    polar=dict(bgcolor="rgba(4,18,32,.6)",
                        angularaxis=dict(tickmode="array",tickvals=[0,45,90,135,180,225,270,315],
                            ticktext=["N","NE","E","SE","S","SO","O","NO"],direction="clockwise",rotation=90,
                            tickfont=dict(color="#94b8cc",size=12),linecolor="#1e3a4f"),
                        radialaxis=dict(tickfont=dict(color="#94b8cc",size=10),gridcolor="#1e3a4f")),height=480)
                st.plotly_chart(fig3,use_container_width=True)

            section("🌬️","wind_speed mensuel — Juin · Juillet · Août")
            ws_ete_wh = where_clause_with_extra("MONTH IN (6,7,8)")
            df_ws_ete = q(f"""
                SELECT MONTH,
                       ROUND(AVG(wind_speed),3) AS avg_ws, ROUND(MAX(wind_speed),2) AS max_ws,
                       ROUND(STDDEV(wind_speed),3) AS std_ws,
                       ROUND(SUM(CASE WHEN wind_speed>=10 THEN 1 ELSE 0 END)*100.0/COUNT(*),1) AS pct_fort,
                       ROUND(SUM(CASE WHEN wind_speed>=5  THEN 1 ELSE 0 END)*100.0/COUNT(*),1) AS pct_mod
                FROM {VIEW} {ws_ete_wh} GROUP BY MONTH ORDER BY MONTH
            """)
            if not df_ws_ete.empty:
                df_ws_ete["MOIS_L"] = df_ws_ete["MONTH"].map({6:"Juin",7:"Juillet",8:"Août"})
                col_w1, col_w2 = st.columns(2)
                with col_w1:
                    fig_ws = go.Figure()
                    fig_ws.add_trace(go.Scatter(x=df_ws_ete["MOIS_L"],y=df_ws_ete["avg_ws"]+df_ws_ete["std_ws"],fill=None,mode="lines",line=dict(width=0),showlegend=False,hoverinfo="skip"))
                    fig_ws.add_trace(go.Scatter(x=df_ws_ete["MOIS_L"],y=df_ws_ete["avg_ws"]-df_ws_ete["std_ws"],fill="tonexty",mode="lines",line=dict(width=0),fillcolor="rgba(6,182,212,.12)",showlegend=False,hoverinfo="skip"))
                    fig_ws.add_trace(go.Scatter(x=df_ws_ete["MOIS_L"],y=df_ws_ete["avg_ws"],name="wind_speed moyen",mode="lines+markers",line=dict(color="#06b6d4",width=2.5),marker=dict(size=9,color="#06b6d4")))
                    fig_ws.add_trace(go.Scatter(x=df_ws_ete["MOIS_L"],y=df_ws_ete["max_ws"],name="wind_speed max",mode="lines+markers",line=dict(color="#f97316",width=1.5,dash="dot"),marker=dict(size=7,symbol="diamond")))
                    for seuil,clr,lbl in [(10,"#ef4444","Danger 10 m/s"),(5,"#f59e0b","Modéré 5 m/s")]:
                        fig_ws.add_hline(y=seuil,line_dash="dash",line_color=clr,annotation_text=lbl,annotation_font_color=clr,annotation_font_size=10)
                    apply_theme(fig_ws); fig_ws.update_layout(title="wind_speed moyen et max — été",xaxis_title="Mois",yaxis_title="Vent (m/s)",height=360)
                    st.plotly_chart(fig_ws,use_container_width=True)
                with col_w2:
                    fig_ws2 = go.Figure()
                    fig_ws2.add_trace(go.Bar(x=df_ws_ete["MOIS_L"],y=df_ws_ete["pct_fort"],name="% vent fort ≥ 10 m/s",
                        marker_color=df_ws_ete["pct_fort"].apply(lambda v:"#ef4444" if v>20 else "#f59e0b" if v>10 else "#10b981"),
                        text=df_ws_ete["pct_fort"].map(lambda v:f"{v:.1f}%"),textposition="outside"))
                    fig_ws2.add_trace(go.Scatter(x=df_ws_ete["MOIS_L"],y=df_ws_ete["pct_mod"],name="% vent modéré ≥ 5 m/s",mode="lines+markers",line=dict(color="#06b6d4",width=2,dash="dot"),marker=dict(size=8)))
                    apply_theme(fig_ws2); fig_ws2.update_layout(barmode="overlay",title="% occurrences vent fort / modéré — été",xaxis_title="Mois",yaxis_title="%",height=360)
                    st.plotly_chart(fig_ws2,use_container_width=True)

        with tab3:
            df_pl = q(f"""
                SELECT NOM_PLAGE,NOM_WILAYA,
                       ROUND(AVG(MESURE),4) AS avg_hsv,ROUND(MAX(MESURE),4) AS max_hsv,
                       ROUND(AVG(wind_speed),4) AS avg_ws,ROUND(AVG(mwp),4) AS avg_mwp,
                       ROUND(SUM(CASE WHEN MESURE>=2 THEN 1 ELSE 0 END)*100.0/COUNT(*),2) AS pct_danger
                FROM {VIEW} {wh_ete} GROUP BY NOM_PLAGE,NOM_WILAYA
            """)
            if not df_pl.empty:
                df_pl["avg_mwp"] = df_pl["avg_mwp"].replace(0,None).fillna(df_pl["avg_mwp"].median())
                df_pl["risk_score"] = (0.4*df_pl["avg_hsv"]+0.2*df_pl["max_hsv"]+0.2*(df_pl["pct_danger"]/100)+0.1*df_pl["avg_ws"]+0.1*(1/df_pl["avg_mwp"])).round(4)
                df_pl = df_pl.sort_values("risk_score",ascending=False).head(30).reset_index(drop=True)
                fig = go.Figure(go.Bar(x=df_pl["risk_score"],y=df_pl["NOM_PLAGE"],orientation="h",
                    marker_color=df_pl["risk_score"].apply(_risk_color),
                    text=df_pl["risk_score"].map(lambda v:f"{v:.4f}"),textposition="outside"))
                apply_theme(fig); fig.update_layout(title="Top 30 plages — Risk Score été",xaxis_title="Risk Score",
                    yaxis=dict(autorange="reversed"),height=max(380,30*22),showlegend=False)
                st.plotly_chart(fig,use_container_width=True)
    else:
        st.info("👈 Sélectionnez un type d'analyse dans le menu de gauche.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE : ACTIVITÉS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == T("activities"):

    if activity_page == T("drowning_alerts"):
        page_header("#ef4444","🏊","Alertes Noyades","MESURE · wind_speed · mwp · mwd — Sécurité balnéaire")
        st.markdown("""
        <div class="info-card" style="border-left-color:#ef4444;background:rgba(239,68,68,.06);">
            <div class="title" style="color:#f87171;">Seuils — Variables M1 : MESURE · wind_speed · mwp · mwd</div>
            <div class="threshold-row"><span class="threshold-key">🟢 Calme</span><span class="threshold-val" style="color:#34d399;">MESURE &lt; 1.0 m · wind_speed &lt; 5 m/s</span></div>
            <div class="threshold-row"><span class="threshold-key">🟡 Vigilance</span><span class="threshold-val" style="color:#fbbf24;">MESURE 1–2 m · wind_speed 5–10 m/s</span></div>
            <div class="threshold-row"><span class="threshold-key">🔴 Danger</span><span class="threshold-val" style="color:#f87171;">MESURE &gt; 2 m · wind_speed &gt; 10 m/s · mwp &gt; 8 s</span></div>
        </div>""",unsafe_allow_html=True)
        wh = W(); show_kpis(wh)

        tab1,tab2,tab3 = st.tabs([T("alerts"),T("wind_mwd"),T("by_beach")])

        with tab1:
            col1,col2 = st.columns(2)
            with col1:
                section("📊","Répartition globale alertes")
                df_al = q(f"SELECT ALERTE,COUNT(*) AS n FROM {VIEW} {wh} GROUP BY ALERTE")
                if not df_al.empty:
                    ord_al = ['Calme (< 1 m)','Vigilance (1–2 m)','Danger (> 2 m)']
                    df_al["ALERTE"] = pd.Categorical(df_al["ALERTE"],categories=ord_al,ordered=True)
                    df_al = df_al.sort_values("ALERTE"); total_al = df_al["n"].sum()
                    for _,row in df_al.iterrows():
                        pct=row["n"]/total_al*100; color=ALERTE_COLORS.get(row["ALERTE"],"#666")
                        st.markdown(f'<div style="background:var(--bg-card);border:1px solid var(--border-s);border-left:3px solid {color};border-radius:8px;padding:10px 14px;margin-bottom:8px;display:flex;justify-content:space-between;"><span>{row["ALERTE"]}</span><b style="color:{color};">{pct:.1f}%</b></div>',unsafe_allow_html=True)
            with col2:
                df_ann_al = q(f"""
                    SELECT YEAR,
                           SUM(CASE WHEN MESURE>=2 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_danger,
                           SUM(CASE WHEN MESURE>=1 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_vig,
                           SUM(CASE WHEN wind_speed>=10 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_vent
                    FROM {VIEW} {wh} GROUP BY YEAR ORDER BY YEAR
                """)
                if not df_ann_al.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_ann_al["YEAR"],y=df_ann_al["pct_vig"],name="Vigilance MESURE",line=dict(color="#f59e0b")))
                    fig.add_trace(go.Scatter(x=df_ann_al["YEAR"],y=df_ann_al["pct_danger"],name="Danger MESURE",line=dict(color="#ef4444"),fill="tozeroy",fillcolor="rgba(239,68,68,.08)"))
                    fig.add_trace(go.Scatter(x=df_ann_al["YEAR"],y=df_ann_al["pct_vent"],name="wind_speed ≥10 m/s",line=dict(color="#06b6d4",dash="dot")))
                    apply_theme(fig); fig.update_layout(title="% annuel alertes",xaxis_title="Année",yaxis_title="%",height=360)
                    st.plotly_chart(fig,use_container_width=True)

            section("⏱️","Distribution MWP (Période des vagues)")
            mwp_wh = where_clause_with_extra("mwp IS NOT NULL")
            df_mwp = q(f"SELECT ROUND(mwp,0) AS mwp_r,COUNT(*) AS n FROM {VIEW} {mwp_wh} GROUP BY mwp_r ORDER BY mwp_r")
            if not df_mwp.empty:
                fig = go.Figure(go.Bar(x=df_mwp["mwp_r"],y=df_mwp["n"],
                    marker_color=df_mwp["mwp_r"].apply(lambda v:"#ef4444" if v>8 else "#f59e0b" if v>6 else "#10b981")))
                fig.add_vline(x=8,line_dash="dash",line_color="#ef4444",annotation_text="Danger >8s")
                apply_theme(fig); fig.update_layout(title="Distribution mwp — risque noyade",xaxis_title="mwp (s)",height=300)
                st.plotly_chart(fig,use_container_width=True)

        with tab2:
            col1,col2 = st.columns(2)
            with col1:
                df_ws_m = q(f"""
                    SELECT MONTH,AVG(wind_speed) AS avg_ws,MAX(wind_speed) AS max_ws,
                           SUM(CASE WHEN wind_speed>=10 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_fort
                    FROM {VIEW} {wh} GROUP BY MONTH ORDER BY MONTH
                """)
                if not df_ws_m.empty:
                    df_ws_m["MOIS_L"] = df_ws_m["MONTH"].map(T("months_short"))
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_ws_m["MOIS_L"],y=df_ws_m["avg_ws"],name="wind_speed moy",mode="lines+markers",line=dict(color="#06b6d4",width=2)))
                    fig.add_trace(go.Bar(x=df_ws_m["MOIS_L"],y=df_ws_m["pct_fort"],name="% fort ≥10",marker_color="rgba(239,68,68,.5)",yaxis="y2"))
                    fig.add_hline(y=10,line_dash="dash",line_color="#f59e0b",annotation_text="10 m/s")
                    apply_theme(fig)
                    fig.update_layout(title="wind_speed mensuel",yaxis=dict(title="Vent (m/s)",**PLOTLY_THEME["yaxis"]),
                        yaxis2=dict(title="% fort",overlaying="y",side="right",gridcolor="rgba(0,0,0,0)",tickcolor="#4a7a96",tickfont=dict(color="#94b8cc")),height=360)
                    st.plotly_chart(fig,use_container_width=True)
            with col2:
                mwd_wh = where_clause_with_extra("mwd IS NOT NULL")
                df_mwd = q(f"SELECT FLOOR(mwd/22.5)*22.5 AS dir_bin,COUNT(*) AS n,AVG(MESURE) AS avg_hsv FROM {VIEW} {mwd_wh} GROUP BY dir_bin ORDER BY dir_bin")
                if not df_mwd.empty:
                    fig2 = go.Figure(go.Barpolar(r=df_mwd["n"],theta=df_mwd["dir_bin"],width=22,
                        marker_color=df_mwd["avg_hsv"],marker_colorscale=[[0,"#10b981"],[.5,"#f59e0b"],[1,"#ef4444"]],marker_showscale=True))
                    apply_theme(fig2)
                    fig2.update_layout(title="Rose des vagues (mwd)",
                        polar=dict(bgcolor="rgba(12,24,41,.8)",
                            angularaxis=dict(tickmode="array",tickvals=[0,45,90,135,180,225,270,315],ticktext=["N","NE","E","SE","S","SO","O","NO"],direction="clockwise",rotation=90,gridcolor="rgba(30,90,150,.3)",tickfont=dict(color="#94b8cc")),
                            radialaxis=dict(gridcolor="rgba(30,90,150,.2)",tickfont=dict(color="#94b8cc"))),height=400)
                    st.plotly_chart(fig2,use_container_width=True)

        with tab3:
            df_pl_al = q(f"""
                SELECT NOM_PLAGE,NOM_WILAYA,
                       AVG(MESURE) AS avg_hsv,MAX(MESURE) AS max_hsv,
                       AVG(wind_speed) AS avg_ws,MAX(wind_speed) AS max_ws,
                       AVG(mwp) AS avg_mwp,MAX(mwp) AS max_mwp,AVG(mwd) AS avg_mwd,
                       SUM(CASE WHEN MESURE>=2 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_danger,
                       SUM(CASE WHEN wind_speed>=10 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_vent_fort
                FROM {VIEW} {wh} GROUP BY NOM_PLAGE,NOM_WILAYA ORDER BY pct_danger DESC LIMIT 30
            """)
            if not df_pl_al.empty:
                df_pl_al["avg_mwp"] = df_pl_al["avg_mwp"].fillna(df_pl_al["avg_mwp"].median())
                df_pl_al["risk_score"] = ((df_pl_al["avg_hsv"]/3)*0.5+(df_pl_al["avg_ws"]/15)*0.3+(df_pl_al["avg_mwp"]/10)*0.2).round(4)
                fig = px.scatter(df_pl_al,x="avg_hsv",y="avg_ws",size="risk_score",color="NOM_WILAYA",hover_name="NOM_PLAGE",
                    hover_data={"NOM_WILAYA":True,"avg_hsv":":.3f","max_hsv":":.2f","avg_ws":":.2f","avg_mwp":":.2f","avg_mwd":":.1f","pct_danger":":.1f","pct_vent_fort":":.1f","risk_score":":.4f","max_ws":False,"max_mwp":False},
                    labels={"avg_hsv":"MESURE moy (m)","avg_ws":"wind_speed moy (m/s)"})
                fig.add_vline(x=2.0,line_dash="dash",line_color="#ef4444",annotation_text="MESURE 2m")
                fig.add_hline(y=10,line_dash="dash",line_color="#f59e0b",annotation_text="wind_speed 10 m/s")
                apply_theme(fig); fig.update_layout(title="Risque MESURE × wind_speed par plage",height=420)
                st.plotly_chart(fig,use_container_width=True)
                df_show = df_pl_al[["NOM_PLAGE","NOM_WILAYA","avg_hsv","max_hsv","avg_ws","avg_mwp","pct_danger","pct_vent_fort","risk_score"]].round(3)
                df_show.columns = ["Plage","Wilaya","MESURE Moy","MESURE Max","wind_speed Moy","mwp Moy","% Danger","% Vent Fort","⚠️ Risk Score"]
                st.dataframe(df_show,use_container_width=True,hide_index=True)

    elif activity_page == T("desalination"):
        if not is_m2:
            st.warning(T("m1_drowning_only"))
            st.info("👆 Sélectionnez **🟣 M2 — ERA5 + CMEMS** dans la sidebar pour accéder au Dessalement SWRO.")
            st.stop()
        page_header("#0ea5e9","💧","Dessalement SWRO","salinity · spm · MESURE · mwp · wind_speed — osmose inverse")
        st.markdown("""
        <div class="info-card" style="border-left-color:#0ea5e9;background:rgba(14,165,233,.06);">
            <div class="title" style="color:#38bdf8;">Variables M2 — Dessalement SWRO</div>
            <div class="threshold-row"><span class="threshold-key">🧂 salinity optimale</span><span class="threshold-val" style="color:#38bdf8;">36–38 PSU · Alerte si &lt; 35 ou &gt; 39</span></div>
            <div class="threshold-row"><span class="threshold-key">🌫️ spm (turbidité)</span><span class="threshold-val" style="color:#fbbf24;">Alerte si &gt; 0.1 m⁻¹ (colmatage membranes)</span></div>
            <div class="threshold-row"><span class="threshold-key">🌊 MESURE prise d'eau</span><span class="threshold-val" style="color:#f87171;">Alerte si &gt; 3.0 m</span></div>
            <div class="threshold-row"><span class="threshold-key">🌬️ wind_speed</span><span class="threshold-val" style="color:#f87171;">&gt; 10 m/s → turbidité accrue</span></div>
            <div class="threshold-row"><span class="threshold-key">🌊 mwp</span><span class="threshold-val">Période des vagues — impact prise d'eau</span></div>
        </div>""",unsafe_allow_html=True)
        wh = W(); show_kpis(wh)

        tab1,tab2 = st.tabs(["🧂 Salinité & SPM",T("op_windows")])

        with tab1:
            col1,col2 = st.columns(2)
            with col1:
                sal_wh = where_clause_with_extra("salinity IS NOT NULL")
                df_sal_m = q(f"SELECT MONTH,AVG(salinity) AS avg_sal,MIN(salinity) AS min_sal,MAX(salinity) AS max_sal,STDDEV(salinity) AS std_sal FROM {VIEW} {sal_wh} GROUP BY MONTH ORDER BY MONTH")
                if not df_sal_m.empty:
                    df_sal_m["MOIS_L"] = df_sal_m["MONTH"].map(T("months_short"))
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_sal_m["MOIS_L"],y=df_sal_m["avg_sal"]+df_sal_m["std_sal"],fill=None,mode="lines",line=dict(width=0),showlegend=False))
                    fig.add_trace(go.Scatter(x=df_sal_m["MOIS_L"],y=df_sal_m["avg_sal"]-df_sal_m["std_sal"],fill="tonexty",mode="lines",line=dict(width=0),fillcolor="rgba(34,211,238,.1)",showlegend=False))
                    fig.add_trace(go.Scatter(x=df_sal_m["MOIS_L"],y=df_sal_m["avg_sal"],name="salinity moy",mode="lines+markers",line=dict(color="#22d3ee",width=2)))
                    for val,color,label in [(35,"#ef4444","Min 35 PSU"),(38,"#f59e0b","Max opt 38 PSU"),(39,"#f97316","Seuil max 39 PSU")]:
                        fig.add_hline(y=val,line_dash="dash",line_color=color,annotation_text=label,annotation_font_color=color)
                    apply_theme(fig); fig.update_layout(title="Salinité mensuelle — dessalement",yaxis_title="salinity (PSU)",height=340)
                    st.plotly_chart(fig,use_container_width=True)
            with col2:
                spm_wh = where_clause_with_extra("spm IS NOT NULL")
                df_spm_m = q(f"SELECT MONTH,AVG(spm) AS avg_spm,PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY spm) AS p90_spm FROM {VIEW} {spm_wh} GROUP BY MONTH ORDER BY MONTH")
                if not df_spm_m.empty:
                    df_spm_m["MOIS_L"] = df_spm_m["MONTH"].map(T("months_short"))
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=df_spm_m["MOIS_L"],y=df_spm_m["avg_spm"],name="spm moy",
                        marker_color=df_spm_m["avg_spm"].apply(lambda v:"#ef4444" if v>0.1 else "#f59e0b" if v>0.05 else "#10b981"),
                        text=df_spm_m["avg_spm"].map(lambda v:f"{v:.4f}"),textposition="outside"))
                    fig.add_trace(go.Scatter(x=df_spm_m["MOIS_L"],y=df_spm_m["p90_spm"],mode="lines+markers",name="P90 spm",line=dict(color="#f59e0b",width=2)))
                    fig.add_hline(y=0.1,line_dash="dash",line_color="#ef4444",annotation_text="Seuil turbidité 0.1 m⁻¹")
                    apply_theme(fig); fig.update_layout(title="SPM mensuel — colmatage membranes",yaxis_title="spm/KD490 (m⁻¹)",height=340)
                    st.plotly_chart(fig,use_container_width=True)

            section("🌊","MESURE · mwp · wind_speed — impact prise d'eau")
            col3,col4 = st.columns(2)
            with col3:
                df_hsv_m = q(f"SELECT MONTH,AVG(MESURE) AS avg_hsv,AVG(mwp) AS avg_mwp,SUM(CASE WHEN MESURE>3.0 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_alert FROM {VIEW} {wh} GROUP BY MONTH ORDER BY MONTH")
                if not df_hsv_m.empty:
                    df_hsv_m["MOIS_L"] = df_hsv_m["MONTH"].map(T("months_short"))
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_hsv_m["MOIS_L"],y=df_hsv_m["avg_hsv"],name="MESURE moy (m)",mode="lines+markers",line=dict(color="#0ea5e9",width=2)))
                    fig.add_trace(go.Scatter(x=df_hsv_m["MOIS_L"],y=df_hsv_m["avg_mwp"],name="mwp moy (s)",mode="lines",line=dict(color="#a78bfa",dash="dot",width=2),yaxis="y2"))
                    fig.add_hline(y=3.0,line_dash="dash",line_color="#ef4444",annotation_text="Alerte MESURE 3m")
                    apply_theme(fig)
                    fig.update_layout(title="MESURE & mwp mensuel",yaxis=dict(title="MESURE (m)",**PLOTLY_THEME["yaxis"]),
                        yaxis2=dict(title="mwp (s)",overlaying="y",side="right",gridcolor="rgba(0,0,0,0)",tickcolor="#4a7a96",tickfont=dict(color="#94b8cc")),height=340)
                    st.plotly_chart(fig,use_container_width=True)
            with col4:
                df_ws_m = q(f"SELECT MONTH,AVG(wind_speed) AS avg_ws,SUM(CASE WHEN wind_speed>=10 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_fort FROM {VIEW} {wh} GROUP BY MONTH ORDER BY MONTH")
                if not df_ws_m.empty:
                    df_ws_m["MOIS_L"] = df_ws_m["MONTH"].map(T("months_short"))
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=df_ws_m["MOIS_L"],y=df_ws_m["avg_ws"],name="wind_speed moy",
                        marker_color=df_ws_m["avg_ws"].apply(lambda v:"#ef4444" if v>10 else "#f59e0b" if v>6 else "#10b981")))
                    fig.add_trace(go.Scatter(x=df_ws_m["MOIS_L"],y=df_ws_m["pct_fort"],name="% fort ≥10",mode="lines",line=dict(color="#f97316",width=2),yaxis="y2"))
                    fig.add_hline(y=10,line_dash="dash",line_color="#ef4444",annotation_text="Seuil 10 m/s")
                    apply_theme(fig)
                    fig.update_layout(title="wind_speed mensuel",yaxis=dict(title="m/s",**PLOTLY_THEME["yaxis"]),
                        yaxis2=dict(title="% fort",overlaying="y",side="right",gridcolor="rgba(0,0,0,0)",tickcolor="#4a7a96",tickfont=dict(color="#94b8cc")),height=340)
                    st.plotly_chart(fig,use_container_width=True)

        with tab2:
            ops_wh = where_clause_with_extra("salinity IS NOT NULL AND spm IS NOT NULL")
            df_ops = q(f"""
                SELECT MONTH,
                       SUM(CASE WHEN salinity BETWEEN 35 AND 39 AND spm < 0.1 AND wind_speed < 10 AND MESURE < 3.0 AND mwp < 8 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_ok_all,
                       SUM(CASE WHEN salinity BETWEEN 35 AND 39 AND spm < 0.1 AND MESURE < 3.0 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_ok_base
                FROM {VIEW} {ops_wh} GROUP BY MONTH ORDER BY MONTH
            """)
            if not df_ops.empty:
                df_ops["MOIS_L"] = df_ops["MONTH"].map(T("months"))
                col1,col2 = st.columns(2)
                with col1:
                    fig = go.Figure(go.Bar(x=df_ops["MOIS_L"],y=df_ops["pct_ok_all"],
                        marker_color=df_ops["pct_ok_all"].apply(lambda v:"#10b981" if v>80 else "#f59e0b" if v>60 else "#ef4444"),
                        text=df_ops["pct_ok_all"].map(lambda v:f"{v:.1f}%"),textposition="outside"))
                    apply_theme(fig); fig.update_layout(title="% optimal SWRO (salinity+spm+MESURE+wind+mwp)",yaxis_title="%",height=360)
                    st.plotly_chart(fig,use_container_width=True)
                with col2:
                    fig2 = go.Figure()
                    fig2.add_trace(go.Bar(x=df_ops["MOIS_L"],y=df_ops["pct_ok_base"],name="Sans vent+mwp",marker_color="#0ea5e9"))
                    fig2.add_trace(go.Bar(x=df_ops["MOIS_L"],y=df_ops["pct_ok_all"],name="Tous critères",marker_color="#ef4444"))
                    apply_theme(fig2); fig2.update_layout(barmode="overlay",title="Impact wind_speed+mwp sur disponibilité SWRO",yaxis_title="%",height=360)
                    st.plotly_chart(fig2,use_container_width=True)
            else:
                st.info("Données insuffisantes pour les fenêtres opérationnelles.")

    elif activity_page == T("aquaculture"):
        if not is_m2:
            st.warning(T("m1_drowning_only"))
            st.info("👆 Sélectionnez **🟣 M2 — ERA5 + CMEMS** dans la sidebar pour accéder à l'Aquaculture.")
            st.stop()
        page_header("#10b981","🐟","Aquaculture Marine","o2 · sst · mwp · wind_speed · MESURE · spm — Variables M2")
        st.markdown("""
        <div class="info-card" style="border-left-color:#10b981;background:rgba(16,185,129,.06);">
            <div class="title" style="color:#34d399;">Variables M2 — Aquaculture (o2 · sst · mwp · wind_speed · MESURE · spm)</div>
            <div class="threshold-row"><span class="threshold-key">🫧 o2 dissous</span><span class="threshold-val" style="color:#34d399;">&gt; 200 mmol/m³ — survie des espèces</span></div>
            <div class="threshold-row"><span class="threshold-key">🌡️ sst croissance</span><span class="threshold-val" style="color:#34d399;">16 °C – 24 °C</span></div>
            <div class="threshold-row"><span class="threshold-key">🌊 mwp</span><span class="threshold-val">&lt; 8 s — sécurité cages</span></div>
            <div class="threshold-row"><span class="threshold-key">🌬️ wind_speed</span><span class="threshold-val">&lt; 8 m/s — stabilité structures</span></div>
            <div class="threshold-row"><span class="threshold-key">🌊 MESURE</span><span class="threshold-val" style="color:#34d399;">&lt; 1.2 m — sécurité cages flottantes</span></div>
            <div class="threshold-row"><span class="threshold-key">🌫️ spm</span><span class="threshold-val">&lt; 0.1 m⁻¹ — turbidité acceptable</span></div>
        </div>""",unsafe_allow_html=True)
        wh = W(); show_kpis(wh)

        tab1,tab2,tab3,tab4 = st.tabs([T("favorable_windows"),"🫧 O₂ & SST","🌊 MESURE & SPM",T("best_sites")])

        with tab1:
            aqua_wh = where_clause_with_extra("o2 IS NOT NULL AND sst IS NOT NULL AND spm IS NOT NULL")
            df_aqua_m = q(f"""
                SELECT MONTH,
                       SUM(CASE WHEN o2>200 AND sst BETWEEN 16 AND 24 AND mwp<8 AND wind_speed<8 AND MESURE<1.2 AND spm<0.1 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_ok_all,
                       SUM(CASE WHEN MESURE<1.2 AND sst BETWEEN 16 AND 24 AND mwp<8 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_ok_base,
                       AVG(MESURE) AS avg_hsv, AVG(sst) AS avg_sst, AVG(wind_speed) AS avg_ws, AVG(o2) AS avg_o2, AVG(spm) AS avg_spm
                FROM {VIEW} {aqua_wh} GROUP BY MONTH ORDER BY MONTH
            """)
            if not df_aqua_m.empty:
                df_aqua_m["MOIS_L"] = df_aqua_m["MONTH"].map(T("months_short"))
                fig = go.Figure()
                fig.add_trace(go.Bar(x=df_aqua_m["MOIS_L"],y=df_aqua_m["pct_ok_base"],name="MESURE+sst+mwp",marker_color="rgba(14,165,233,.5)"))
                fig.add_trace(go.Bar(x=df_aqua_m["MOIS_L"],y=df_aqua_m["pct_ok_all"],name="Tous critères (o2+spm inclus)",
                    marker_color=df_aqua_m["pct_ok_all"].apply(lambda v:"#10b981" if v>70 else "#f59e0b" if v>40 else "#ef4444"),
                    text=df_aqua_m["pct_ok_all"].map(lambda v:f"{v:.0f}%"),textposition="outside"))
                apply_theme(fig); fig.update_layout(barmode="overlay",title="% conditions favorables aquaculture",yaxis_title="%",height=360)
                st.plotly_chart(fig,use_container_width=True)

        with tab2:
            col1,col2 = st.columns(2)
            with col1:
                o2_wh = where_clause_with_extra("o2 IS NOT NULL")
                df_o2_m = q(f"SELECT MONTH,AVG(o2) AS avg_o2,MIN(o2) AS min_o2,STDDEV(o2) AS std_o2 FROM {VIEW} {o2_wh} GROUP BY MONTH ORDER BY MONTH")
                if not df_o2_m.empty:
                    df_o2_m["MOIS_L"] = df_o2_m["MONTH"].map(T("months_short"))
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_o2_m["MOIS_L"],y=df_o2_m["avg_o2"]+df_o2_m["std_o2"],fill=None,mode="lines",line=dict(width=0),showlegend=False))
                    fig.add_trace(go.Scatter(x=df_o2_m["MOIS_L"],y=df_o2_m["avg_o2"]-df_o2_m["std_o2"],fill="tonexty",mode="lines",line=dict(width=0),fillcolor="rgba(139,92,246,.1)",showlegend=False))
                    fig.add_trace(go.Scatter(x=df_o2_m["MOIS_L"],y=df_o2_m["avg_o2"],name="o2 moy",mode="lines+markers",line=dict(color="#a78bfa",width=2)))
                    fig.add_hline(y=200,line_dash="dash",line_color="#ef4444",annotation_text="Seuil hypoxie 200 mmol/m³")
                    apply_theme(fig); fig.update_layout(title="o2 dissous mensuel",yaxis_title="o2 (mmol/m³)",height=340)
                    st.plotly_chart(fig,use_container_width=True)
            with col2:
                sst_wh = where_clause_with_extra("sst IS NOT NULL")
                df_sst_m = q(f"SELECT MONTH,AVG(sst) AS avg_sst,MIN(sst) AS min_sst,MAX(sst) AS max_sst,STDDEV(sst) AS std_sst FROM {VIEW} {sst_wh} GROUP BY MONTH ORDER BY MONTH")
                if not df_sst_m.empty:
                    df_sst_m["MOIS_L"] = df_sst_m["MONTH"].map(T("months_short"))
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_sst_m["MOIS_L"],y=df_sst_m["avg_sst"]+df_sst_m["std_sst"],fill=None,mode="lines",line=dict(width=0),showlegend=False))
                    fig.add_trace(go.Scatter(x=df_sst_m["MOIS_L"],y=df_sst_m["avg_sst"]-df_sst_m["std_sst"],fill="tonexty",mode="lines",line=dict(width=0),fillcolor="rgba(6,182,212,.1)",showlegend=False))
                    fig.add_trace(go.Scatter(x=df_sst_m["MOIS_L"],y=df_sst_m["avg_sst"],name="sst moy",mode="lines+markers",line=dict(color="#06b6d4",width=2)))
                    for val,color,label in [(16,"#f59e0b","16°C"),(24,"#ef4444","24°C")]:
                        fig.add_hline(y=val,line_dash="dash",line_color=color,annotation_text=label,annotation_font_color=color)
                    apply_theme(fig); fig.update_layout(title="sst mensuelle",yaxis_title="sst (°C)",height=340)
                    st.plotly_chart(fig,use_container_width=True)

        with tab3:
            col1,col2 = st.columns(2)
            with col1:
                df_mesure_m = q(f"SELECT MONTH,AVG(MESURE) AS avg_hsv,AVG(mwp) AS avg_mwp,SUM(CASE WHEN MESURE>=1.2 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_alert FROM {VIEW} {wh} GROUP BY MONTH ORDER BY MONTH")
                if not df_mesure_m.empty:
                    df_mesure_m["MOIS_L"] = df_mesure_m["MONTH"].map(T("months_short"))
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=df_mesure_m["MOIS_L"],y=df_mesure_m["avg_hsv"],name="MESURE moy",
                        marker_color=df_mesure_m["avg_hsv"].apply(lambda v:"#ef4444" if v>=1.2 else "#f59e0b" if v>=0.8 else "#10b981")))
                    fig.add_trace(go.Scatter(x=df_mesure_m["MOIS_L"],y=df_mesure_m["avg_mwp"],name="mwp moy (s)",mode="lines",line=dict(color="#a78bfa",width=2,dash="dot"),yaxis="y2"))
                    fig.add_hline(y=1.2,line_dash="dash",line_color="#ef4444",annotation_text="Seuil MESURE 1.2m")
                    apply_theme(fig)
                    fig.update_layout(title="MESURE & mwp mensuel — sécurité cages",yaxis=dict(title="MESURE (m)",**PLOTLY_THEME["yaxis"]),
                        yaxis2=dict(title="mwp (s)",overlaying="y",side="right",gridcolor="rgba(0,0,0,0)",tickcolor="#4a7a96",tickfont=dict(color="#94b8cc")),height=340)
                    st.plotly_chart(fig,use_container_width=True)
            with col2:
                spm_aqua_wh = where_clause_with_extra("spm IS NOT NULL")
                df_spm_aqua = q(f"SELECT MONTH,AVG(spm) AS avg_spm,SUM(CASE WHEN spm>0.1 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_turbide FROM {VIEW} {spm_aqua_wh} GROUP BY MONTH ORDER BY MONTH")
                if not df_spm_aqua.empty:
                    df_spm_aqua["MOIS_L"] = df_spm_aqua["MONTH"].map(T("months_short"))
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=df_spm_aqua["MOIS_L"],y=df_spm_aqua["avg_spm"],name="spm moy",
                        marker_color=df_spm_aqua["avg_spm"].apply(lambda v:"#ef4444" if v>0.1 else "#f59e0b" if v>0.05 else "#10b981")))
                    fig.add_hline(y=0.1,line_dash="dash",line_color="#ef4444",annotation_text="Seuil turbidité 0.1 m⁻¹")
                    apply_theme(fig); fig.update_layout(title="spm mensuel",yaxis_title="spm (m⁻¹)",height=340)
                    st.plotly_chart(fig,use_container_width=True)

        with tab4:
            sites_wh = where_clause_with_extra("o2 IS NOT NULL AND sst IS NOT NULL AND spm IS NOT NULL")
            df_sites = q(f"""
                SELECT NOM_PLAGE,NOM_WILAYA,FIRST(X) AS X,FIRST(Y) AS Y,
                       SUM(CASE WHEN o2>200 AND sst BETWEEN 16 AND 24 AND mwp<8 AND wind_speed<8 AND MESURE<1.2 AND spm<0.1 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_ok,
                       AVG(MESURE) AS avg_hsv, AVG(sst) AS avg_sst, AVG(wind_speed) AS avg_ws, AVG(o2) AS avg_o2, AVG(spm) AS avg_spm
                FROM {VIEW} {sites_wh} GROUP BY NOM_PLAGE,NOM_WILAYA
            """)
            if not df_sites.empty:
                df_sites["avg_sst"] = df_sites["avg_sst"].fillna(20.0)
                df_sites["avg_o2"]  = df_sites["avg_o2"].fillna(200.0)
                df_sites["avg_spm"] = df_sites["avg_spm"].fillna(0.05)
                df_sites["risk_aqua"] = (
                    df_sites["avg_hsv"]/2*0.25 + df_sites["avg_ws"]/10*0.2 +
                    (df_sites["avg_sst"]-20.0).abs()/8.0*0.2 +
                    (df_sites["avg_o2"].clip(0,300).apply(lambda v: max(0,(200-v)/200))*0.2) +
                    df_sites["avg_spm"].clip(0,0.2)/0.2*0.15
                ).round(4)
                df_top = df_sites.sort_values("pct_ok",ascending=False).head(10)
                for _,row in df_top.iterrows():
                    score = row["pct_ok"]; color = "#10b981" if score>60 else "#f59e0b" if score>35 else "#ef4444"
                    st.markdown(f'<div style="background:var(--bg-card);border:1px solid var(--border-s);border-left:3px solid {color};border-radius:var(--r-md);padding:10px 14px;margin-bottom:6px;display:flex;align-items:center;justify-content:space-between;"><div><span style="font-size:.85rem;font-weight:600;color:var(--text-h);">{row["NOM_PLAGE"]}</span><span style="font-size:.75rem;color:var(--text-m);margin-left:8px;">{row["NOM_WILAYA"]}</span></div><div style="text-align:right;"><span style="font-family:var(--fm);font-size:.95rem;font-weight:700;color:{color};">{score:.1f}% fav.</span><span style="font-size:.72rem;color:var(--text-m);margin-left:6px;">MESURE:{row["avg_hsv"]:.2f}m · o2:{row["avg_o2"]:.0f} · sst:{row["avg_sst"]:.1f}°C · spm:{row["avg_spm"]:.4f}</span></div></div>',unsafe_allow_html=True)
                if df_sites[["X","Y"]].notna().all().all():
                    fig = px.scatter_mapbox(df_sites,lat="Y",lon="X",color="risk_aqua",size="pct_ok",
                        hover_name="NOM_PLAGE",hover_data={"NOM_WILAYA":True,"avg_hsv":":.2f","avg_ws":":.1f","avg_sst":":.1f","avg_o2":":.0f","avg_spm":":.5f","pct_ok":":.1f","risk_aqua":":.4f","X":False,"Y":False},
                        color_continuous_scale=["#10b981","#f59e0b","#ef4444"],range_color=[0,.8],size_max=18,zoom=5,
                        mapbox_style="carto-darkmatter",labels={"risk_aqua":"Risk Aqua","pct_ok":"% fav."})
                    apply_theme(fig); fig.update_layout(height=450,margin=dict(l=0,r=0,t=40,b=0),title="Sites aquaculture — Risk Score")
                    st.plotly_chart(fig,use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE : SYNTHÈSE & EXPORT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == T("synthesis"):
    page_header("#8b5cf6","📋","Synthèse & Export","Tableaux récapitulatifs et téléchargement")
    wh = W()
    tab1,tab2,tab3 = st.tabs([T("synth_by_beach"),T("monthly_synth"),T("export")])

    with tab1:
        section("📊","Statistiques complètes par plage")
        with st.spinner("Calcul..."):
            cols_extra = ""
            if has_col(VIEW,"mwp"):        cols_extra += ",AVG(mwp) AS avg_mwp"
            if has_col(VIEW,"wind_speed"): cols_extra += ",AVG(wind_speed) AS avg_ws,MAX(wind_speed) AS max_ws,SUM(CASE WHEN wind_speed>=10 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_vent_fort"
            if has_col(VIEW,"mwd"):        cols_extra += ",AVG(mwd) AS avg_mwd"
            if is_m2:
                if has_col(VIEW,"sst"):      cols_extra += ",AVG(sst) AS avg_sst"
                if has_col(VIEW,"salinity"): cols_extra += ",AVG(salinity) AS avg_sal"
                if has_col(VIEW,"o2"):       cols_extra += ",AVG(o2) AS avg_o2,SUM(CASE WHEN o2<200 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_hypoxie"
                if has_col(VIEW,"spm"):      cols_extra += ",AVG(spm) AS avg_spm"
            df_synth = q(f"""
                SELECT NOM_PLAGE,NOM_WILAYA,COUNT(*) AS n,
                       ROUND(AVG(MESURE),3) AS avg_hsv,ROUND(MAX(MESURE),2) AS max_hsv,
                       ROUND(STDDEV(MESURE),3) AS std_hsv,
                       ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY MESURE),2) AS p95,
                       ROUND(SUM(CASE WHEN MESURE>=1 THEN 1 ELSE 0 END)*100.0/COUNT(*),1) AS pct_vig,
                       ROUND(SUM(CASE WHEN MESURE>=2 THEN 1 ELSE 0 END)*100.0/COUNT(*),1) AS pct_danger,
                       ROUND(SUM(CASE WHEN AQUA_OK THEN 1 ELSE 0 END)*100.0/COUNT(*),1) AS pct_aqua,
                       ROUND(SUM(CASE WHEN DESSAL_OK THEN 1 ELSE 0 END)*100.0/COUNT(*),1) AS pct_dessal
                       {cols_extra}
                FROM {VIEW} {wh} GROUP BY NOM_PLAGE,NOM_WILAYA ORDER BY avg_hsv DESC
            """)
        if not df_synth.empty:
            st.dataframe(df_synth,use_container_width=True,hide_index=True)
            csv = df_synth.to_csv(index=False).encode("utf-8")
            st.download_button(T("download_csv"),csv,"synthese_plages.csv","text/csv")

    with tab2:
        section("📅","Statistiques mensuelles globales")
        with st.spinner():
            cols_m = ""
            if has_col(VIEW,"mwp"):        cols_m += ",ROUND(AVG(mwp),2) AS avg_mwp"
            if has_col(VIEW,"msl"):        cols_m += ",ROUND(AVG(msl),1) AS avg_msl"
            if has_col(VIEW,"wind_speed"): cols_m += ",ROUND(AVG(wind_speed),2) AS avg_ws,ROUND(MAX(wind_speed),1) AS max_ws,ROUND(SUM(CASE WHEN wind_speed>=10 THEN 1 ELSE 0 END)*100.0/COUNT(*),1) AS pct_vent_fort"
            if is_m2:
                if has_col(VIEW,"sst"):      cols_m += ",ROUND(AVG(sst),2) AS avg_sst"
                if has_col(VIEW,"salinity"): cols_m += ",ROUND(AVG(salinity),3) AS avg_sal"
                if has_col(VIEW,"o2"):       cols_m += ",ROUND(AVG(o2),2) AS avg_o2,ROUND(SUM(CASE WHEN o2<200 THEN 1 ELSE 0 END)*100.0/COUNT(*),2) AS pct_hypoxie"
                if has_col(VIEW,"spm"):      cols_m += ",ROUND(AVG(spm),5) AS avg_spm"
            df_mo = q(f"""
                SELECT MONTH,COUNT(*) AS n,
                       ROUND(AVG(MESURE),3) AS avg_hsv,ROUND(MAX(MESURE),2) AS max_hsv,ROUND(STDDEV(MESURE),3) AS std_hsv,
                       ROUND(SUM(CASE WHEN MESURE>=1 THEN 1 ELSE 0 END)*100.0/COUNT(*),1) AS pct_vig,
                       ROUND(SUM(CASE WHEN MESURE>=2 THEN 1 ELSE 0 END)*100.0/COUNT(*),1) AS pct_danger,
                       ROUND(SUM(CASE WHEN AQUA_OK THEN 1 ELSE 0 END)*100.0/COUNT(*),1) AS pct_aqua,
                       ROUND(SUM(CASE WHEN DESSAL_OK THEN 1 ELSE 0 END)*100.0/COUNT(*),1) AS pct_dessal
                       {cols_m}
                FROM {VIEW} {wh} GROUP BY MONTH ORDER BY MONTH
            """)
        if not df_mo.empty:
            df_mo["MOIS"] = df_mo["MONTH"].map(T("months"))
            df_mo = df_mo.drop(columns=["MONTH"])
            cols_ord = ["MOIS"] + [c for c in df_mo.columns if c!="MOIS"]
            st.dataframe(df_mo[cols_ord],use_container_width=True,hide_index=True)
            csv2 = df_mo[cols_ord].to_csv(index=False).encode("utf-8")
            st.download_button(T("download_csv")+" (mensuel)",csv2,"synthese_mensuelle.csv","text/csv")

    with tab3:
        section("💾","Export personnalisé")
        base_cols = ["NOM_PLAGE","NOM_WILAYA","DATETIME","MESURE","ALERTE","NIVEAU","wind_speed","mwp","mwd","DISTANCE","SEASON","YEAR","MONTH"]
        if is_m2: base_cols += ["sst","salinity","o2","spm","msl"]
        col_choices = st.multiselect("Colonnes à exporter", base_cols, default=["NOM_PLAGE","NOM_WILAYA","DATETIME","MESURE","ALERTE","NIVEAU","wind_speed","mwp","mwd"])
        max_rows = st.slider("Nombre maximum de lignes",1000,100000,10000,1000)
        if col_choices:
            with st.spinner("Extraction..."):
                df_exp = q(f"SELECT {', '.join(col_choices)} FROM {VIEW} {wh} LIMIT {max_rows}")
            if not df_exp.empty:
                st.dataframe(df_exp,use_container_width=True,hide_index=True)
                csv3 = df_exp.to_csv(index=False).encode("utf-8")
                st.download_button(T("download_csv"),csv3,"export_hsv.csv","text/csv")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE : CARTE DES DANGERS  — VERSION CORRIGÉE
# Corrections :
#   1. Pas de duplication de la colonne "val" dans extra_sel
#   2. Critères Dessalement/Aquaculture calculés directement dans val_sql
#   3. Filtre > 0 désactivé pour Dessalement/Aquaculture (évite carte vide)
#   4. HAVING utilise FIRST(X) au lieu de lon (alias non résolu dans HAVING)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == T("danger_map"):
    page_header("#ef4444","🗺️","Carte des Dangers","Cartographie interactive — côtes algériennes")
    wh = W()

    MAP_OPTIONS = [T("avg_hsv_map"), T("max_hsv_map"), T("alert_m1_map")]
    if is_m2:
        MAP_OPTIONS += [T("dessal_map"), T("aqua_map")]
    map_metric = st.radio(T("indicator_mapped"), MAP_OPTIONS, horizontal=True)

    with st.spinner(T("loading_map")):

        # ── Configuration selon la métrique sélectionnée ──────────────────────
        if map_metric == T("avg_hsv_map"):
            val_sql      = "ROUND(AVG(MESURE), 3)"
            label        = "HSV Moy (m)"
            cscale       = "Blues"
            extra_sel    = ", ROUND(AVG(MESURE),3) AS avg_hsv, ROUND(MAX(MESURE),2) AS max_hsv"
            hover_fields = {"NOM_WILAYA":True,"avg_hsv":":.3f","max_hsv":":.2f","lon":False,"lat":False}

        elif map_metric == T("max_hsv_map"):
            val_sql      = "ROUND(MAX(MESURE), 2)"
            label        = "HSV Max (m)"
            cscale       = "Reds"
            extra_sel    = ", ROUND(AVG(MESURE),3) AS avg_hsv, ROUND(MAX(MESURE),2) AS max_hsv"
            hover_fields = {"NOM_WILAYA":True,"avg_hsv":":.3f","max_hsv":":.2f","lon":False,"lat":False}

        elif map_metric == T("alert_m1_map"):
            val_sql      = "ROUND(SUM(CASE WHEN MESURE>=2 THEN 1 ELSE 0 END)*100.0/COUNT(*), 2)"
            label        = "% Danger noyade"
            cscale       = ["#10b981","#f59e0b","#ef4444"]
            extra_sel    = (
                ", ROUND(AVG(MESURE),3) AS avg_hsv"
                ", ROUND(MAX(MESURE),2) AS max_hsv"
                ", ROUND(AVG(wind_speed),2) AS avg_ws"
                ", ROUND(AVG(mwp),2) AS avg_mwp"
                ", ROUND(SUM(CASE WHEN MESURE>=2 THEN 1 ELSE 0 END)*100.0/COUNT(*),2) AS pct_danger"
            )
            hover_fields = {
                "NOM_WILAYA":True,"avg_hsv":":.3f","max_hsv":":.2f",
                "avg_ws":":.2f","avg_mwp":":.2f","pct_danger":":.1f",
                "lon":False,"lat":False
            }

        elif map_metric == T("dessal_map"):
            # ✅ CORRECTION : calcul direct MESURE<=3.0, pas de duplication de val
            val_sql      = "ROUND(SUM(CASE WHEN MESURE<=3.0 THEN 1 ELSE 0 END)*100.0/COUNT(*), 2)"
            label        = "% Dessalement OK"
            cscale       = ["#ef4444","#f59e0b","#10b981"]
            extra_sel    = (
                ", ROUND(AVG(MESURE),3) AS avg_hsv"
                ", ROUND(AVG(wind_speed),2) AS avg_ws"
                ", ROUND(SUM(CASE WHEN MESURE<=3.0 THEN 1 ELSE 0 END)*100.0/COUNT(*),2) AS pct_dessal"
            )
            hover_fields = {
                "NOM_WILAYA":True,"avg_hsv":":.3f","avg_ws":":.2f","pct_dessal":":.1f",
                "lon":False,"lat":False
            }

        elif map_metric == T("aqua_map"):
            # ✅ CORRECTION : calcul direct MESURE<1.2, pas de duplication de val
            val_sql      = "ROUND(SUM(CASE WHEN MESURE<1.2 THEN 1 ELSE 0 END)*100.0/COUNT(*), 2)"
            label        = "% Aquaculture OK"
            cscale       = ["#ef4444","#f59e0b","#10b981"]
            extra_sel    = (
                ", ROUND(AVG(MESURE),3) AS avg_hsv"
                ", ROUND(AVG(wind_speed),2) AS avg_ws"
                ", ROUND(AVG(sst),1) AS avg_sst"
                ", ROUND(AVG(o2),0) AS avg_o2"
                ", ROUND(SUM(CASE WHEN MESURE<1.2 THEN 1 ELSE 0 END)*100.0/COUNT(*),2) AS pct_aqua"
            )
            hover_fields = {
                "NOM_WILAYA":True,"avg_hsv":":.3f","avg_ws":":.2f",
                "avg_sst":":.1f","avg_o2":":.0f","pct_aqua":":.1f",
                "lon":False,"lat":False
            }

        else:
            val_sql      = "ROUND(AVG(MESURE), 3)"
            label        = "HSV Moy (m)"
            cscale       = "Blues"
            extra_sel    = ", ROUND(AVG(MESURE),3) AS avg_hsv"
            hover_fields = {"NOM_WILAYA":True,"lon":False,"lat":False}

        # ── Requête principale ─────────────────────────────────────────────────
        # ✅ CORRECTION : HAVING sur FIRST(X)/FIRST(Y) (alias lon/lat non dispo dans HAVING)
        df_map = q(f"""
            SELECT
                NOM_PLAGE,
                NOM_WILAYA,
                FIRST(X) AS lon,
                FIRST(Y) AS lat,
                {val_sql} AS val
                {extra_sel}
            FROM {VIEW} {wh}
            GROUP BY NOM_PLAGE, NOM_WILAYA
            HAVING FIRST(X) IS NOT NULL
               AND FIRST(Y) IS NOT NULL
        """)

    if not df_map.empty:
        # Nettoyage : supprimer les lignes avec coordonnées ou val manquants
        df_map = df_map.dropna(subset=["val", "lon", "lat"])

        # ✅ CORRECTION : pour Dessalement et Aquaculture, on GARDE les 0%
        # (une plage à 0% de conditions favorables est quand même informative sur la carte)
        if map_metric not in [T("dessal_map"), T("aqua_map")]:
            df_map = df_map[df_map["val"] > 0].copy()
        else:
            df_map = df_map.copy()

        # Taille des marqueurs : valeur absolue, clippée à 0.1 minimum pour éviter size=0
        df_map["_size"] = df_map["val"].abs().clip(lower=0.1)

        if df_map.empty:
            st.warning("Aucune plage ne satisfait les critères sélectionnés.")
        else:
            # Filtrer hover_fields sur les colonnes réellement présentes
            hf_clean = {
                k: v for k, v in hover_fields.items()
                if k in df_map.columns or k in ["lon", "lat"]
            }

            fig = px.scatter_mapbox(
                df_map,
                lat="lat", lon="lon",
                color="val",
                size="_size",
                hover_name="NOM_PLAGE",
                hover_data=hf_clean,
                color_continuous_scale=cscale,
                size_max=22,
                zoom=5,
                mapbox_style="carto-darkmatter",
                labels={"val": label, "_size": ""}
            )
            apply_theme(fig)
            fig.update_layout(
                height=520,
                margin=dict(l=0, r=0, t=40, b=0),
                title=f"Côtes algériennes — {map_metric}"
            )
            st.plotly_chart(fig, use_container_width=True)

            section("📊", T("by_wilaya"))
            df_wil = q(f"""
                SELECT NOM_WILAYA,
                       COUNT(DISTINCT NOM_PLAGE) AS nb_plages,
                       ROUND(AVG(MESURE),3)  AS avg_hsv,
                       ROUND(MAX(MESURE),2)  AS max_hsv,
                       ROUND(SUM(CASE WHEN MESURE>=2 THEN 1 ELSE 0 END)*100.0/COUNT(*),1) AS pct_danger
                FROM {VIEW} {wh}
                GROUP BY NOM_WILAYA
                ORDER BY avg_hsv DESC
            """)
            if not df_wil.empty:
                st.dataframe(df_wil, use_container_width=True, hide_index=True)
    else:
        st.warning("Données cartographiques non disponibles.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE : PRÉDICTION TEMPS RÉEL
# ═══════════════════════════════════════════════════════════════════════════════
elif page == T("realtime_pred"):

    PRED_M1_LABEL = T("pred_m1_label")
    PRED_M2_LABEL = T("pred_m2_label")
    pred_wants_m1 = (pred_model_page == PRED_M1_LABEL)
    pred_wants_m2 = (pred_model_page == PRED_M2_LABEL)

    if pred_wants_m2 and not is_m2:
        st.markdown("""
        <div class="incompat-banner">
            <div class="ib-icon">🚫</div>
            <div class="ib-title">Incompatibilité de modèle</div>
            <div class="ib-body">Vous avez sélectionné <strong>Prédiction M2 — ERA5 + CMEMS</strong>
            mais le modèle de données actif est <strong>🔵 M1 — ERA5 seul</strong>.<br><br>
            Le modèle M2 nécessite les variables enrichies CMEMS (Salinité, O₂ dissous, KD490)
            ainsi que ERA5 (wind_speed, MWP, SST) qui ne sont pas disponibles dans M1.</div>
            <div class="ib-step">👉 <strong>Solution :</strong> Dans la sidebar, section
            <em>MODÈLE DE DONNÉES</em>, sélectionnez
            <strong>🟣 M2 — ERA5 + CMEMS (1999–2023)</strong> puis revenez sur cette page.</div>
        </div>""", unsafe_allow_html=True)
        st.stop()

    elif pred_wants_m1 and is_m2:
        st.markdown("""
        <div class="incompat-banner">
            <div class="ib-icon">🚫</div>
            <div class="ib-title">Incompatibilité de modèle</div>
            <div class="ib-body">Vous avez sélectionné <strong>Prédiction M1 — ERA5 seul</strong>
            mais le modèle de données actif est <strong>🟣 M2 — ERA5 + CMEMS</strong>.<br><br>
            Le pipeline M1 utilise exclusivement les features ERA5 (V7).</div>
            <div class="ib-step">👉 <strong>Solution :</strong> Dans la sidebar, section
            <em>MODÈLE DE DONNÉES</em>, sélectionnez
            <strong>🔵 M1 — ERA5 seul (1985–2023)</strong> puis revenez sur cette page.</div>
        </div>""", unsafe_allow_html=True)
        st.stop()

    elif pred_wants_m2 and is_m2:

        page_header(
            "#8b5cf6", "🟣", "Prédiction Temps Réel — M2",
            "ERA5 (wind · MWP · SST) + CMEMS (Sal · O₂ · KD490) + Open-Meteo Marine (HSV) → LSTM → HSV J+1 · J+7"
        )

        WINDOW_DAYS_M2 = 30
        HORIZONS_M2    = [1, 7]

        FEATURES_DESSAL_M2 = [
            "wind_speed", "mwp", "salinity", "spm",
            "hour_sin", "hour_cos", "month_sin", "month_cos",
            "x_norm", "y_norm"
        ]
        FEATURES_AQUA_M2 = [
            "wind_speed", "mwp", "o2", "sst",
            "hour_sin", "hour_cos", "month_sin", "month_cos",
            "x_norm", "y_norm"
        ]

        LAT_MIN_M2, LAT_MAX_M2 = 36.70, 37.10
        LON_MIN_M2, LON_MAX_M2 = -1.80,  8.70

        SEUIL_HSV_DESSAL = 1.4
        SEUIL_HSV_AQUA   = 1.2

        PATH_DESSAL_MODEL  = "models_m2/global_lstm_dessal.keras"
        PATH_AQUA_MODEL    = "models_m2/global_lstm_aqua.keras"
        PATH_SCALER_DESSAL = "models_m2/scaler_dessal.pkl"
        PATH_SCALER_AQUA   = "models_m2/scaler_aqua.pkl"

        import traceback as _tb

        try:
            import joblib
            import tensorflow as tf
        except ImportError as _e:
            st.error(f"❌ Dépendance manquante : {_e}\npip install tensorflow joblib")
            st.stop()

        _missing_m2 = [
            p for p in [PATH_DESSAL_MODEL, PATH_AQUA_MODEL,
                        PATH_SCALER_DESSAL, PATH_SCALER_AQUA]
            if not os.path.exists(p)
        ]
        if _missing_m2:
            st.error("❌ Fichiers M2 manquants :\n" + "\n".join(f"  • `{p}`" for p in _missing_m2))
            st.stop()

        def _loss_m2(y_true, y_pred):
            return tf.reduce_mean(tf.abs(y_true - y_pred))
        _loss_m2.__name__ = "mae"
        CUSTOM_OBJ_M2 = {"mae": _loss_m2, "asymmetric_huber": _loss_m2, "mse": _loss_m2}

        @st.cache_resource(show_spinner="Chargement modèle Dessalement M2...")
        def _load_model_dessal():
            return tf.keras.models.load_model(PATH_DESSAL_MODEL, custom_objects=CUSTOM_OBJ_M2, compile=False)

        @st.cache_resource(show_spinner="Chargement modèle Aquaculture M2...")
        def _load_model_aqua():
            return tf.keras.models.load_model(PATH_AQUA_MODEL, custom_objects=CUSTOM_OBJ_M2, compile=False)

        @st.cache_resource(show_spinner="Chargement scaler Dessalement...")
        def _load_sc_dessal():
            return joblib.load(PATH_SCALER_DESSAL)

        @st.cache_resource(show_spinner="Chargement scaler Aquaculture...")
        def _load_sc_aqua():
            return joblib.load(PATH_SCALER_AQUA)

        @st.cache_resource(show_spinner=False)
        def _load_finetuned_m2(plage_name):
            if not plage_name:
                return None, False
            safe_name  = plage_name.replace(" ", "_")
            model_path = os.path.join("models_m2", "plots", safe_name, "best_local.keras")
            if os.path.exists(model_path):
                try:
                    model = tf.keras.models.load_model(model_path, custom_objects=CUSTOM_OBJ_M2, compile=False)
                    return model, True
                except Exception as e:
                    st.warning(f"⚠️ Fine-tuné M2 non chargeable : {e}")
                    return None, False
            return None, False

        def _fetch_hsv_openmeteo(lat, lon, date_start, date_end):
            import urllib.request as _ur
            import json as _js
            url = (
                f"https://marine-api.open-meteo.com/v1/marine?"
                f"latitude={lat:.4f}&longitude={lon:.4f}"
                f"&hourly=wave_height"
                f"&start_date={date_start.strftime('%Y-%m-%d')}"
                f"&end_date={date_end.strftime('%Y-%m-%d')}"
                f"&timezone=UTC"
            )
            try:
                with _ur.urlopen(url, timeout=30) as r:
                    d = _js.loads(r.read())
                h  = d["hourly"]
                n  = len(h["time"])
                df_h = pd.DataFrame({
                    "DATETIME": pd.to_datetime(h["time"]),
                    "MESURE"  : pd.to_numeric(h.get("wave_height", [np.nan] * n), errors="coerce"),
                })
                df_h["DATETIME"] = df_h["DATETIME"].dt.tz_localize(None)
                df_h["_D"]       = df_h["DATETIME"].dt.normalize()
                df_daily = (
                    df_h.groupby("_D")
                    .agg(MESURE=("MESURE", "mean"))
                    .reset_index()
                    .rename(columns={"_D": "DATETIME"})
                )
                st.caption(f"✅ HSV Open-Meteo Marine : {len(df_daily)} jours récupérés")
                return df_daily
            except Exception as e:
                st.warning(f"⚠️ Open-Meteo Marine (HSV) : {e} → valeur par défaut 0.5m")
                dates = pd.date_range(date_start, date_end, freq="D")
                return pd.DataFrame({"DATETIME": dates, "MESURE": 0.5})

        def _fetch_era5_openmeteo_fallback_m2(lat, lon, date_start, date_end):
            import urllib.request as _ur
            import json as _js

            start_str = pd.Timestamp(date_start).strftime("%Y-%m-%d")
            end_str   = pd.Timestamp(date_end).strftime("%Y-%m-%d")
            dates     = pd.date_range(pd.Timestamp(date_start), pd.Timestamp(date_end), freq="D")
            df_result = pd.DataFrame({"DATETIME": dates})

            def _marine_var(var_api, col_out, default_val):
                url = (
                    f"https://marine-api.open-meteo.com/v1/marine?"
                    f"latitude={lat:.4f}&longitude={lon:.4f}"
                    f"&hourly={var_api}"
                    f"&start_date={start_str}&end_date={end_str}"
                    f"&timezone=UTC"
                )
                try:
                    with _ur.urlopen(url, timeout=30) as r:
                        d = _js.loads(r.read())
                    h = d["hourly"]
                    n = len(h["time"])
                    df_h = pd.DataFrame({
                        "DATETIME": pd.to_datetime(h["time"]),
                        col_out   : pd.to_numeric(h.get(var_api, [np.nan]*n), errors="coerce"),
                    })
                    df_h["DATETIME"] = df_h["DATETIME"].dt.tz_localize(None)
                    df_h["_D"]       = df_h["DATETIME"].dt.normalize()
                    df_day = (
                        df_h.groupby("_D")
                        .agg(**{col_out: (col_out, "mean")})
                        .reset_index()
                        .rename(columns={"_D": "DATETIME"})
                    )
                    st.caption(f"✅ Open-Meteo Marine ({col_out}) : {len(df_day)} jours")
                    return df_day
                except Exception as e:
                    st.caption(f"ℹ️ Marine {col_out} indisponible ({e}) → défaut {default_val}")
                    return pd.DataFrame({"DATETIME": dates, col_out: default_val})

            df_mwp = None
            for _mwp_var in ["wave_period_mean", "swell_wave_period", "wind_wave_period"]:
                _df_try = _marine_var(_mwp_var, "mwp", None)
                if "mwp" in _df_try.columns and not _df_try["mwp"].isna().all():
                    df_mwp = _df_try
                    break

            if df_mwp is None or df_mwp["mwp"].isna().all():
                _df_sw  = _marine_var("swell_wave_period", "_sw",  None)
                _df_ww  = _marine_var("wind_wave_period",  "_ww",  None)
                _merged = _df_sw.merge(_df_ww, on="DATETIME", how="outer")
                _cols   = [c for c in ["_sw", "_ww"] if c in _merged.columns]
                if _cols:
                    _merged["mwp"] = _merged[_cols].mean(axis=1)
                    df_mwp = _merged[["DATETIME", "mwp"]]
                    st.caption(f"✅ MWP (moyenne swell+wind) : {len(df_mwp)} jours")
                else:
                    df_mwp = pd.DataFrame({"DATETIME": dates, "mwp": 4.0})
                    st.caption("ℹ️ MWP → valeur par défaut 4.0 s")

            df_result = df_result.merge(df_mwp, on="DATETIME", how="left")

            df_sst = _marine_var("sea_surface_temperature", "sst", None)
            if "sst" not in df_sst.columns or df_sst["sst"].isna().all():
                today_utc_sst = datetime.utcnow().date()
                base_sst = (
                    "https://historical-forecast-api.open-meteo.com/v1/forecast"
                    if pd.Timestamp(date_end).date() <= today_utc_sst
                    else "https://api.open-meteo.com/v1/forecast"
                )
                url_t2m = (
                    f"{base_sst}?latitude={lat:.4f}&longitude={lon:.4f}"
                    f"&hourly=temperature_2m&start_date={start_str}&end_date={end_str}"
                    f"&timezone=UTC"
                )
                try:
                    with _ur.urlopen(url_t2m, timeout=30) as r:
                        d = _js.loads(r.read())
                    h = d["hourly"]
                    n = len(h["time"])
                    df_t2 = pd.DataFrame({
                        "DATETIME": pd.to_datetime(h["time"]),
                        "sst"     : pd.to_numeric(h.get("temperature_2m", [np.nan]*n), errors="coerce"),
                    })
                    df_t2["DATETIME"] = df_t2["DATETIME"].dt.tz_localize(None)
                    df_t2["_D"]       = df_t2["DATETIME"].dt.normalize()
                    df_sst = (
                        df_t2.groupby("_D")
                        .agg(sst=("sst", "mean"))
                        .reset_index()
                        .rename(columns={"_D": "DATETIME"})
                    )
                    st.caption(f"✅ SST approx. (T air 2m) : {len(df_sst)} jours")
                except Exception:
                    df_sst = pd.DataFrame({"DATETIME": dates, "sst": 20.0})
                    st.caption("ℹ️ SST → valeur par défaut 20.0 °C")
            df_result = df_result.merge(df_sst, on="DATETIME", how="left")

            today_utc = datetime.utcnow().date()
            if pd.Timestamp(date_end).date() <= today_utc:
                base_url_atm = "https://historical-forecast-api.open-meteo.com/v1/forecast"
            else:
                base_url_atm = "https://api.open-meteo.com/v1/forecast"

            url_atm = (
                f"{base_url_atm}?"
                f"latitude={lat:.4f}&longitude={lon:.4f}"
                f"&hourly=wind_speed_10m"
                f"&start_date={start_str}&end_date={end_str}"
                f"&timezone=UTC"
                f"&wind_speed_unit=ms"
            )
            try:
                with _ur.urlopen(url_atm, timeout=30) as r:
                    d = _js.loads(r.read())
                h = d["hourly"]
                n = len(h["time"])
                df_atm = pd.DataFrame({
                    "DATETIME"   : pd.to_datetime(h["time"]),
                    "wind_speed" : pd.to_numeric(h.get("wind_speed_10m", [np.nan]*n), errors="coerce"),
                })
                df_atm["DATETIME"] = df_atm["DATETIME"].dt.tz_localize(None)
                df_atm["_D"]       = df_atm["DATETIME"].dt.normalize()
                df_atm = (
                    df_atm.groupby("_D")
                    .agg(wind_speed=("wind_speed", "mean"))
                    .reset_index()
                    .rename(columns={"_D": "DATETIME"})
                )
                st.caption(f"✅ Open-Meteo Atmosphérique (wind_speed) : {len(df_atm)} jours")
                df_result = df_result.merge(df_atm, on="DATETIME", how="left")
            except Exception as e:
                st.warning(f"⚠️ Open-Meteo Atmosphérique (wind_speed) : {e} → valeur par défaut 5.0 m/s")
                df_result["wind_speed"] = 5.0

            df_result["mwp"]        = df_result["mwp"].fillna(4.0)
            df_result["sst"]        = df_result["sst"].fillna(20.0)
            df_result["wind_speed"] = df_result["wind_speed"].fillna(5.0)
            df_result["mwp"]        = np.clip(df_result["mwp"],        0, 25)
            df_result["sst"]        = np.clip(df_result["sst"],        0, 35)
            df_result["wind_speed"] = np.clip(df_result["wind_speed"], 0, 60)

            return df_result.sort_values("DATETIME").reset_index(drop=True)

        def _fetch_era5_daily(lat, lon, date_start, date_end, cds_key=""):
            ERA5_CUTOFF = datetime.utcnow() - timedelta(days=37)
            if pd.Timestamp(date_start) >= pd.Timestamp(ERA5_CUTOFF):
                st.caption("ℹ️ Période hors couverture ERA5 (délai ~37j) → Open-Meteo fallback")
                return _fetch_era5_openmeteo_fallback_m2(lat, lon, date_start, date_end)
            if not cds_key.strip():
                st.caption("ℹ️ Clé CDS absente → Open-Meteo fallback (wind_speed, mwp, sst)")
                return _fetch_era5_openmeteo_fallback_m2(lat, lon, date_start, date_end)
            try:
                import cdsapi
            except ImportError:
                st.caption("ℹ️ cdsapi non installé → Open-Meteo fallback")
                return _fetch_era5_openmeteo_fallback_m2(lat, lon, date_start, date_end)

            effective_end = min(pd.Timestamp(date_end), pd.Timestamp(ERA5_CUTOFF))
            try:
                import tempfile, xarray as xr
                c = cdsapi.Client(key=cds_key.strip(), quiet=True)
                years  = list(range(pd.Timestamp(date_start).year, effective_end.year + 1))
                months = list(range(1, 13))
                days   = [str(d).zfill(2) for d in range(1, 32)]
                with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
                    tmp_path = tmp.name
                c.retrieve(
                    "reanalysis-era5-single-levels",
                    {
                        "product_type": "reanalysis",
                        "variable"    : ["10m_u_component_of_wind","10m_v_component_of_wind","mean_wave_period","sea_surface_temperature"],
                        "year"        : [str(y) for y in years],
                        "month"       : [str(m).zfill(2) for m in months],
                        "day"         : days,
                        "time"        : ["00:00", "06:00", "12:00", "18:00"],
                        "area"        : [lat+0.5, lon-0.5, lat-0.5, lon+0.5],
                        "format"      : "netcdf",
                    },
                    tmp_path,
                )
                ds = xr.open_dataset(tmp_path)
                df_era = ds.to_dataframe().reset_index()
                df_era = df_era.rename(columns={"valid_time":"DATETIME","u10":"_u10","v10":"_v10","mwp":"mwp","sst":"sst"})
                df_era["wind_speed"] = np.sqrt(df_era["_u10"]**2 + df_era["_v10"]**2)
                df_era["DATETIME"]   = pd.to_datetime(df_era["DATETIME"]).dt.tz_localize(None)
                if df_era["sst"].mean() > 200:
                    df_era["sst"] = df_era["sst"] - 273.15
                df_era["_D"] = df_era["DATETIME"].dt.normalize()
                df_daily = (
                    df_era.groupby("_D")
                    .agg(wind_speed=("wind_speed","mean"),mwp=("mwp","mean"),sst=("sst","mean"))
                    .reset_index()
                    .rename(columns={"_D":"DATETIME"})
                )
                df_daily = df_daily[
                    (df_daily["DATETIME"] >= pd.Timestamp(date_start)) &
                    (df_daily["DATETIME"] <= pd.Timestamp(effective_end))
                ].reset_index(drop=True)
                st.caption(f"✅ ERA5 CDS : {len(df_daily)} jours (wind_speed, mwp, sst)")
                if pd.Timestamp(date_end) > effective_end:
                    gap_start = effective_end + timedelta(days=1)
                    st.caption(f"ℹ️ Gap ERA5 {gap_start.date()} → {pd.Timestamp(date_end).date()} → Open-Meteo fallback")
                    df_gap = _fetch_era5_openmeteo_fallback_m2(lat, lon, gap_start, date_end)
                    df_daily = pd.concat([df_daily, df_gap], ignore_index=True)
                return df_daily.sort_values("DATETIME").reset_index(drop=True)
            except Exception as e:
                st.warning(f"⚠️ ERA5 CDS échoué : {e} → Open-Meteo fallback")
                return _fetch_era5_openmeteo_fallback_m2(lat, lon, date_start, date_end)

        def _fetch_cmems_daily(lat, lon, date_start, date_end, cmems_user, cmems_pass, variables):
            try:
                import copernicusmarine as cm
            except ImportError:
                raise RuntimeError("pip install copernicusmarine")
            import os
            os.environ["COPERNICUSMARINE_SERVICE_USERNAME"] = cmems_user.strip()
            os.environ["COPERNICUSMARINE_SERVICE_PASSWORD"] = cmems_pass.strip()
            CUTOFF_MY = datetime.utcnow() - timedelta(days=40)
            DS_PHY_MY  = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
            DS_PHY_NRT = "cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m"
            DS_BGC_MY  = "cmems_mod_glo_bgc_my_0.25_P1D-m"
            DS_BGC_NRT = "cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m"
            DS_OPT_MY  = "cmems_obs-oc_glo_bgc-transp_my_l4-multi-4km_P1D"
            DS_OPT_NRT = "cmems_obs-oc_glo_bgc-transp_nrt_l4-gapfree-multi-4km_P1D"
            def _pick_ds(ds_my, ds_nrt):
                need_nrt = pd.Timestamp(date_end)   >= pd.Timestamp(CUTOFF_MY)
                need_my  = pd.Timestamp(date_start) <  pd.Timestamp(CUTOFF_MY)
                if need_nrt and not need_my: return ds_nrt
                elif need_my and not need_nrt: return ds_my
                else: return ds_nrt
            def _fetch_one(dataset_id, var_name, rename_to):
                try:
                    import xarray as xr
                    ds = cm.open_dataset(
                        dataset_id=dataset_id, variables=[var_name],
                        minimum_latitude=lat-0.5, maximum_latitude=lat+0.5,
                        minimum_longitude=lon-0.5, maximum_longitude=lon+0.5,
                        start_datetime=date_start.strftime("%Y-%m-%dT00:00:00"),
                        end_datetime=date_end.strftime("%Y-%m-%dT23:59:59"),
                    )
                    df = ds[[var_name]].to_dataframe().reset_index()
                    df = df.rename(columns={"time":"DATETIME", var_name:rename_to})
                    df["DATETIME"] = pd.to_datetime(df["DATETIME"]).dt.tz_localize(None)
                    df["_D"] = df["DATETIME"].dt.normalize()
                    df_daily = (
                        df.groupby("_D")
                        .agg(**{rename_to: (rename_to, "mean")})
                        .reset_index()
                        .rename(columns={"_D":"DATETIME"})
                    )
                    st.caption(f"✅ CMEMS {rename_to} ({dataset_id[:30]}…) : {len(df_daily)} jours")
                    return df_daily
                except Exception as e:
                    st.warning(f"⚠️ CMEMS {rename_to} : {e}")
                    return pd.DataFrame()
            frames = []
            if "salinity" in variables: frames.append(_fetch_one(_pick_ds(DS_PHY_MY, DS_PHY_NRT), "so", "salinity"))
            if "o2"       in variables: frames.append(_fetch_one(_pick_ds(DS_BGC_MY, DS_BGC_NRT), "o2", "o2"))
            if "spm"      in variables: frames.append(_fetch_one(_pick_ds(DS_OPT_MY, DS_OPT_NRT), "KD490", "spm"))
            frames = [f for f in frames if not f.empty]
            if not frames: return pd.DataFrame()
            result = frames[0]
            for extra in frames[1:]:
                result = result.merge(extra, on="DATETIME", how="outer")
            return result.sort_values("DATETIME").drop_duplicates("DATETIME").reset_index(drop=True)

        def _build_features_m2(df_hsv, df_era5, df_cmems, lat, lon):
            df = df_hsv.copy()
            df["DATETIME"] = pd.to_datetime(df["DATETIME"]).dt.normalize()
            if df_era5 is not None and not df_era5.empty:
                df_era5["DATETIME"] = pd.to_datetime(df_era5["DATETIME"]).dt.normalize()
                df = df.merge(df_era5, on="DATETIME", how="left")
            if df_cmems is not None and not df_cmems.empty:
                df_cmems["DATETIME"] = pd.to_datetime(df_cmems["DATETIME"]).dt.normalize()
                df = df.merge(df_cmems, on="DATETIME", how="left")
            df = df.sort_values("DATETIME").reset_index(drop=True)
            num_cols = [c for c in df.columns if c != "DATETIME"]
            df[num_cols] = df[num_cols].interpolate(method="linear",limit=10,limit_direction="both").ffill().bfill()
            defaults = {"wind_speed":5.0,"mwp":4.0,"sst":20.0,"salinity":37.0,"spm":0.05,"o2":220.0}
            for col, val in defaults.items():
                if col not in df.columns:
                    df[col] = val
                    st.caption(f"ℹ️ '{col}' absent → valeur par défaut {val}")
            h = df["DATETIME"].dt.hour
            m = df["DATETIME"].dt.month
            df["hour_sin"]  = np.sin(2*np.pi*h/24)
            df["hour_cos"]  = np.cos(2*np.pi*h/24)
            df["month_sin"] = np.sin(2*np.pi*m/12)
            df["month_cos"] = np.cos(2*np.pi*m/12)
            df["y_norm"] = np.clip((lat-LAT_MIN_M2)/(LAT_MAX_M2-LAT_MIN_M2),0.0,1.0)
            df["x_norm"] = np.clip((lon-LON_MIN_M2)/(LON_MAX_M2-LON_MIN_M2),0.0,1.0)
            clips = {"MESURE":(0,20),"wind_speed":(0,60),"mwp":(0,25),"salinity":(30,42),"sst":(0,35),"o2":(0,400),"spm":(0,5)}
            for col,(vmin,vmax) in clips.items():
                if col in df.columns:
                    df[col] = np.clip(df[col],vmin,vmax)
            return df.fillna(0.0)

        def _prepare_tensor_m2(df, features, scaler):
            for feat in features:
                if feat not in df.columns:
                    df[feat] = 0.0
            X = df[features].values.astype(np.float32)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            n_expected = scaler.n_features_in_
            n_current  = X.shape[1]
            if n_current < n_expected:
                pad = np.zeros((X.shape[0], n_expected-n_current), dtype=np.float32)
                X   = np.hstack([X, pad])
            elif n_current > n_expected:
                X = X[:, :n_expected]
            X = scaler.transform(X)
            X = np.clip(X, -10, 10)
            if len(X) < WINDOW_DAYS_M2:
                n_missing = WINDOW_DAYS_M2 - len(X)
                pad_rows  = np.tile(X[0], (n_missing, 1))
                X         = np.vstack([pad_rows, X])
            X = X[-WINDOW_DAYS_M2:]
            return X.reshape(1, WINDOW_DAYS_M2, n_expected)

        def _eval_hsv(hsv, seuil):
            ok    = hsv <= seuil
            color = "#10b981" if ok else "#ef4444"
            label = "✅ Opérationnel" if ok else "🚫 Dangereux"
            return color, label, ok

        col_cfg2, col_res2 = st.columns([1, 1.6])

        with col_cfg2:
            section("⚙️", "Configuration M2")
            if data_ok:
                plages_m2     = q(f"SELECT DISTINCT NOM_PLAGE, FIRST(X) AS lon, FIRST(Y) AS lat FROM {VIEW} GROUP BY NOM_PLAGE ORDER BY NOM_PLAGE")
                plage_list_m2 = plages_m2["NOM_PLAGE"].tolist()
            else:
                plages_m2     = pd.DataFrame()
                plage_list_m2 = []
            if not plage_list_m2:
                st.error("❌ Aucune plage disponible dans le dataset M2.")
                st.stop()
            sel_plage_m2 = st.selectbox("🏖️ Plage", plage_list_m2, key="plage_m2")
            if not plages_m2.empty and sel_plage_m2:
                _r     = plages_m2[plages_m2["NOM_PLAGE"] == sel_plage_m2].iloc[0]
                lat_m2 = float(_r["lat"])
                lon_m2 = float(_r["lon"])
            else:
                lat_m2, lon_m2 = 36.75, 3.06
            lat_m2 = st.number_input("Latitude",  value=lat_m2, format="%.4f", key="lat_m2")
            lon_m2 = st.number_input("Longitude", value=lon_m2, format="%.4f", key="lon_m2")
            st.markdown("---")
            today_m2     = datetime.utcnow().date()
            pred_date_m2 = st.date_input(
                "📅 Date cible",
                value=today_m2 + timedelta(days=1),
                min_value=today_m2 - timedelta(days=30),
                max_value=today_m2 + timedelta(days=7),
                key="pred_date_m2",
            )
            pred_dt_m2 = datetime.combine(pred_date_m2, datetime.min.time())
            st.markdown("---")
            st.markdown("**🔑 Accès API**")
            cds_key_m2    = st.text_input("CDS API KEY (ERA5)", type="password", key="cds_m2")
            cmems_user_m2 = st.text_input("👤 CMEMS Username", key="cmems_user_m2")
            cmems_pass_m2 = st.text_input("🔒 CMEMS Password", type="password", key="cmems_pass_m2")
            if not cmems_user_m2 or not cmems_pass_m2:
                st.caption("ℹ️ Sans identifiants CMEMS → variables océanographiques imputées par interpolation.")
            st.markdown("---")
            app_choice   = st.radio("🎯 Application", ["💧 Dessalement SWRO", "🐟 Aquaculture"], key="app_choice_m2")
            is_dessal_m2 = "Dessalement" in app_choice
            safe_m2       = sel_plage_m2.replace(" ", "_") if sel_plage_m2 else ""
            local_path_m2 = os.path.join("models_m2", "plots", safe_m2, "best_local.keras")
            if os.path.exists(local_path_m2):
                st.success(f"✅ Modèle fine-tuné M2 disponible pour **{sel_plage_m2}**")
            else:
                st.info("ℹ️ Modèle global M2 utilisé pour cette plage")
            seuil_actif = SEUIL_HSV_DESSAL if is_dessal_m2 else SEUIL_HSV_AQUA
            st.markdown(
                f"<div style='background:#1e293b;border-left:4px solid #8b5cf6;padding:8px 12px;border-radius:6px;font-size:.85rem;'>"
                f"⚠️ Seuil de danger HSV : <b style='color:#f59e0b'>{seuil_actif} m</b></div>",
                unsafe_allow_html=True
            )
            run_m2 = st.button("🚀 Lancer prédiction M2", type="primary", use_container_width=True, key="run_m2")

        with col_res2:
            section("📊", "Résultats M2")

        if run_m2:
            if not sel_plage_m2:
                st.error("❌ Veuillez sélectionner une plage.")
                st.stop()
            try:
                prog = st.progress(0)
                stat = st.empty()
                if is_dessal_m2:
                    features_m2    = FEATURES_DESSAL_M2
                    app_label_m2   = "Dessalement SWRO"
                    cmems_vars_m2  = ["salinity", "spm"]
                    load_model_fn  = _load_model_dessal
                    load_scaler_fn = _load_sc_dessal
                    seuil_m2       = SEUIL_HSV_DESSAL
                else:
                    features_m2    = FEATURES_AQUA_M2
                    app_label_m2   = "Aquaculture Marine"
                    cmems_vars_m2  = ["o2"]
                    load_model_fn  = _load_model_aqua
                    load_scaler_fn = _load_sc_aqua
                    seuil_m2       = SEUIL_HSV_AQUA
                win_end   = pred_dt_m2
                win_start = win_end - timedelta(days=WINDOW_DAYS_M2)
                stat.info(f"⚙️ Chargement modèle {app_label_m2}...")
                prog.progress(5)
                local_m2, used_local_m2 = _load_finetuned_m2(sel_plage_m2)
                model_m2      = local_m2 if used_local_m2 else load_model_fn()
                scaler_m2     = load_scaler_fn()
                model_info_m2 = f"Fine-tuné M2 — {sel_plage_m2}" if used_local_m2 else f"Global M2 — {app_label_m2}"
                stat.info("🌊 Récupération HSV — Open-Meteo Marine...")
                prog.progress(15)
                df_hsv_m2 = _fetch_hsv_openmeteo(lat_m2, lon_m2, win_start, win_end)
                stat.info("🌬️ Récupération ERA5 (wind_speed=√(u10²+v10²), MWP, SST)...")
                prog.progress(30)
                df_era5_m2 = _fetch_era5_daily(lat_m2, lon_m2, win_start, win_end, cds_key_m2)
                df_cmems_m2 = pd.DataFrame()
                if cmems_user_m2.strip() and cmems_pass_m2.strip():
                    stat.info(f"🌊 CMEMS : {', '.join(cmems_vars_m2)} (NRT/MY auto)...")
                    prog.progress(48)
                    try:
                        df_cmems_m2 = _fetch_cmems_daily(lat_m2, lon_m2, win_start, win_end, cmems_user_m2, cmems_pass_m2, cmems_vars_m2)
                    except Exception as _ec:
                        st.warning(f"⚠️ CMEMS : {_ec} → valeurs par défaut utilisées")
                else:
                    st.caption("ℹ️ CMEMS ignoré (pas d'identifiants) → valeurs par défaut")
                stat.info("🔧 Construction features M2...")
                prog.progress(62)
                df_feat_m2 = _build_features_m2(df_hsv_m2, df_era5_m2, df_cmems_m2, lat_m2, lon_m2)
                if len(df_feat_m2) < 5:
                    st.error(f"❌ Données insuffisantes : seulement {len(df_feat_m2)} jours.")
                    st.stop()
                stat.info("📐 Préparation tenseur (1 × 30j × features)...")
                prog.progress(74)
                X_m2 = _prepare_tensor_m2(df_feat_m2, features_m2, scaler_m2)
                stat.info(f"🧠 Inférence LSTM {app_label_m2}...")
                prog.progress(87)
                raw_out  = model_m2.predict(X_m2, verbose=0)[0]
                pred_j1  = float(np.clip(raw_out[0], 0.05, 9.0))
                pred_j7  = float(np.clip(raw_out[1] if len(raw_out) > 1 else raw_out[0], 0.05, 9.0))
                prog.progress(100)
                stat.empty()
                c1, lbl1, ok1 = _eval_hsv(pred_j1, seuil_m2)
                c7, lbl7, ok7 = _eval_hsv(pred_j7, seuil_m2)
                with col_res2:
                    st.markdown(
                        f"<span style='background:#7c3aed;color:white;padding:4px 12px;border-radius:12px;font-size:.82em;'>"
                        f"🟣 M2 — {app_label_m2} · {model_info_m2}</span>",
                        unsafe_allow_html=True
                    )
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.success(f"✅ {sel_plage_m2} · {pred_date_m2.strftime('%d/%m/%Y')}")
                    col_j1, col_j7 = st.columns(2)
                    dt1 = pred_dt_m2 + timedelta(days=1)
                    dt7 = pred_dt_m2 + timedelta(days=7)
                    with col_j1:
                        st.metric(f"HSV J+1 · {dt1.strftime('%d/%m')}", f"{pred_j1:.2f} m")
                        st.markdown(
                            f"<div style='text-align:center;font-weight:700;color:{c1};font-size:1.1rem;'>{lbl1}</div>"
                            f"<div style='text-align:center;font-size:.78rem;color:#94b8cc;margin-top:4px;'>"
                            f"Seuil : <b style='color:{c1}'>{seuil_m2} m</b></div>",
                            unsafe_allow_html=True
                        )
                    with col_j7:
                        st.metric(f"HSV J+7 · {dt7.strftime('%d/%m')}", f"{pred_j7:.2f} m", delta=f"{pred_j7-pred_j1:+.2f} m vs J+1")
                        st.markdown(
                            f"<div style='text-align:center;font-weight:700;color:{c7};font-size:1.1rem;'>{lbl7}</div>"
                            f"<div style='text-align:center;font-size:.78rem;color:#94b8cc;margin-top:4px;'>"
                            f"Seuil : <b style='color:{c7}'>{seuil_m2} m</b></div>",
                            unsafe_allow_html=True
                        )
                    fig_m2 = go.Figure()
                    fig_m2.add_trace(go.Scatter(
                        x=[dt1, dt7], y=[pred_j1, pred_j7],
                        mode="lines+markers",
                        line=dict(width=4, color="#8b5cf6", dash="dash"),
                        marker=dict(size=16, color=[c1, c7], line=dict(width=2, color="white")),
                        name="HSV prédit"
                    ))
                    fig_m2.add_hline(y=seuil_m2, line_dash="dash", line_color="#ef4444", line_width=2,
                        annotation_text=f"⚠️ Seuil danger {seuil_m2}m", annotation_font_color="#ef4444")
                    for dt, pred, clr, lbl in [(dt1,pred_j1,c1,"J+1"),(dt7,pred_j7,c7,"J+7")]:
                        fig_m2.add_annotation(x=dt,y=pred,text=f"<b>{pred:.2f}m</b><br>{lbl}",showarrow=False,yshift=22,font=dict(size=11,color=clr))
                    apply_theme(fig_m2)
                    fig_m2.update_layout(
                        title=f"Prévision HSV — {app_label_m2} · {sel_plage_m2}",
                        yaxis_title="HSV (m)",
                        yaxis=dict(range=[0, max(pred_j1,pred_j7)*1.5+0.5]),
                        xaxis_title="Date", height=330, showlegend=False
                    )
                    st.plotly_chart(fig_m2, use_container_width=True)
                    section("📋", "Récapitulatif")
                    recap_data = {
                        "Horizon"      : ["J+1", "J+7"],
                        "Date"         : [dt1.strftime("%d/%m/%Y"), dt7.strftime("%d/%m/%Y")],
                        "HSV prédit"   : [f"{pred_j1:.2f} m", f"{pred_j7:.2f} m"],
                        f"Seuil {seuil_m2}m": [lbl1, lbl7],
                    }
                    st.dataframe(pd.DataFrame(recap_data), use_container_width=True, hide_index=True)
                    with st.expander("🔍 Données de contexte (fenêtre 30 jours)"):
                        show_cols = ["DATETIME", "MESURE", "wind_speed", "mwp"]
                        if is_dessal_m2: show_cols += ["salinity", "spm"]
                        else:            show_cols += ["o2", "sst"]
                        show_cols = [c for c in show_cols if c in df_feat_m2.columns]
                        st.dataframe(df_feat_m2[show_cols].tail(30), use_container_width=True)
                    with st.expander("⚙️ Détails techniques M2"):
                        era5_status  = f"✅ {len(df_era5_m2)} jours" if df_era5_m2 is not None and not df_era5_m2.empty else "⚠️ Non disponible → interpolation"
                        cmems_status = f"✅ {len(df_cmems_m2)} jours" if not df_cmems_m2.empty else "⚠️ Non disponible → valeurs par défaut"
                        st.markdown(
                            f"**Modèle** : {model_info_m2}  \n"
                            f"**Application** : {app_label_m2}  \n"
                            f"**Seuil HSV** : {seuil_m2} m  \n"
                            f"**Features** ({len(features_m2)}) : `{'`, `'.join(features_m2)}`  \n"
                            f"**Note** : `spm` = KD490 (renommage pipeline)  \n"
                            f"**Fenêtre** : {WINDOW_DAYS_M2} jours  \n"
                            f"**Points contexte** : {len(df_feat_m2)} jours  \n"
                            f"---  \n"
                            f"**Sources données** :  \n"
                            f"• Open-Meteo Marine → HSV : ✅ {len(df_hsv_m2)} jours  \n"
                            f"• ERA5 → wind_speed=√(u10²+v10²), MWP, SST : {era5_status}  \n"
                            f"• CMEMS → {', '.join(cmems_vars_m2)} : {cmems_status}  \n"
                            f"---  \n"
                            f"**Résultats bruts** : J+1 = {pred_j1:.4f}m · J+7 = {pred_j7:.4f}m"
                        )
            except Exception as _e_m2:
                st.error(f"❌ Erreur pipeline M2 : {_e_m2}")
                with st.expander("🐛 Traceback complet"):
                    st.code(_tb.format_exc())

    else:
        # ── CAS M1 ────────────────────────────────────────────────────────────
        page_header("#8b5cf6","🔮","Prédiction Temps Réel — M1 V7","ERA5 (contexte 48h) + Open-Meteo Marine → LSTM V7 → HSV → Alerte")

        st.markdown("""
        <div style="display:inline-flex;align-items:center;gap:8px;background:rgba(14,165,233,.1);border:1px solid rgba(14,165,233,.3);border-radius:8px;padding:6px 14px;font-size:.8rem;color:#38bdf8;margin-bottom:1rem;">
            🔵 <strong>M1 V7 — ERA5 seul</strong> &nbsp;·&nbsp; 11 features &nbsp;·&nbsp;
            LSTM Global + Transfer Learning &nbsp;·&nbsp; Horizons +1h / +6h / +12h
        </div>""", unsafe_allow_html=True)

        import glob
        import zipfile
        import traceback
        import urllib.request as _urllib_request
        import json as _json_mod

        try:
            import joblib
            import tensorflow as tf
            TF_OK = True
        except ImportError as _e:
            TF_OK = False
            st.error(f"❌ Dépendances manquantes : {_e}")
            st.stop()

        WINDOW   = 48
        HORIZONS = [1, 6, 12]
        FEATURES = ["MESURE","wind_speed","mwp","mwd_sin","mwd_cos","hour_sin","hour_cos","month_sin","month_cos","x_norm","y_norm"]
        N_FEAT   = len(FEATURES)
        LAT_MIN, LAT_MAX = 36.70, 37.10
        LON_MIN, LON_MAX = -1.80, 8.70
        SEUIL_WARN            = 0.5
        SEUIL_WATCH           = 1.0
        SEUIL_DANGER_FALLBACK = 1.40

        def _load_seuil_danger():
            seuil_path = os.path.join(os.path.dirname(LSTM_PATH),"seuil_config.json")
            if os.path.exists(seuil_path):
                try:
                    with open(seuil_path) as f:
                        cfg = _json_mod.load(f)
                    return float(cfg.get("seuil_danger_m", SEUIL_DANGER_FALLBACK))
                except Exception:
                    pass
            return SEUIL_DANGER_FALLBACK

        def asymmetric_huber_loss_v7(delta=0.5, underestimate_penalty=3.0):
            horizon_weights = tf.constant([1.0,1.2,1.5], dtype=tf.float32)
            def loss(y_true, y_pred):
                error       = y_true - y_pred
                abs_error   = tf.abs(error)
                quadratic   = tf.minimum(abs_error, delta)
                linear      = abs_error - quadratic
                huber       = 0.5*quadratic**2 + delta*linear
                weight_asym = tf.where(error>0, tf.ones_like(error)*underestimate_penalty, tf.ones_like(error))
                return tf.reduce_mean(huber*weight_asym*horizon_weights)
            loss.__name__ = "asymmetric_huber"
            return loss

        _loss_instance = asymmetric_huber_loss_v7()
        CUSTOM_OBJECTS = {"asymmetric_huber":_loss_instance}

        @st.cache_resource(show_spinner="Chargement modèle LSTM global V7...")
        def _load_global_lstm():
            return tf.keras.models.load_model(LSTM_PATH, custom_objects=CUSTOM_OBJECTS, compile=False)

        @st.cache_resource(show_spinner="Chargement scaler global V7...")
        def _load_scaler():
            return joblib.load(SCALER_PATH)

        def _load_finetuned(plage_name):
            if not plage_name: return None, False
            base_dir   = os.path.dirname(LSTM_PATH)
            safe_name  = plage_name.replace(" ","_")
            model_path = os.path.join(base_dir,"plots",f"{safe_name}_lstm.keras")
            if os.path.exists(model_path):
                try:
                    model = tf.keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS, compile=False)
                    return model, True
                except Exception as e:
                    st.warning(f"⚠️ Modèle fine-tuné non chargeable : {e}. Fallback global.")
                    return None, False
            return None, False

        def _run_inference(X_tensor, plage_name):
            local_model, used_local = _load_finetuned(plage_name)
            if used_local and local_model is not None:
                model_used = local_model; model_info = f"Fine-tuné V7 — {plage_name}"
            else:
                model_used = _load_global_lstm(); used_local = False; model_info = "Global V7 (fallback)"
            preds = model_used.predict(X_tensor, verbose=0)[0]
            return preds, used_local, model_info

        models_ok = all([os.path.exists(LSTM_PATH), os.path.exists(SCALER_PATH)])
        if not models_ok:
            st.error(f"❌ Modèle LSTM global ou scaler introuvable\n\nLSTM attendu : `{LSTM_PATH}`\nScaler attendu : `{SCALER_PATH}`")
            st.stop()

        def _get_seuil_danger():
            if "seuil_danger_v7" not in st.session_state:
                st.session_state["seuil_danger_v7"] = _load_seuil_danger()
            return st.session_state["seuil_danger_v7"]

        def _danger_level(hsv, seuil_danger):
            if hsv >= seuil_danger:  return "#ef4444","DANGER","🔴"
            elif hsv >= SEUIL_WATCH: return "#f97316","Vigilance","🟠"
            elif hsv >= SEUIL_WARN:  return "#eab308","Attention","🟡"
            else:                    return "#22c55e","Favorable","🟢"

        def _model_badge(used_local, plage):
            if used_local:
                return f"<span style='background:#7c3aed;color:white;padding:3px 10px;border-radius:12px;font-size:0.8em;'>🎯 Modèle fine-tuné V7 — {plage}</span>"
            return "<span style='background:#2563eb;color:white;padding:3px 10px;border-radius:12px;font-size:0.8em;'>🌐 Modèle global V7 (fallback)</span>"

        def _fetch_openmeteo_forecast(lat, lon, date_start, date_end):
            start_str = date_start.strftime("%Y-%m-%d")
            end_str   = date_end.strftime("%Y-%m-%d")
            params = (
                f"latitude={lat:.4f}&longitude={lon:.4f}"
                f"&hourly=wave_height,wave_period,wave_direction,"
                f"wind_speed_10m,wind_direction_10m"
                f"&start_date={start_str}&end_date={end_str}&timezone=UTC"
            )
            url = f"https://marine-api.open-meteo.com/v1/marine?{params}"
            try:
                with _urllib_request.urlopen(url, timeout=30) as resp:
                    data = _json_mod.loads(resp.read().decode())
            except Exception as e:
                raise RuntimeError(f"Erreur Open-Meteo Marine API : {e}")
            if "hourly" not in data:
                raise RuntimeError(f"Réponse Open-Meteo invalide : {str(data)[:300]}")
            h = data["hourly"]
            n = len(h["time"])
            df = pd.DataFrame({
                "DATETIME"      : pd.to_datetime(h["time"]),
                "MESURE"        : h.get("wave_height",       [np.nan]*n),
                "mwp"           : h.get("wave_period",       [np.nan]*n),
                "mwd"           : h.get("wave_direction",    [np.nan]*n),
                "wind_speed_raw": h.get("wind_speed_10m",    [np.nan]*n),
                "wind_dir_raw"  : h.get("wind_direction_10m",[np.nan]*n),
            })
            ws = pd.to_numeric(df["wind_speed_raw"],errors="coerce").fillna(0.0).values.astype(float)
            wd = pd.to_numeric(df["wind_dir_raw"],  errors="coerce").fillna(0.0).values.astype(float)
            df["u10"] = -ws*np.sin(np.deg2rad(wd))
            df["v10"] = -ws*np.cos(np.deg2rad(wd))
            for _c in ["MESURE","mwp","mwd"]:
                df[_c] = pd.to_numeric(df[_c], errors="coerce")
            df = df.sort_values("DATETIME").reset_index(drop=True)
            df["DATETIME"] = pd.to_datetime(df["DATETIME"]).dt.tz_localize(None)
            df = df[(df["DATETIME"]>=date_start)&(df["DATETIME"]<=date_end)].copy()
            df["MESURE"] = df["MESURE"].clip(0,20).ffill().bfill().fillna(0.3)
            df["mwp"]    = df["mwp"].clip(0,25).ffill().bfill().fillna(4.0)
            df["mwd"]    = df["mwd"].clip(0,360).ffill().bfill().fillna(180.0)
            return df[["DATETIME","MESURE","mwp","mwd","u10","v10"]]

        def _build_context_window(era5_df, forecast_df, target_dt, window_h=48):
            frames = []
            if era5_df is not None and not era5_df.empty:
                frames.append(era5_df.copy())
            if forecast_df is not None and not forecast_df.empty:
                if not frames:
                    frames.append(forecast_df.copy())
                else:
                    last_era5_dt = pd.to_datetime(frames[-1]["DATETIME"]).max()
                    fc_tail = forecast_df[forecast_df["DATETIME"]>last_era5_dt].copy()
                    if not fc_tail.empty:
                        frames.append(fc_tail)
            if not frames:
                raise RuntimeError("Aucune donnée disponible.")
            merged = pd.concat(frames, ignore_index=True)
            merged["DATETIME"] = pd.to_datetime(merged["DATETIME"]).dt.tz_localize(None)
            merged = merged.sort_values("DATETIME").drop_duplicates("DATETIME")
            window_end   = target_dt
            window_start = window_end - timedelta(hours=window_h-1)
            result = merged[(merged["DATETIME"]>=window_start)&(merged["DATETIME"]<=window_end)].copy().reset_index(drop=True)
            if len(result)==0:
                result = merged.tail(window_h).copy().reset_index(drop=True)
            return result

        def _build_features(df, lat, lon):
            df = df.copy()
            for col in ["MESURE","mwp","mwd","u10","v10"]:
                if col not in df.columns:
                    df[col] = 0.0
            df = df.ffill().bfill().fillna(0.0)
            df["MESURE"]     = np.clip(df["MESURE"],0,20)
            df["mwp"]        = np.clip(df["mwp"],0,25)
            df["wind_speed"] = np.clip(np.sqrt(df["u10"]**2+df["v10"]**2),0,60)
            mwd = df["mwd"].fillna(0.0)
            df["mwd_sin"]   = np.sin(np.deg2rad(mwd))
            df["mwd_cos"]   = np.cos(np.deg2rad(mwd))
            h = df["DATETIME"].dt.hour
            m = df["DATETIME"].dt.month
            df["hour_sin"]  = np.sin(2*np.pi*h/24)
            df["hour_cos"]  = np.cos(2*np.pi*h/24)
            df["month_sin"] = np.sin(2*np.pi*m/12)
            df["month_cos"] = np.cos(2*np.pi*m/12)
            lat_range = LAT_MAX-LAT_MIN if LAT_MAX!=LAT_MIN else 1.0
            lon_range = LON_MAX-LON_MIN if LON_MAX!=LON_MIN else 1.0
            df["y_norm"] = np.clip((lat-LAT_MIN)/lat_range,0.0,1.0)
            df["x_norm"] = np.clip((lon-LON_MIN)/lon_range,0.0,1.0)
            for feat in FEATURES:
                if feat not in df.columns:
                    df[feat] = 0.0
            return df

        def _prepare_window(df, scaler):
            X = df[FEATURES].values.astype(np.float32)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            X = scaler.transform(X)
            X = np.clip(X,-10,10)
            if len(X)<WINDOW:
                pad = np.tile(X[0],(WINDOW-len(X),1))
                X   = np.vstack([pad,X])
            X = X[-WINDOW:]
            return X.reshape(1,WINDOW,N_FEAT)

        def _physical_calibration(preds, df_feat, pred_dt, seuil_danger):
            preds       = np.array(preds, dtype=float)
            n           = len(df_feat)
            month       = pred_dt.month
            wind_series = df_feat["wind_speed"].values
            wind_trend  = float(np.polyfit(np.arange(min(12,n)),wind_series[-min(12,n):],1)[0])
            mwd_sin_last     = float(df_feat["mwd_sin"].iloc[-1]) if "mwd_sin" in df_feat.columns else 0.0
            mwd_coast_factor = max(0.8,1.0+0.15*(-mwd_sin_last))
            calibrated = np.empty(len(HORIZONS), dtype=float)
            for idx,(h_horizon,w_size) in enumerate(zip(HORIZONS,[1,6,12])):
                raw_pred   = float(preds[idx]) if idx<len(preds) else float(preds[-1])
                sub        = df_feat.tail(w_size) if n>=w_size else df_feat
                mean_wind  = float(sub["wind_speed"].mean())
                max_hsv    = float(sub["MESURE"].max())
                recent_hsv = float(sub["MESURE"].iloc[-1])
                if month in [11,12,1,2,3]:
                    raw_pred  = max(raw_pred,0.8+0.03*h_horizon)
                    wind_mult = 1.0
                    if mean_wind>7:  wind_mult *= 1.08+0.005*h_horizon
                    if mean_wind>10: wind_mult *= 1.15+0.008*h_horizon
                    if mean_wind>14: wind_mult *= 1.25+0.010*h_horizon
                    if max_hsv  >2:  wind_mult *= 1.10+0.004*h_horizon
                    raw_pred *= wind_mult
                else:
                    wind_mult = 1.0
                    if mean_wind>8:  wind_mult *= 1.06+0.003*h_horizon
                    if mean_wind>12: wind_mult *= 1.12+0.005*h_horizon
                    raw_pred *= wind_mult
                raw_pred *= (1.0+(mwd_coast_factor-1.0)*(h_horizon/12.0))
                if wind_trend>0.3:
                    raw_pred *= (1.0+0.02*wind_trend*(h_horizon/6.0))
                elif wind_trend<-0.3:
                    raw_pred *= max(0.85,1.0-0.015*abs(wind_trend)*(h_horizon/6.0))
                p25      = float(np.percentile(sub["MESURE"],25))
                t        = h_horizon/12.0
                floor    = recent_hsv*0.85*(1-t)+max(0.30,p25)*t
                raw_pred = max(raw_pred,floor)
                calibrated[idx] = np.clip(raw_pred,0.05,9.0)
            return calibrated

        col_cfg, col_res = st.columns([1,1.6])

        with col_cfg:
            section("⚙️","Configuration")
            if data_ok:
                plages_all_df = q(f"SELECT DISTINCT NOM_PLAGE, FIRST(X) AS lon, FIRST(Y) AS lat FROM {VIEW} GROUP BY NOM_PLAGE ORDER BY NOM_PLAGE")
                plage_names   = plages_all_df["NOM_PLAGE"].tolist()
            else:
                plages_all_df = pd.DataFrame()
                plage_names   = []
            if not plage_names:
                st.error("❌ Aucune plage disponible dans le dataset M1.")
                st.stop()
            selected_plage = st.selectbox("🏖️ Plage", plage_names)
            if not plages_all_df.empty and selected_plage:
                row         = plages_all_df[plages_all_df["NOM_PLAGE"]==selected_plage].iloc[0]
                lat_default = float(row["lat"])
                lon_default = float(row["lon"])
            else:
                lat_default = 36.75
                lon_default = 3.06
            lat = st.number_input("Latitude",  value=lat_default, format="%.4f")
            lon = st.number_input("Longitude", value=lon_default, format="%.4f")
            st.markdown("---")
            today       = datetime.utcnow().date()
            era5_cutoff = (datetime.utcnow() - timedelta(days=60)).date()
            from datetime import date as _date
            pred_date = st.date_input("📅 Date cible de prédiction",
                value=today+timedelta(days=1),
                min_value=_date(1985,1,1),
                max_value=today+timedelta(days=1),
                help="• Historique → ERA5\n• Futur → Open-Meteo Marine")
            pred_dt = datetime.combine(pred_date, datetime.min.time())
            if pred_date <= era5_cutoff:
                st.info("📡 **Mode ERA5 pur** · Date historique"); use_forecast=False
            elif pred_date <= today:
                st.warning("🔀 **Mode hybride** · ERA5 + Open-Meteo"); use_forecast=True
            else:
                st.success(f"🚀 **Mode forecast** · Open-Meteo Marine"); use_forecast=True
            st.markdown("---")
            cds_key = st.text_input("🔑 CDS API KEY (ERA5)", type="password")
            safe_name  = selected_plage.replace(" ","_") if selected_plage else ""
            local_path = os.path.join(os.path.dirname(LSTM_PATH),"plots",f"{safe_name}_lstm.keras")
            if os.path.exists(local_path):
                st.success(f"✅ Modèle fine-tuné V7 disponible pour **{selected_plage}**")
            else:
                st.info("ℹ️ Modèle global V7 utilisé pour cette plage")
            run_btn = st.button("🚀 Lancer prédiction", type="primary", use_container_width=True)

        with col_res:
            section("📊","Résultats")
            results_area = st.empty()

        if run_btn:
            if not selected_plage:
                st.error("❌ Veuillez sélectionner une plage.")
                st.stop()
            try:
                seuil_danger = _get_seuil_danger()
                progress     = st.progress(0)
                status       = st.empty()
                status.info("⚙️ Chargement du scaler global V7...")
                progress.progress(8)
                scaler = _load_scaler()
                df_era5 = None; df_forecast = None; sources_used = []
                try:
                    status.info("🌊 Récupération Open-Meteo Marine...")
                    progress.progress(35)
                    om_start    = pred_dt - timedelta(hours=WINDOW)
                    om_end      = pred_dt + timedelta(hours=24)
                    df_forecast = _fetch_openmeteo_forecast(lat, lon, om_start, om_end)
                    if not df_forecast.empty:
                        sources_used.append(f"✅ Open-Meteo Marine ({len(df_forecast)} pts)")
                except Exception as e_om:
                    st.warning(f"⚠️ Open-Meteo Marine : {e_om}")
                    df_forecast = None
                if df_era5 is None and df_forecast is None:
                    st.error("❌ Aucune source de données disponible.")
                    st.stop()
                status.info("🔀 Fusion des sources (48h)...")
                progress.progress(50)
                df_raw = _build_context_window(df_era5, df_forecast, pred_dt, WINDOW)
                if len(df_raw)<10:
                    st.error(f"❌ Données insuffisantes : {len(df_raw)} pas de temps.")
                    st.stop()
                for src in sources_used:
                    st.caption(src)
                status.info("🔧 Construction features V7...")
                progress.progress(62)
                df_feat = _build_features(df_raw, lat, lon)
                status.info("📐 Préparation tenseur (1, 48, 11)...")
                progress.progress(72)
                X_tensor = _prepare_window(df_feat, scaler)
                status.info("🧠 Inférence LSTM V7...")
                progress.progress(84)
                preds_raw, used_local, model_info = _run_inference(X_tensor, selected_plage)
                status.info("⚖️ Calibration physique...")
                progress.progress(94)
                preds = _physical_calibration(preds_raw, df_feat, pred_dt, seuil_danger)
                preds = np.nan_to_num(preds, nan=1.5, posinf=5.0, neginf=0.5)
                preds = np.clip(preds, 0.05, 9.0)
                progress.progress(100)
                status.empty()
                with results_area.container():
                    st.markdown(_model_badge(used_local,selected_plage), unsafe_allow_html=True)
                    st.markdown(f"<div style='font-size:0.75rem;color:#64748b;margin-top:4px;'>Seuil danger : <strong style='color:#ef4444'>{seuil_danger:.2f} m</strong></div>",unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.success(f"✅ Prévision HSV — {selected_plage}")
                    cols = st.columns(len(HORIZONS))
                    for i,(h,col) in enumerate(zip(HORIZONS,cols)):
                        hsv = float(preds[i])
                        color,label,emoji = _danger_level(hsv,seuil_danger)
                        delta_str = "Référence" if i==0 else f"{hsv-float(preds[0]):+.2f} m vs t+1h"
                        with col:
                            st.metric(f"t+{h}h",f"{hsv:.2f} m",delta=delta_str)
                            st.markdown(f"<div style='text-align:center;color:{color};font-weight:700'>{emoji} {label}</div>",unsafe_allow_html=True)
                    marker_colors = [_danger_level(float(p),seuil_danger)[0] for p in preds]
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=[f"t+{h}h" for h in HORIZONS],y=preds,mode="lines+markers",
                        line=dict(width=4,color="#8b5cf6"),
                        marker=dict(size=14,color=marker_colors,line=dict(width=2,color="white")),name="HSV (m)"))
                    fig.add_hrect(y0=seuil_danger,y1=9.0,fillcolor="rgba(239,68,68,0.08)",line_width=0)
                    fig.add_hrect(y0=SEUIL_WATCH,y1=seuil_danger,fillcolor="rgba(249,115,22,0.07)",line_width=0)
                    fig.add_hrect(y0=SEUIL_WARN,y1=SEUIL_WATCH,fillcolor="rgba(234,179,8,0.05)",line_width=0)
                    fig.add_hline(y=seuil_danger,line_dash="dash",line_color="red",line_width=2,annotation_text=f"Seuil danger {seuil_danger:.2f}m")
                    fig.add_hline(y=SEUIL_WATCH,line_dash="dot",line_color="orange",line_width=1,annotation_text=f"Vigilance {SEUIL_WATCH}m")
                    fig.add_hline(y=SEUIL_WARN,line_dash="dot",line_color="#eab308",line_width=1,annotation_text=f"Attention {SEUIL_WARN}m")
                    for i,(h,p) in enumerate(zip(HORIZONS,preds)):
                        fig.add_annotation(x=f"t+{h}h",y=float(p),text=f"<b>{float(p):.2f}m</b>",showarrow=False,yshift=16,font=dict(size=11,color=marker_colors[i]))
                    fig.update_layout(title=f"Prévision HSV — {selected_plage} | {pred_date.strftime('%d/%m/%Y')}",
                        yaxis_title="Hauteur Significative des Vagues (m)",
                        yaxis=dict(range=[0,max(max(preds)*1.25,2.0)]),height=380,showlegend=False)
                    apply_theme(fig)
                    st.plotly_chart(fig, use_container_width=True)
                    with st.expander("🔍 Données de contexte (48h)"):
                        display_cols = [c for c in ["DATETIME","MESURE","wind_speed","mwp","mwd"] if c in df_feat.columns]
                        st.dataframe(df_feat[display_cols].tail(48), use_container_width=True)
                    with st.expander("⚙️ Détails techniques V7"):
                        c1_t, c2_t = st.columns(2)
                        with c1_t:
                            st.markdown(f"**Modèle** : `{model_info}`\n**Fenêtre** : {WINDOW}h\n**Features** : {N_FEAT}")
                        with c2_t:
                            st.markdown(f"**Vent moy** : {df_feat['wind_speed'].mean():.1f} m/s\n**HSV max contexte** : {df_feat['MESURE'].max():.2f} m")
                        comp_df = pd.DataFrame({
                            "Horizon"    : [f"t+{h}h" for h in HORIZONS],
                            "Brut (m)"   : [f"{float(preds_raw[i]):.3f}" for i in range(len(HORIZONS))],
                            "Calibré (m)": [f"{float(preds[i]):.3f}"     for i in range(len(HORIZONS))],
                            "Niveau"     : [_danger_level(float(preds[i]),seuil_danger)[2]+" "+_danger_level(float(preds[i]),seuil_danger)[1] for i in range(len(HORIZONS))],
                        })
                        st.dataframe(comp_df, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"❌ Erreur pipeline V7 : {e}")
                with st.expander("🐛 Traceback complet"):
                    st.code(traceback.format_exc())

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
lang_info = {"fr":"Français","en":"English","ar":"العربية"}.get(st.session_state.get("lang","fr"),"Français")

if is_m2:
    footer_content = (
        f"Système HSV · Côtes Algériennes · ERA5 + CMEMS (M2) · 1999–2023 &nbsp;·&nbsp; "
        f"LSTM + Transfer Learning · <strong>2 horizons : +1j · +7j</strong> &nbsp;·&nbsp; "
        f"DuckDB + Streamlit + Plotly &nbsp;·&nbsp; "
        f"Copernicus CDS ERA5 · CMEMS · Open-Meteo Marine &nbsp;·&nbsp; "
        f"🌐 {lang_info}"
    )
else:
    footer_content = (
        f"Système HSV · Côtes Algériennes · ERA5 (M1) · 1985–2023 &nbsp;·&nbsp; "
        f"LSTM + Transfer Learning · <strong>3 horizons : +1h · +6h · +12h</strong> &nbsp;·&nbsp; "
        f"DuckDB + Streamlit + Plotly &nbsp;·&nbsp; "
        f"Copernicus CDS ERA5 · Open-Meteo Marine &nbsp;·&nbsp; "
        f"🌐 {lang_info}"
    )

st.markdown(f"""
<div style="text-align:center;padding:1rem 0;font-size:.75rem;color:var(--text-m);">
    {footer_content}
</div>""", unsafe_allow_html=True)
