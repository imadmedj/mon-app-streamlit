"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  DASHBOARD HSV — CÔTES ALGÉRIENNES                                          ║
║  Modèle 1 : ERA5 seul          · 1985–2023 · MESURE, wind_speed, mwp, mwd  ║
║  Modèle 2 : ERA5 + CMEMS       · 1999–2023 · + salinity, o2, spm, sst      ║
║  NOUVEAU   : Prédiction Temps Réel via API Copernicus (CDS + CMEMS)         ║
║  Dataset  : data/lstm_final_clean   (~20 M lignes)                          ║
║  Dataset2 : data/dataset_model2_1999_2023_clean  (~12 M lignes)             ║
║  Optimisé : DuckDB (SQL sur Parquet)                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import duckdb
import os
import tempfile
import json
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# TRADUCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
TRANSLATIONS = {
    "fr": {
        "app_title": "HSV Algérie",
        "app_subtitle": "Côtes · ERA5 + CMEMS",
        "model_data": "MODÈLE DE DONNÉES",
        "model1_label": "🔵 M1 — ERA5 seul (1985–2023)",
        "model2_label": "🟣 M2 — ERA5 + CMEMS (1999–2023)",
        "navigation": "Navigation",
        "temporal_filters": "FILTRES TEMPORELS",
        "geo_filters": "FILTRES GÉOGRAPHIQUES",
        "year": "Année","month": "Mois","hour": "Heure",
        "wilaya": "Wilaya","beach": "Plage","all": "Tous...","all_f": "Toutes...",
        "home": "🏠 Accueil","global_analysis": "📊 Analyse Globale",
        "summer_analysis": "🏖️ Analyse Été","activities": "🌊 Activités",
        "analysis": "📊 Analyse","drowning_alerts": "🏊 Alertes Noyades",
        "desalination": "💧 Dessalement SWRO","aquaculture": "🐟 Aquaculture",
        "synthesis": "📋 Synthèse & Export","danger_map": "🗺️ Carte des Dangers",
        "realtime_pred": "🔮 Prédiction Temps Réel",
        "hero_title": "Système d'Analyse des Vagues Côtières — Algérie",
        "hero_sub": "Prévision HSV par LSTM + Transfer Learning · Deux modèles complémentaires :",
        "hero_sub2": "M1 ERA5 seul 1985–2023 (20M mesures) · M2 ERA5 + CMEMS 1999–2023 avec Salinité, O₂ dissous et Matières en suspension.",
        "drowning_alerts_pill": "Alertes noyades","desalination_pill": "Dessalement SWRO",
        "aquaculture_pill": "Aquaculture marine","marine_quality_pill": "Qualité marine O₂/Salinité",
        "two_models_pill": "Deux modèles LSTM","global_stats": "Statistiques globales — Modèle actif",
        "annual_evolution": "Évolution annuelle de la HSV","critical_thresholds": "Seuils critiques — tableau de synthèse",
        "measures": "Mesures","avg_hsv": "HSV Moyenne","max_hsv": "HSV Maximum","p95": "Percentile 95","std": "Écart-type",
        "months": {1:"Janvier",2:"Février",3:"Mars",4:"Avril",5:"Mai",6:"Juin",
                   7:"Juillet",8:"Août",9:"Septembre",10:"Octobre",11:"Novembre",12:"Décembre"},
        "months_short": {1:"Jan",2:"Fév",3:"Mar",4:"Avr",5:"Mai",6:"Jun",
                         7:"Jul",8:"Aoû",9:"Sep",10:"Oct",11:"Nov",12:"Déc"},
        "no_data": "Aucune donnée disponible.","no_filtered_data": "Aucune donnée pour les filtres sélectionnés.",
        "computing": "Calcul KPIs...","loading_map": "Chargement carte...",
        "m2_required": "⚠️ Cette page nécessite le **Modèle 2** (ERA5 + CMEMS). Veuillez sélectionner **M2** dans la sidebar.",
        "m1_drowning_only": "🏊 En Modèle 1, seule la page **Alertes Noyades** est disponible.\nPour Dessalement et Aquaculture, merci de sélectionner **🟣 M2 — ERA5 + CMEMS**.",
        "variables": "Variables","lang_button": "🌐 Langue","select_lang": "Sélectionner la langue",
        "time_series": "📈 Série Temporelle","distribution": "🗂️ Distribution","seasonality": "📅 Saisonnalité",
        "by_beach": "🏖️ Par Plage","alerts": "📊 Alertes","wind_mwd": "🌬️ Vent & MWD",
        "favorable_windows": "📊 Fenêtres Favorables","best_sites": "🏖️ Meilleurs Sites",
        "sst_tab": "🌡️ SST","pressure_tab": "📊 Pression MSL","op_windows": "⚙️ Fenêtres Opérationnelles",
        "evolution": "📅 Évolution","by_wilaya": "Classement par wilaya","synth_by_beach": "📊 Synthèse par Plage",
        "monthly_synth": "📅 Synthèse Mensuelle","export": "💾 Export","download_csv": "⬇️ Télécharger CSV",
        "indicator_mapped": "Indicateur cartographié","avg_hsv_map": "HSV Moyenne","max_hsv_map": "HSV Maximum",
        "alert_m1_map": "Alertes Noyades M1","alert_m2_map": "Alertes Noyades M2",
        "dessal_map": "Dessalement","aqua_map": "Aquaculture",
        "seasons": {"Hiver":"Hiver","Printemps":"Printemps","Été":"Été","Automne":"Automne"},
        # Nouvelles clés pour la prédiction
        "pred_m1_label": "🔵 Prédiction M1 — ERA5",
        "pred_m2_label": "🟣 Prédiction M2 — ERA5 + CMEMS",
        "pred_submenu": "Type de prédiction",
    },
    "en": {
        "app_title": "HSV Algeria","app_subtitle": "Coastline · ERA5 + CMEMS",
        "model_data": "DATA MODEL","model1_label": "🔵 M1 — ERA5 only (1985–2023)",
        "model2_label": "🟣 M2 — ERA5 + CMEMS (1999–2023)","navigation": "Navigation",
        "temporal_filters": "TEMPORAL FILTERS","geo_filters": "GEOGRAPHIC FILTERS",
        "year": "Year","month": "Month","hour": "Hour","wilaya": "Wilaya","beach": "Beach",
        "all": "All...","all_f": "All...","home": "🏠 Home","global_analysis": "📊 Global Analysis",
        "summer_analysis": "🏖️ Summer Analysis","activities": "🌊 Activities","analysis": "📊 Analysis",
        "drowning_alerts": "🏊 Drowning Alerts","desalination": "💧 SWRO Desalination",
        "aquaculture": "🐟 Aquaculture","synthesis": "📋 Summary & Export","danger_map": "🗺️ Danger Map",
        "realtime_pred": "🔮 Real-Time Prediction",
        "hero_title": "Coastal Wave Analysis System — Algeria",
        "hero_sub": "HSV Prediction via LSTM + Transfer Learning · Two complementary models:",
        "hero_sub2": "M1 ERA5 only 1985–2023 (20M records) · M2 ERA5 + CMEMS 1999–2023 with Salinity, Dissolved O₂ and Suspended Matter.",
        "drowning_alerts_pill": "Drowning alerts","desalination_pill": "SWRO Desalination",
        "aquaculture_pill": "Marine aquaculture","marine_quality_pill": "Marine quality O₂/Salinity",
        "two_models_pill": "Two LSTM models","global_stats": "Global statistics — Active model",
        "annual_evolution": "Annual HSV evolution","critical_thresholds": "Critical thresholds — summary table",
        "measures": "Records","avg_hsv": "Avg HSV","max_hsv": "Max HSV","p95": "Percentile 95","std": "Std Dev",
        "months": {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",
                   7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"},
        "months_short": {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                         7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"},
        "no_data": "No data available.","no_filtered_data": "No data for the selected filters.",
        "computing": "Computing KPIs...","loading_map": "Loading map...",
        "m2_required": "⚠️ This page requires **Model 2** (ERA5 + CMEMS). Please select **M2** in the sidebar.",
        "m1_drowning_only": "🏊 In Model 1, only the **Drowning Alerts** page is available.\nFor Desalination and Aquaculture, please select **🟣 M2 — ERA5 + CMEMS**.",
        "variables": "Variables","lang_button": "🌐 Language","select_lang": "Select language",
        "time_series": "📈 Time Series","distribution": "🗂️ Distribution","seasonality": "📅 Seasonality",
        "by_beach": "🏖️ By Beach","alerts": "📊 Alerts","wind_mwd": "🌬️ Wind & MWD",
        "favorable_windows": "📊 Favorable Windows","best_sites": "🏖️ Best Sites",
        "sst_tab": "🌡️ SST","pressure_tab": "📊 Pressure MSL","op_windows": "⚙️ Operational Windows",
        "evolution": "📅 Evolution","by_wilaya": "Ranking by wilaya","synth_by_beach": "📊 Summary by Beach",
        "monthly_synth": "📅 Monthly Summary","export": "💾 Export","download_csv": "⬇️ Download CSV",
        "indicator_mapped": "Mapped indicator","avg_hsv_map": "Average HSV","max_hsv_map": "Maximum HSV",
        "alert_m1_map": "Drowning Alerts M1","alert_m2_map": "Drowning Alerts M2",
        "dessal_map": "Desalination","aqua_map": "Aquaculture",
        "seasons": {"Hiver":"Winter","Printemps":"Spring","Été":"Summer","Automne":"Autumn"},
        "pred_m1_label": "🔵 Prediction M1 — ERA5",
        "pred_m2_label": "🟣 Prediction M2 — ERA5 + CMEMS",
        "pred_submenu": "Prediction type",
    },
    "ar": {
        "app_title": "نظام HSV الجزائر","app_subtitle": "السواحل · ERA5 + CMEMS",
        "model_data": "نموذج البيانات","model1_label": "🔵 N1 — ERA5 فقط (1985–2023)",
        "model2_label": "🟣 N2 — ERA5 + CMEMS (1999–2023)","navigation": "التنقل",
        "temporal_filters": "مرشحات زمنية","geo_filters": "مرشحات جغرافية",
        "year": "السنة","month": "الشهر","hour": "الساعة","wilaya": "الولاية","beach": "الشاطئ",
        "all": "الكل...","all_f": "الكل...","home": "🏠 الرئيسية","global_analysis": "📊 التحليل العام",
        "summer_analysis": "🏖️ تحليل الصيف","activities": "🌊 الأنشطة","analysis": "📊 التحليل",
        "drowning_alerts": "🏊 تنبيهات الغرق","desalination": "💧 تحلية المياه SWRO",
        "aquaculture": "🐟 تربية الأحياء البحرية","synthesis": "📋 الملخص والتصدير","danger_map": "🗺️ خريطة المخاطر",
        "realtime_pred": "🔮 التنبؤ الفوري",
        "hero_title": "نظام تحليل الأمواج الساحلية — الجزائر",
        "hero_sub": "توقع HSV بواسطة LSTM + Transfer Learning · نموذجان تكاملييان:",
        "hero_sub2": "N1 ERA5 فقط 1985–2023 · N2 ERA5 + CMEMS 1999–2023 مع الملوحة، O₂ الذائب والمواد العالقة.",
        "drowning_alerts_pill": "تنبيهات الغرق","desalination_pill": "تحلية المياه SWRO",
        "aquaculture_pill": "تربية الأحياء البحرية","marine_quality_pill": "جودة البيئة البحرية O₂/ملوحة",
        "two_models_pill": "نموذجان LSTM","global_stats": "إحصائيات عامة — النموذج النشط",
        "annual_evolution": "التطور السنوي لـ HSV","critical_thresholds": "الحدود الحرجة — جدول ملخص",
        "measures": "القياسات","avg_hsv": "متوسط HSV","max_hsv": "أقصى HSV","p95": "المئين 95","std": "الانحراف المعياري",
        "months": {1:"جانفي",2:"فيفري",3:"مارس",4:"أبريل",5:"ماي",6:"جوان",
                   7:"جويلية",8:"أوت",9:"سبتمبر",10:"أكتوبر",11:"نوفمبر",12:"ديسمبر"},
        "months_short": {1:"جان",2:"فيف",3:"مار",4:"أبر",5:"ماي",6:"جوا",
                         7:"جوي",8:"أوت",9:"سبت",10:"أكت",11:"نوف",12:"ديس"},
        "no_data": "لا توجد بيانات متاحة.","no_filtered_data": "لا توجد بيانات للمرشحات المحددة.",
        "computing": "جارٍ الحساب...","loading_map": "جارٍ تحميل الخريطة...",
        "m2_required": "⚠️ هذه الصفحة تتطلب **النموذج 2** (ERA5 + CMEMS). يرجى اختيار **N2** في الشريط الجانبي.",
        "m1_drowning_only": "🏊 في النموذج 1، صفحة **تنبيهات الغرق** فقط متاحة.",
        "variables": "المتغيرات","lang_button": "🌐 اللغة","select_lang": "اختر اللغة",
        "time_series": "📈 السلسلة الزمنية","distribution": "🗂️ التوزيع","seasonality": "📅 الموسمية",
        "by_beach": "🏖️ حسب الشاطئ","alerts": "📊 التنبيهات","wind_mwd": "🌬️ الرياح والاتجاه",
        "favorable_windows": "📊 النوافذ الملائمة","best_sites": "🏖️ أفضل المواقع",
        "sst_tab": "🌡️ درجة حرارة السطح","pressure_tab": "📊 ضغط MSL","op_windows": "⚙️ النوافذ التشغيلية",
        "evolution": "📅 التطور","by_wilaya": "التصنيف حسب الولاية","synth_by_beach": "📊 ملخص حسب الشاطئ",
        "monthly_synth": "📅 الملخص الشهري","export": "💾 تصدير","download_csv": "⬇️ تنزيل CSV",
        "indicator_mapped": "المؤشر المرسوم","avg_hsv_map": "متوسط HSV","max_hsv_map": "أقصى HSV",
        "alert_m1_map": "تنبيهات الغرق N1","alert_m2_map": "تنبيهات الغرق N2",
        "dessal_map": "التحلية","aqua_map": "الأحياء البحرية",
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
[data-testid="stSidebar"]{background:var(--bg-surface)!important;border-right:1px solid var(--border-s)!important;}
[data-testid="stSidebar"] *{color:var(--text-p)!important;}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{font-family:var(--fd)!important;color:var(--text-h)!important;}
[data-testid="stSidebar"] label{font-size:.68rem!important;text-transform:uppercase;letter-spacing:.12em;color:var(--text-m)!important;font-weight:600!important;}
[data-testid="stMetric"]{background:var(--bg-card)!important;border:1px solid var(--border-s)!important;border-radius:var(--r-lg)!important;padding:1.1rem 1.3rem!important;position:relative;overflow:hidden;transition:border-color .2s,background .2s;}
[data-testid="stMetric"]:hover{border-color:var(--border-m)!important;background:var(--bg-card-h)!important;}
[data-testid="stMetric"]::after{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--a1),var(--a2));}
[data-testid="stMetricLabel"]{color:var(--text-m)!important;font-size:.7rem!important;text-transform:uppercase!important;letter-spacing:.12em!important;font-weight:600!important;}
[data-testid="stMetricValue"]{font-family:var(--fd)!important;color:var(--text-h)!important;font-size:1.7rem!important;font-weight:700!important;}
.stTabs [data-baseweb="tab-list"]{background:var(--bg-card)!important;border:1px solid var(--border-s)!important;border-radius:var(--r-md)!important;padding:4px!important;gap:3px;}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:var(--text-m)!important;border-radius:var(--r-sm)!important;font-size:.8rem!important;font-weight:600!important;letter-spacing:.04em;padding:7px 18px!important;border:none!important;}
.stTabs [aria-selected="true"]{background:var(--a1)!important;color:white!important;}
.stDownloadButton>button,.stButton>button{background:var(--a1)!important;color:white!important;border:none!important;border-radius:var(--r-md)!important;font-weight:600!important;}
hr{border:none;border-top:1px solid var(--border-s)!important;margin:1.5rem 0!important;}
.hero{background:var(--bg-surface);border:1px solid var(--border-s);border-radius:var(--r-xl);padding:2.2rem 2rem 1.8rem;margin-bottom:1.5rem;position:relative;overflow:hidden;}
.hero::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--a1),var(--a2),var(--a3));}
.hero h1{font-family:var(--fd);font-size:1.9rem;font-weight:800;color:var(--text-h);margin:0 0 .4rem;line-height:1.2;}
.hero .sub{font-size:.85rem;color:var(--text-m);line-height:1.6;max-width:680px;}
.hero .pills{margin-top:1rem;display:flex;flex-wrap:wrap;gap:8px;}
.section-title{font-family:var(--fd);font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.16em;color:var(--a1);margin:2rem 0 .8rem;display:flex;align-items:center;gap:8px;}
.section-title::after{content:'';flex:1;height:1px;background:var(--border-s);}
.info-card{background:var(--bg-card);border:1px solid var(--border-s);border-left:3px solid;border-radius:var(--r-md);padding:.9rem 1.1rem;margin-bottom:.7rem;}
.info-card .title{font-family:var(--fd);font-size:.85rem;font-weight:700;color:var(--text-h);margin-bottom:.5rem;}
.threshold-row{display:flex;align-items:center;justify-content:space-between;padding:6px 0;border-bottom:1px solid var(--border-s);font-size:.8rem;}
.threshold-row:last-child{border-bottom:none;}
.threshold-key{color:var(--text-m);}
.threshold-val{font-weight:600;font-family:var(--fm);font-size:.78rem;}
.pill{display:inline-flex;align-items:center;padding:3px 10px;border-radius:20px;font-size:.7rem;font-weight:600;letter-spacing:.04em;border:1px solid;}
.pill-blue{background:rgba(14,165,233,.12);color:#38bdf8;border-color:rgba(14,165,233,.3);}
.pill-green{background:rgba(16,185,129,.12);color:#34d399;border-color:rgba(16,185,129,.3);}
.pill-red{background:rgba(239,68,68,.12);color:#f87171;border-color:rgba(239,68,68,.3);}
.pill-amber{background:rgba(245,158,11,.12);color:#fbbf24;border-color:rgba(245,158,11,.3);}
.pill-purple{background:rgba(139,92,246,.12);color:#a78bfa;border-color:rgba(139,92,246,.3);}
.pill-cyan{background:rgba(6,182,212,.12);color:#22d3ee;border-color:rgba(6,182,212,.3);}
.page-header{display:flex;align-items:center;gap:12px;margin-bottom:1.5rem;padding-bottom:1rem;border-bottom:1px solid var(--border-s);}
.page-header-icon{width:40px;height:40px;border-radius:var(--r-md);display:flex;align-items:center;justify-content:center;font-size:1.2rem;flex-shrink:0;}
.page-header h1{font-family:var(--fd);font-size:1.4rem;font-weight:700;color:var(--text-h);margin:0;line-height:1.2;}
.page-header p{font-size:.8rem;color:var(--text-m);margin:2px 0 0;}
.sidebar-logo{text-align:center;padding:1.2rem 0 1.5rem;border-bottom:1px solid var(--border-s);margin-bottom:1rem;}
.sidebar-logo .logo-icon{font-size:2rem;}
.sidebar-logo .logo-title{font-family:var(--fd);font-size:1.1rem;font-weight:800;color:var(--text-h);margin:6px 0 2px;}
.sidebar-logo .logo-sub{font-size:.68rem;text-transform:uppercase;letter-spacing:.12em;color:var(--text-m);}

/* ── Pred sub-menu card ──────────────────────────────── */
.pred-choice-card{
    background:var(--bg-card);
    border:1px solid var(--border-s);
    border-radius:var(--r-lg);
    padding:1.4rem 1.6rem;
    margin-bottom:.8rem;
    position:relative;
    overflow:hidden;
    transition:border-color .2s, background .2s;
}
.pred-choice-card:hover{border-color:var(--border-m);background:var(--bg-card-h);}
.pred-choice-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;}
.pred-m1-card::before{background:linear-gradient(90deg,var(--a1),var(--a2));}
.pred-m2-card::before{background:linear-gradient(90deg,var(--purple),#a855f7);}
.pred-choice-card .card-title{font-family:var(--fd);font-size:1rem;font-weight:700;color:var(--text-h);margin-bottom:.4rem;}
.pred-choice-card .card-desc{font-size:.78rem;color:var(--text-m);line-height:1.5;}
.pred-choice-card .card-badges{margin-top:.7rem;display:flex;flex-wrap:wrap;gap:6px;}

/* ── Incompatibilité banner ──────────────────────────── */
.incompat-banner{
    background:rgba(239,68,68,.08);
    border:2px solid rgba(239,68,68,.4);
    border-radius:var(--r-xl);
    padding:2rem 2.2rem;
    margin-bottom:1.5rem;
    position:relative;
    overflow:hidden;
}
.incompat-banner::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#ef4444,#f87171);}
.incompat-banner .ib-icon{font-size:2.5rem;margin-bottom:.6rem;}
.incompat-banner .ib-title{font-family:var(--fd);font-size:1.3rem;font-weight:800;color:#f87171;margin-bottom:.5rem;}
.incompat-banner .ib-body{font-size:.88rem;color:var(--text-p);line-height:1.7;}
.incompat-banner .ib-step{background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.25);border-radius:var(--r-md);padding:.7rem 1rem;margin-top:1rem;font-size:.82rem;color:#fca5a5;}

/* ── WIP banner ──────────────────────────────────────── */
.wip-banner{
    background:rgba(139,92,246,.08);
    border:2px solid rgba(139,92,246,.35);
    border-radius:var(--r-xl);
    padding:2rem 2.2rem;
    margin-bottom:1.5rem;
    position:relative;
    overflow:hidden;
}
.wip-banner::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,var(--purple),#a855f7);}
.wip-banner .wb-icon{font-size:2.5rem;margin-bottom:.6rem;}
.wip-banner .wb-title{font-family:var(--fd);font-size:1.3rem;font-weight:800;color:#a78bfa;margin-bottom:.5rem;}
.wip-banner .wb-body{font-size:.88rem;color:var(--text-p);line-height:1.7;}

/* ── Feature comparison table ────────────────────────── */
.feat-table{width:100%;border-collapse:collapse;font-size:.82rem;margin-top:1rem;}
.feat-table th{background:var(--bg-card);color:var(--text-m);font-size:.68rem;text-transform:uppercase;letter-spacing:.1em;padding:8px 14px;text-align:left;border-bottom:1px solid var(--border-m);}
.feat-table td{padding:8px 14px;border-bottom:1px solid var(--border-s);color:var(--text-p);}
.feat-table tr:last-child td{border-bottom:none;}
.feat-table .check-yes{color:#34d399;font-weight:700;}
.feat-table .check-no{color:#4a7a96;}
.feat-table .feat-name{color:var(--text-h);font-family:var(--fm);font-size:.78rem;}

/* ── Pipeline steps ─────────────────────────────────── */
.pipeline-wrap{display:flex;flex-direction:column;gap:0;margin:1.2rem 0;}
.pipeline-step{display:flex;align-items:flex-start;gap:14px;padding:14px 16px;background:var(--bg-card);border:1px solid var(--border-s);border-radius:0;position:relative;}
.pipeline-step:first-child{border-radius:var(--r-lg) var(--r-lg) 0 0;}
.pipeline-step:last-child{border-radius:0 0 var(--r-lg) var(--r-lg);}
.pipeline-step.active{border-color:var(--a1);background:rgba(14,165,233,.07);}
.pipeline-step.done{border-color:var(--a3);background:rgba(16,185,129,.05);}
.pipeline-step.error{border-color:var(--danger);background:rgba(239,68,68,.05);}
.ps-num{width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:.75rem;font-weight:700;flex-shrink:0;background:var(--bg-surface);border:1px solid var(--border-m);color:var(--text-m);}
.ps-num.active{background:var(--a1);color:white;border-color:var(--a1);}
.ps-num.done{background:var(--a3);color:white;border-color:var(--a3);}
.ps-num.error{background:var(--danger);color:white;border-color:var(--danger);}
.ps-content .ps-title{font-size:.85rem;font-weight:600;color:var(--text-h);}
.ps-content .ps-sub{font-size:.75rem;color:var(--text-m);margin-top:2px;}

/* ── Alerte card prédiction ─────────────────────────── */
.alert-card{border-radius:var(--r-xl);padding:1.8rem 2rem;margin-bottom:1rem;text-align:center;position:relative;overflow:hidden;}
.alert-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;}
.alert-safe{background:rgba(16,185,129,.1);border:2px solid rgba(16,185,129,.4);}
.alert-safe::before{background:linear-gradient(90deg,#10b981,#34d399);}
.alert-watch{background:rgba(245,158,11,.1);border:2px solid rgba(245,158,11,.4);}
.alert-watch::before{background:linear-gradient(90deg,#f59e0b,#fbbf24);}
.alert-warn{background:rgba(249,115,22,.1);border:2px solid rgba(249,115,22,.4);}
.alert-warn::before{background:linear-gradient(90deg,#f97316,#fb923c);}
.alert-danger{background:rgba(239,68,68,.12);border:2px solid rgba(239,68,68,.5);}
.alert-danger::before{background:linear-gradient(90deg,#ef4444,#f87171);}
.alert-icon{font-size:2.5rem;margin-bottom:.5rem;}
.alert-title{font-family:var(--fd);font-size:1.6rem;font-weight:800;margin-bottom:.3rem;}
.alert-hsv{font-family:var(--fm);font-size:1.1rem;}
.alert-sub{font-size:.8rem;color:var(--text-m);margin-top:.5rem;}

/* ── Horizon card ────────────────────────────────────── */
.horizon-card{background:var(--bg-card);border:1px solid var(--border-s);border-radius:var(--r-lg);padding:1rem;text-align:center;transition:border-color .2s;}
.horizon-card:hover{border-color:var(--border-m);}
.horizon-card .h-label{font-size:.7rem;text-transform:uppercase;letter-spacing:.1em;color:var(--text-m);font-weight:600;}
.horizon-card .h-value{font-family:var(--fd);font-size:1.4rem;font-weight:700;margin:.3rem 0;}
.horizon-card .h-alert{font-size:.75rem;font-weight:600;padding:2px 8px;border-radius:20px;display:inline-block;}

/* ── Log console ─────────────────────────────────────── */
.log-console{background:#020c16;border:1px solid var(--border-s);border-radius:var(--r-md);padding:1rem 1.2rem;font-family:var(--fm);font-size:.75rem;color:#4ade80;max-height:220px;overflow-y:auto;line-height:1.7;}
.log-console .log-err{color:#f87171;}
.log-console .log-warn{color:#fbbf24;}
.log-console .log-info{color:#38bdf8;}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ═══════════════════════════════════════════════════════════════════════════════
PATH_M1 = "data/lstm_final_clean"
PATH_M2 = "data/dataset_model2_1999_2023_clean"
LSTM_PATH   = "models/global_lstm.keras"
GRU_PATH    = "models/global_gru.keras"
SCALER_PATH = "models/scaler.pkl"

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

WINDOW       = 72
HORIZONS     = [1, 6, 12, 24]
TARGET_IDX   = 0
SEUIL_DANGER = 1.50
SEUIL_WATCH  = 1.0
SEUIL_WARN   = 0.5
FEATURES = [
    "MESURE","wind_speed","mwp","mwd_sin","mwd_cos",
    "hour_sin","hour_cos","month_sin","month_cos",
    "day_sin","day_cos","year_sin","year_cos",
    "x_norm","y_norm",
]

LAT_MIN, LAT_MAX = 36.7, 37.4
LON_MIN, LON_MAX = -1.8, 8.6

def apply_theme(fig): fig.update_layout(**PLOTLY_THEME); return fig
def section(icon, title): st.markdown(f'<div class="section-title">{icon} {title}</div>', unsafe_allow_html=True)
def page_header(icon_bg, icon, title, subtitle):
    st.markdown(f"""
    <div class="page-header">
        <div class="page-header-icon" style="background:{icon_bg}20;border:1px solid {icon_bg}40;">{icon}</div>
        <div><h1>{title}</h1><p>{subtitle}</p></div>
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# DUCKDB
# ═══════════════════════════════════════════════════════════════════════════════
data_m1_ok = os.path.exists(PATH_M1)
data_m2_ok = os.path.exists(PATH_M2)

def _collect_parquets(path):
    files = []
    if not path: return files
    if os.path.isdir(path):
        for r, _, fs in os.walk(path):
            for f in fs:
                if f.endswith(".parquet"):
                    files.append(os.path.join(r, f))
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
    mwp_expr  = _col("mwp"); mwd_expr = _col("mwd")
    sal_expr  = _col("salinity"); o2_expr = _col("o2"); spm_expr = _col("spm")

    try:
        avg_sst = con.execute(f"SELECT AVG(CAST(sst AS DOUBLE)) FROM read_parquet([{files_sql}]) LIMIT 100000").fetchone()[0]
        sst_conv = "CAST(sst AS DOUBLE) - 273.15" if avg_sst and avg_sst > 100 else "CAST(sst AS DOUBLE)"
    except: sst_conv = "NULL::DOUBLE"
    try:
        avg_msl = con.execute(f"SELECT AVG(CAST(msl AS DOUBLE)) FROM read_parquet([{files_sql}]) LIMIT 100000").fetchone()[0]
        msl_conv = "CAST(msl AS DOUBLE)/100.0" if avg_msl and avg_msl > 10000 else "CAST(msl AS DOUBLE)"
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
            {u10_expr} AS u10, {v10_expr} AS v10,
            {ws_expr}  AS wind_speed,
            {mwp_expr} AS mwp,
            {mwd_expr} AS mwd,
            ({sst_conv}) AS sst,
            ({msl_conv}) AS msl,
            {sal_expr} AS salinity,
            {o2_expr}  AS o2,
            {spm_expr} AS spm,
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
            CASE WHEN ({sst_conv}) BETWEEN 16 AND 26
                      AND CAST(MESURE AS DOUBLE)<=3.0
                      AND {ws_expr}<=10.0
                      AND ({msl_conv})>=1005.0
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
    LANG_FLAGS = {"fr": "🇫🇷 Français", "en": "🇬🇧 English", "ar": "🇩🇿 العربية"}
    current_flag = LANG_FLAGS.get(st.session_state.get("lang", "fr"), "🇫🇷 Français")
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
    </div>""", unsafe_allow_html=True)

    st.markdown(f"**{T('model_data')}**")
    model_choice = st.radio("Modèle", [T("model1_label"), T("model2_label")], label_visibility="collapsed")
    is_m2 = "M2" in model_choice or "N2" in model_choice
    VIEW = "hsv2" if is_m2 else "hsv"
    data_ok = data_m2_ok if is_m2 else data_m1_ok

    if is_m2:
        st.markdown("""<div style="background:rgba(139,92,246,.1);border:1px solid rgba(139,92,246,.3);border-radius:8px;padding:8px 12px;font-size:.75rem;color:#a78bfa;margin-bottom:.8rem;">
            🟣 <b>M2</b> · ERA5 + CMEMS<br><span style="color:#94b8cc;">salinity · o2 · spm · sst · mwp · wind_speed · MESURE</span></div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div style="background:rgba(14,165,233,.1);border:1px solid rgba(14,165,233,.3);border-radius:8px;padding:8px 12px;font-size:.75rem;color:#38bdf8;margin-bottom:.8rem;">
            🔵 <b>M1</b> · ERA5 seul<br><span style="color:#94b8cc;">MESURE · wind_speed · mwp · mwd</span></div>""", unsafe_allow_html=True)

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

    # ── NOUVEAU : Sous-menu Prédiction ─────────────────────────────────
    pred_model_page = None
    if page == T("realtime_pred"):
        st.markdown(
            "<div style='font-size:.68rem;text-transform:uppercase;letter-spacing:.12em;"
            "color:#4a7a96;font-weight:600;margin-top:.5rem;margin-bottom:.3rem;'>"
            f"🔮 {T('pred_submenu')}</div>",
            unsafe_allow_html=True
        )
        pred_model_page = st.radio(
            "Pred modèle",
            [T("pred_m1_label"), T("pred_m2_label")],
            label_visibility="collapsed"
        )
        # Badge contextuel selon le choix
        if pred_model_page == T("pred_m1_label"):
            active_badge_color = "#0ea5e9" if not is_m2 else "#ef4444"
            active_badge_text  = "✅ Modèle actif correspondant" if not is_m2 else "⚠️ Modèle actif : M2 — incompatible"
            border_c = "rgba(14,165,233,.3)" if not is_m2 else "rgba(239,68,68,.4)"
            bg_c     = "rgba(14,165,233,.07)" if not is_m2 else "rgba(239,68,68,.07)"
            st.markdown(
                f"<div style='background:{bg_c};border:1px solid {border_c};"
                f"border-radius:8px;padding:6px 10px;font-size:.72rem;"
                f"color:{active_badge_color};margin-top:4px;'>"
                f"{active_badge_text}</div>",
                unsafe_allow_html=True
            )
        else:
            active_badge_color = "#8b5cf6" if is_m2 else "#ef4444"
            active_badge_text  = "✅ Modèle actif correspondant" if is_m2 else "⚠️ Modèle actif : M1 — incompatible"
            border_c = "rgba(139,92,246,.3)" if is_m2 else "rgba(239,68,68,.4)"
            bg_c     = "rgba(139,92,246,.07)" if is_m2 else "rgba(239,68,68,.07)"
            st.markdown(
                f"<div style='background:{bg_c};border:1px solid {border_c};"
                f"border-radius:8px;padding:6px 10px;font-size:.72rem;"
                f"color:{active_badge_color};margin-top:4px;'>"
                f"{active_badge_text}</div>",
                unsafe_allow_html=True
            )

    st.markdown("---")
    all_wilayas, all_plages, all_years = get_lists(VIEW) if data_ok else ([], [], [])

    st.markdown(f"**{T('temporal_filters')}**")
    year_filter  = st.multiselect(T("year"),  all_years, default=[], placeholder=T("all_f"))
    month_filter = st.multiselect(T("month"), list(range(1, 13)), format_func=lambda x: T("months")[x], default=[], placeholder=T("all"))
    hour_filter  = st.multiselect(T("hour"),  list(range(0, 24)), format_func=lambda x: f"{x:02d}h00", default=[], placeholder=T("all_f"))

    st.markdown(f"**{T('geo_filters')}**")
    wilaya_filter = st.multiselect(T("wilaya"), all_wilayas, default=[], placeholder=T("all_f"))
    if wilaya_filter and data_ok:
        wil_in = ",".join(f"'{w}'" for w in wilaya_filter)
        plages_dispo = q(f"SELECT DISTINCT NOM_PLAGE FROM {VIEW} WHERE NOM_WILAYA IN ({wil_in}) ORDER BY NOM_PLAGE")["NOM_PLAGE"].tolist() if data_ok else []
    else:
        plages_dispo = all_plages
    plage_filter = st.multiselect(T("beach"), plages_dispo, default=[], placeholder=T("all_f"))

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
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
    if r.empty or r["total"].iloc[0] == 0: st.warning(T("no_filtered_data")); return
    total = r["total"].iloc[0]; avg_h = r["avg_hsv"].iloc[0]; max_h = r["max_hsv"].iloc[0]
    std_h = r["std_hsv"].iloc[0]; n15 = r["n15"].iloc[0]; n25 = r["n25"].iloc[0]
    n40 = r["n40"].iloc[0]; p95 = r["p95"].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"📦 {T('measures')}",  f"{int(total):,}")
    c2.metric(f"🌊 {T('avg_hsv')}",   f"{avg_h:.3f} m", f"± {std_h:.3f} m")
    c3.metric(f"⚠️ {T('max_hsv')}",   f"{max_h:.2f} m")
    c4.metric(f"📈 {T('p95')}",       f"{p95:.2f} m")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("🟡 ≥ 1.5 m", f"{int(n15):,}", f"{n15/total*100:.1f}%")
    c6.metric("🟠 ≥ 2.5 m", f"{int(n25):,}", f"{n25/total*100:.1f}%")
    c7.metric("🔴 ≥ 4.0 m", f"{int(n40):,}", f"{n40/total*100:.1f}%")
    c8.metric(f"📐 {T('std')}",       f"{std_h:.3f} m")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE : PRÉDICTION TEMPS RÉEL — ROUTEUR M1 / M2
# ═══════════════════════════════════════════════════════════════════════════════
if page == T("realtime_pred"):

    # ── Labels normalisés (indépendants de la langue) ──────────────────
    PRED_M1_LABEL = T("pred_m1_label")
    PRED_M2_LABEL = T("pred_m2_label")

    pred_wants_m1 = (pred_model_page == PRED_M1_LABEL)
    pred_wants_m2 = (pred_model_page == PRED_M2_LABEL)

    # ══════════════════════════════════════════════════════════════════════
    # CAS 1 — M2 demandé mais M1 actif → ERREUR INCOMPATIBILITÉ
    # ══════════════════════════════════════════════════════════════════════
    if pred_wants_m2 and not is_m2:
        st.markdown("""
        <div class="incompat-banner">
            <div class="ib-icon">🚫</div>
            <div class="ib-title">Incompatibilité de modèle</div>
            <div class="ib-body">
                Vous avez sélectionné <strong>Prédiction M2 — ERA5 + CMEMS</strong>
                mais le modèle de données actif est <strong>🔵 M1 — ERA5 seul</strong>.<br><br>
                Le modèle M2 nécessite les variables enrichies CMEMS
                (Salinité, O₂ dissous, SPM, SST) qui ne sont pas disponibles dans M1.
            </div>
            <div class="ib-step">
                👉 <strong>Solution :</strong> Dans la sidebar, section <em>MODÈLE DE DONNÉES</em>,
                sélectionnez <strong>🟣 M2 — ERA5 + CMEMS (1999–2023)</strong>
                puis revenez sur cette page.
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.markdown("""
            <div class="info-card" style="border-left-color:#0ea5e9;">
                <div class="title" style="color:#38bdf8;">🔵 M1 actuellement actif</div>
                <div class="threshold-row"><span class="threshold-key">Variables disponibles</span><span class="threshold-val" style="color:#38bdf8;">MESURE · wind_speed · mwp · mwd</span></div>
                <div class="threshold-row"><span class="threshold-key">Période</span><span class="threshold-val">1985 – 2023</span></div>
                <div class="threshold-row"><span class="threshold-key">Prédiction compatible</span><span class="threshold-val" style="color:#34d399;">✅ Prédiction M1</span></div>
            </div>""", unsafe_allow_html=True)
        with col_info2:
            st.markdown("""
            <div class="info-card" style="border-left-color:#8b5cf6;">
                <div class="title" style="color:#a78bfa;">🟣 M2 — Ce qu'il vous faut</div>
                <div class="threshold-row"><span class="threshold-key">Variables requises</span><span class="threshold-val" style="color:#a78bfa;">salinity · o2 · spm · sst</span></div>
                <div class="threshold-row"><span class="threshold-key">Période</span><span class="threshold-val">1999 – 2023</span></div>
                <div class="threshold-row"><span class="threshold-key">Action requise</span><span class="threshold-val" style="color:#fbbf24;">⚠️ Changer de modèle</span></div>
            </div>""", unsafe_allow_html=True)
        st.stop()

    # ══════════════════════════════════════════════════════════════════════
    # CAS 2 — M1 demandé mais M2 actif → ERREUR INCOMPATIBILITÉ
    # ══════════════════════════════════════════════════════════════════════
    elif pred_wants_m1 and is_m2:
        st.markdown("""
        <div class="incompat-banner">
            <div class="ib-icon">🚫</div>
            <div class="ib-title">Incompatibilité de modèle</div>
            <div class="ib-body">
                Vous avez sélectionné <strong>Prédiction M1 — ERA5 seul</strong>
                mais le modèle de données actif est <strong>🟣 M2 — ERA5 + CMEMS</strong>.<br><br>
                Le pipeline M1 utilise exclusivement les 15 features ERA5.
                Exécuter M1 avec un contexte M2 produirait des prédictions incohérentes.
            </div>
            <div class="ib-step">
                👉 <strong>Solution :</strong> Dans la sidebar, section <em>MODÈLE DE DONNÉES</em>,
                sélectionnez <strong>🔵 M1 — ERA5 seul (1985–2023)</strong>
                puis revenez sur cette page.
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.markdown("""
            <div class="info-card" style="border-left-color:#8b5cf6;">
                <div class="title" style="color:#a78bfa;">🟣 M2 actuellement actif</div>
                <div class="threshold-row"><span class="threshold-key">Variables disponibles</span><span class="threshold-val" style="color:#a78bfa;">MESURE · salinity · o2 · spm · sst · ...</span></div>
                <div class="threshold-row"><span class="threshold-key">Période</span><span class="threshold-val">1999 – 2023</span></div>
                <div class="threshold-row"><span class="threshold-key">Prédiction compatible</span><span class="threshold-val" style="color:#34d399;">✅ Prédiction M2</span></div>
            </div>""", unsafe_allow_html=True)
        with col_info2:
            st.markdown("""
            <div class="info-card" style="border-left-color:#0ea5e9;">
                <div class="title" style="color:#38bdf8;">🔵 M1 — Ce qu'il vous faut</div>
                <div class="threshold-row"><span class="threshold-key">Features LSTM</span><span class="threshold-val" style="color:#38bdf8;">15 features ERA5 uniquement</span></div>
                <div class="threshold-row"><span class="threshold-key">Période</span><span class="threshold-val">1985 – 2023</span></div>
                <div class="threshold-row"><span class="threshold-key">Action requise</span><span class="threshold-val" style="color:#fbbf24;">⚠️ Changer de modèle</span></div>
            </div>""", unsafe_allow_html=True)
        st.stop()

    # ══════════════════════════════════════════════════════════════════════
    # CAS 3 — M2 demandé + M2 actif → PLACEHOLDER (en développement)
    # ══════════════════════════════════════════════════════════════════════
    elif pred_wants_m2 and is_m2:
        page_header(
            "#8b5cf6", "🟣",
            "Prédiction Temps Réel — M2",
            "ERA5 + CMEMS (Salinité · O₂ · SPM · SST) → LSTM enrichi → HSV"
        )

        st.markdown("""
        <div class="wip-banner">
            <div class="wb-icon">🚧</div>
            <div class="wb-title">Module M2 en cours de développement</div>
            <div class="wb-body">
                Le pipeline de prédiction <strong>Modèle 2</strong> (ERA5 + CMEMS) est en cours de développement.<br>
                Il intégrera les variables oceanographiques CMEMS pour une prédiction HSV enrichie
                tenant compte de la qualité de l'eau, de la température de surface et des matières en suspension.<br><br>
                <strong>Disponibilité estimée :</strong> Prochaine version du dashboard.
            </div>
        </div>
        """, unsafe_allow_html=True)

        section("📐", "Architecture M2 prévue")

        col_arch1, col_arch2 = st.columns([1, 1])

        with col_arch1:
            st.markdown("""
            <table class="feat-table">
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>M1 (actuel)</th>
                        <th>M2 (prévu)</th>
                    </tr>
                </thead>
                <tbody>
                    <tr><td class="feat-name">MESURE (HSV)</td><td class="check-yes">✅</td><td class="check-yes">✅</td></tr>
                    <tr><td class="feat-name">wind_speed</td><td class="check-yes">✅</td><td class="check-yes">✅</td></tr>
                    <tr><td class="feat-name">mwp</td><td class="check-yes">✅</td><td class="check-yes">✅</td></tr>
                    <tr><td class="feat-name">mwd (sin/cos)</td><td class="check-yes">✅</td><td class="check-yes">✅</td></tr>
                    <tr><td class="feat-name">Temps cyclique</td><td class="check-yes">✅</td><td class="check-yes">✅</td></tr>
                    <tr><td class="feat-name">Coordonnées</td><td class="check-yes">✅</td><td class="check-yes">✅</td></tr>
                    <tr><td class="feat-name">salinity</td><td class="check-no">❌</td><td class="check-yes">✅</td></tr>
                    <tr><td class="feat-name">o2 dissous</td><td class="check-no">❌</td><td class="check-yes">✅</td></tr>
                    <tr><td class="feat-name">spm</td><td class="check-no">❌</td><td class="check-yes">✅</td></tr>
                    <tr><td class="feat-name">sst</td><td class="check-no">❌</td><td class="check-yes">✅</td></tr>
                </tbody>
            </table>
            """, unsafe_allow_html=True)

        with col_arch2:
            st.markdown("""
            <div class="info-card" style="border-left-color:#8b5cf6;margin-top:0;">
                <div class="title" style="color:#a78bfa;">🟣 Pipeline M2 prévu</div>
                <div class="threshold-row"><span class="threshold-key">Source 1</span><span class="threshold-val" style="color:#38bdf8;">ERA5 Réanalyse (Copernicus CDS)</span></div>
                <div class="threshold-row"><span class="threshold-key">Source 2</span><span class="threshold-val" style="color:#22d3ee;">CMEMS Marine Service</span></div>
                <div class="threshold-row"><span class="threshold-key">Source 3</span><span class="threshold-val" style="color:#34d399;">Open-Meteo Marine (forecast)</span></div>
                <div class="threshold-row"><span class="threshold-key">Features totales</span><span class="threshold-val" style="color:#a78bfa;">19 features (vs 15 pour M1)</span></div>
                <div class="threshold-row"><span class="threshold-key">Fenêtre contexte</span><span class="threshold-val">72h (identique à M1)</span></div>
                <div class="threshold-row"><span class="threshold-key">Horizons</span><span class="threshold-val">+1h · +6h · +12h · +24h</span></div>
                <div class="threshold-row"><span class="threshold-key">Architecture LSTM</span><span class="threshold-val" style="color:#a78bfa;">Transfer Learning depuis M1</span></div>
            </div>

            <div class="info-card" style="border-left-color:#10b981;">
                <div class="title" style="color:#34d399;">💡 Avantages M2</div>
                <div class="threshold-row"><span class="threshold-key">Salinité</span><span class="threshold-val">Densité eau → interaction vagues</span></div>
                <div class="threshold-row"><span class="threshold-key">O₂ dissous</span><span class="threshold-val">Proxy activité marine / upwelling</span></div>
                <div class="threshold-row"><span class="threshold-key">SPM</span><span class="threshold-val">Turbidité → amortissement vagues</span></div>
                <div class="threshold-row"><span class="threshold-key">SST</span><span class="threshold-val">Énergie thermique → intensité vagues</span></div>
            </div>
            """, unsafe_allow_html=True)

        section("🔮", "Prédiction M1 disponible maintenant")
        st.info(
            "⬅️ En attendant M2, la **Prédiction M1** est pleinement opérationnelle.\n\n"
            "Sélectionnez **🔵 Prédiction M1 — ERA5** dans le menu de gauche pour lancer une prédiction."
        )
        st.stop()

    # ══════════════════════════════════════════════════════════════════════
    # CAS 4 — M1 demandé + M1 actif → PIPELINE COMPLET
    # ══════════════════════════════════════════════════════════════════════

    page_header(
        "#8b5cf6",
        "🔮",
        "Prédiction Temps Réel — M1",
        "ERA5 (contexte 72h) + Open-Meteo Marine (forecast J+7) → LSTM → HSV → Alerte"
    )

    st.markdown("""
    <div style="display:inline-flex;align-items:center;gap:8px;background:rgba(14,165,233,.1);
         border:1px solid rgba(14,165,233,.3);border-radius:8px;padding:6px 14px;
         font-size:.8rem;color:#38bdf8;margin-bottom:1rem;">
        🔵 <strong>M1 — ERA5 seul</strong> &nbsp;·&nbsp; 15 features &nbsp;·&nbsp;
        LSTM Global + Transfer Learning &nbsp;·&nbsp; Horizons +1h / +6h / +12h / +24h
    </div>
    """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════
    # CONFIGURATION PIPELINE M1
    # ═══════════════════════════════════════════════════════════════════
    WINDOW   = 72
    HORIZONS = [1, 6, 12, 24]

    FEATURES = [
        "MESURE", "wind_speed", "mwp", "mwd_sin", "mwd_cos",
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "day_sin", "day_cos", "year_sin", "year_cos",
        "x_norm", "y_norm",
    ]

    DANGER_THRESHOLDS = {
        "vert":   (0.0, 0.5, "Mer calme",    "🟢"),
        "jaune":  (0.5, 1.0, "Mer agitée",    "🟡"),
        "orange": (1.0, 1.5, "Risque modéré", "🟠"),
        "rouge":  (1.5, 9.0, "DANGER",        "🔴"),
    }

    import glob
    import zipfile
    import traceback
    import joblib
    import cdsapi
    import xarray as xr
    import tensorflow as tf
    from collections import defaultdict

    def asymmetric_huber_loss(delta=0.5, underestimate_penalty=3.0):
        horizon_weights = tf.constant([1.0, 1.2, 1.5, 2.0], dtype=tf.float32)
        def loss(y_true, y_pred):
            error     = y_true - y_pred
            abs_error = tf.abs(error)
            quadratic = tf.minimum(abs_error, delta)
            linear    = abs_error - quadratic
            huber     = 0.5 * quadratic**2 + delta * linear
            weight_asym = tf.where(
                error > 0,
                tf.ones_like(error) * underestimate_penalty,
                tf.ones_like(error)
            )
            return tf.reduce_mean(huber * weight_asym * horizon_weights)
        loss.__name__ = "asymmetric_huber"
        return loss

    def asymmetric_huber(y_true, y_pred, delta=1.0, alpha=2.5):
        error     = y_true - y_pred
        abs_error = tf.abs(error)
        quadratic = tf.minimum(abs_error, delta)
        linear    = abs_error - quadratic
        loss      = 0.5 * tf.square(quadratic) + delta * linear
        weight    = tf.where(error > 0, alpha, 1.0)
        return tf.reduce_mean(loss * weight)

    CUSTOM_OBJECTS = {
        "asymmetric_huber":      asymmetric_huber,
        "asymmetric_huber_loss": asymmetric_huber_loss(),
        "loss":                  asymmetric_huber_loss(),
    }

    # ── Chargement modèles ─────────────────────────────────────────────
    @st.cache_resource(show_spinner="Chargement modèle LSTM global...")
    def load_global_lstm():
        return tf.keras.models.load_model(
            LSTM_PATH, custom_objects=CUSTOM_OBJECTS, compile=False
        )

    @st.cache_resource(show_spinner="Chargement scaler global...")
    def load_scaler():
        return joblib.load(SCALER_PATH)

    # ── Détection modèle local fine-tuné ──────────────────────────────
    # ✅ CORRECTION : les modèles locaux sont dans plots/ directement (sans sous-dossier)
    def load_finetuned_lstm(plage_name: str):
        base_dir  = os.path.dirname(LSTM_PATH)
        safe_name = plage_name.replace(" ", "_")

        #  Cherche directement dans plots/ sans sous-dossier
        model_path = os.path.join(
            base_dir, "plots", f"{safe_name}_lstm.keras"
        )

        if not os.path.exists(model_path):
            # Recherche alternative
            candidates = glob.glob(
                os.path.join(base_dir, "plots", "*lstm*.keras")
            )
            matched = [
                p for p in candidates
                if safe_name.lower() in os.path.basename(p).lower()
            ]
            if not matched:
                return None, False
            model_path = matched[0]

        try:
            model = tf.keras.models.load_model(
                model_path, custom_objects=CUSTOM_OBJECTS, compile=False
            )
            return model, True
        except Exception as e:
            st.warning(f"⚠️ Modèle local non chargeable : {e}. Fallback global.")
            return None, False

    # ── Inférence LSTM + fallback automatique ─────────────────────────
    def _run_inference(X_tensor, plage_name):
        local_model, used_local = load_finetuned_lstm(plage_name)
        if used_local and local_model is not None:
            model_used = local_model
            model_info = f"Fine-tuné — {plage_name}"
        else:
            model_used = load_global_lstm()
            used_local = False
            model_info = "Global (fallback)"
        preds = model_used.predict(X_tensor, verbose=0)[0]
        return preds, used_local, model_info

    # ── Vérification fichiers ──────────────────────────────────────────
    models_ok = all([os.path.exists(LSTM_PATH), os.path.exists(SCALER_PATH)])
    if not models_ok:
        st.error("❌ Modèle LSTM global ou scaler introuvable")
        st.info(f"LSTM attendu : `{LSTM_PATH}`\nScaler attendu : `{SCALER_PATH}`")
        st.stop()

    # ── Helpers affichage ──────────────────────────────────────────────
    def _danger_level(hsv: float):
        for key, (lo, hi, label, emoji) in DANGER_THRESHOLDS.items():
            if lo <= hsv < hi:
                colors = {"vert":"#22c55e","jaune":"#eab308","orange":"#f97316","rouge":"#ef4444"}
                return colors[key], label, emoji
        return "#ef4444", "DANGER", "🔴"

    def _model_badge(used_local: bool, plage: str) -> str:
        if used_local:
            return (f"<span style='background:#7c3aed;color:white;padding:3px 10px;"
                    f"border-radius:12px;font-size:0.8em;'>🎯 Modèle fine-tuné — {plage}</span>")
        return ("<span style='background:#2563eb;color:white;padding:3px 10px;"
                "border-radius:12px;font-size:0.8em;'>🌐 Modèle global (fallback)</span>")

    # ── Ouverture ERA5 ─────────────────────────────────────────────────
    def _open_era5_file(path: str, lat: float, lon: float):
        def _sel_point(ds):
            if "latitude" in ds.coords and "longitude" in ds.coords:
                ds = ds.sel(latitude=lat, longitude=lon, method="nearest")
            return ds.to_dataframe().reset_index()

        if zipfile.is_zipfile(path):
            tmp_extract = path + "_extract"
            os.makedirs(tmp_extract, exist_ok=True)
            with zipfile.ZipFile(path, "r") as zf:
                zf.extractall(tmp_extract)
            nc_files = glob.glob(os.path.join(tmp_extract, "*.nc"))
            if not nc_files:
                raise RuntimeError("Aucun .nc dans le ZIP ERA5")
            dfs = []
            for nc in nc_files:
                ds = xr.open_dataset(nc, engine="netcdf4")
                dfs.append(_sel_point(ds))
                ds.close()
            if len(dfs) == 1:
                return dfs[0]
            time_key = "valid_time" if "valid_time" in dfs[0].columns else "time"
            merged = dfs[0]
            for extra in dfs[1:]:
                new_cols = [c for c in extra.columns if c not in merged.columns or c == time_key]
                merged = merged.merge(extra[new_cols], on=time_key, how="outer")
            return merged

        for engine in ["netcdf4", "h5netcdf", "scipy"]:
            try:
                ds = xr.open_dataset(path, engine=engine)
                df = _sel_point(ds)
                ds.close()
                return df
            except Exception:
                continue
        raise RuntimeError(f"Impossible d'ouvrir ERA5 : {path}")

    # ── Téléchargement ERA5 ────────────────────────────────────────────
    def _fetch_era5_window(c, lat, lon, date_start, date_end, tmp_dir):
        days_needed = []
        d = date_start.date()
        while d <= date_end.date():
            days_needed.append(d)
            d += timedelta(days=1)

        month_groups = defaultdict(list)
        for day in days_needed:
            month_groups[(day.year, day.month)].append(day)

        frames = []
        for (yr, mo), days in month_groups.items():
            dl_path = os.path.join(tmp_dir, f"era5_{yr}_{mo:02d}.zip")
            c.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "variable": [
                        "significant_height_of_combined_wind_waves_and_swell",
                        "10m_u_component_of_wind",
                        "10m_v_component_of_wind",
                        "mean_wave_period",
                        "mean_wave_direction",
                    ],
                    "year":  [str(yr)],
                    "month": [f"{mo:02d}"],
                    "day":   [f"{x.day:02d}" for x in days],
                    "time":  [f"{h:02d}:00" for h in range(24)],
                    "area":  [lat + 0.5, lon - 0.5, lat - 0.5, lon + 0.5],
                    "format": "netcdf",
                },
                dl_path,
            )
            df_tmp = _open_era5_file(dl_path, lat, lon)
            frames.append(df_tmp)

        df = pd.concat(frames, ignore_index=True)
        col_map = {"swh": "MESURE","mwp": "mwp","mwd": "mwd","u10": "u10","v10": "v10"}
        df.rename(columns=col_map, inplace=True)
        time_col = "valid_time" if "valid_time" in df.columns else "time"
        df["DATETIME"] = pd.to_datetime(df[time_col], utc=True).dt.tz_localize(None)
        df = df.sort_values("DATETIME")
        df = df[(df["DATETIME"] >= date_start) & (df["DATETIME"] <= date_end)].copy()
        return df

    # ── Open-Meteo Marine (forecast gratuit) ──────────────────────────
    def _fetch_openmeteo_forecast(lat, lon, date_start, date_end):
        import urllib.request
        import json as _json

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
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = _json.loads(resp.read().decode())
        except Exception as e:
            raise RuntimeError(f"Erreur Open-Meteo Marine API : {e}")

        if "hourly" not in data:
            raise RuntimeError(f"Réponse Open-Meteo invalide. Réponse : {str(data)[:300]}")

        h = data["hourly"]
        df = pd.DataFrame({
            "DATETIME":       pd.to_datetime(h["time"]),
            "MESURE":         h.get("wave_height",      [np.nan] * len(h["time"])),
            "mwp":            h.get("wave_period",      [np.nan] * len(h["time"])),
            "mwd":            h.get("wave_direction",   [np.nan] * len(h["time"])),
            "wind_speed_raw": h.get("wind_speed_10m",   [np.nan] * len(h["time"])),
            "wind_dir_raw":   h.get("wind_direction_10m",[np.nan] * len(h["time"])),
        })
        ws  = df["wind_speed_raw"].fillna(0).values
        wd  = df["wind_dir_raw"].fillna(0).values
        df["u10"] = -ws * np.sin(np.deg2rad(wd))
        df["v10"] = -ws * np.cos(np.deg2rad(wd))
        df = df[(df["DATETIME"] >= date_start) & (df["DATETIME"] <= date_end)].copy()
        df = df.sort_values("DATETIME").reset_index(drop=True)
        df["MESURE"] = df["MESURE"].clip(0, 20).ffill().bfill().fillna(0.3)
        df["mwp"]    = df["mwp"].clip(0, 25).ffill().bfill().fillna(4.0)
        df["mwd"]    = df["mwd"].clip(0, 360).ffill().bfill().fillna(180.0)
        return df[["DATETIME", "MESURE", "mwp", "mwd", "u10", "v10"]]

    # ── Fusion ERA5 + Open-Meteo ───────────────────────────────────────
    def _build_context_window(era5_df, forecast_df, target_dt, window_h=72):
        frames = []
        if era5_df is not None and not era5_df.empty:
            frames.append(era5_df.copy())
        if forecast_df is not None and not forecast_df.empty:
            if not frames:
                frames.append(forecast_df.copy())
            else:
                last_era5_dt = pd.to_datetime(frames[-1]["DATETIME"]).max()
                fc_tail = forecast_df[forecast_df["DATETIME"] > last_era5_dt].copy()
                if not fc_tail.empty:
                    frames.append(fc_tail)
        if not frames:
            raise RuntimeError("Aucune donnée disponible (ERA5 ni Open-Meteo).")
        merged = pd.concat(frames, ignore_index=True)
        merged = merged.sort_values("DATETIME").drop_duplicates("DATETIME")
        window_end   = target_dt
        window_start = window_end - timedelta(hours=window_h - 1)
        merged = merged[(merged["DATETIME"] >= window_start) & (merged["DATETIME"] <= window_end)].copy()
        return merged.reset_index(drop=True)

    # ── Feature Engineering ────────────────────────────────────────────
    def _build_features(df, lat: float, lon: float):
        df = df.copy()
        for col in ["MESURE", "mwp", "mwd", "u10", "v10"]:
            if col not in df.columns:
                df[col] = 0.0
        df = df.ffill().bfill().fillna(0.0)
        df["MESURE"]     = np.clip(df["MESURE"], 0, 20)
        df["mwp"]        = np.clip(df["mwp"],    0, 25)
        df["wind_speed"] = np.clip(np.sqrt(df["u10"]**2 + df["v10"]**2), 0, 60)
        df["mwd_sin"]    = np.sin(np.deg2rad(df["mwd"]))
        df["mwd_cos"]    = np.cos(np.deg2rad(df["mwd"]))
        h = df["DATETIME"].dt.hour
        m = df["DATETIME"].dt.month
        d = df["DATETIME"].dt.day
        df["hour_sin"]   = np.sin(2 * np.pi * h / 24)
        df["hour_cos"]   = np.cos(2 * np.pi * h / 24)
        df["month_sin"]  = np.sin(2 * np.pi * m / 12)
        df["month_cos"]  = np.cos(2 * np.pi * m / 12)
        df["day_sin"]    = np.sin(2 * np.pi * d / 31)
        df["day_cos"]    = np.cos(2 * np.pi * d / 31)
        df["year_sin"]   = 0.0
        df["year_cos"]   = 0.0
        df["x_norm"]     = np.clip((lon + 2) / 11, 0, 1)
        df["y_norm"]     = np.clip((lat - 36) / 2,  0, 1)
        for feat in FEATURES:
            if feat not in df.columns:
                df[feat] = 0.0
        return df[FEATURES + [c for c in df.columns if c not in FEATURES]]

    # ── Tenseur LSTM ───────────────────────────────────────────────────
    def _prepare_window(df, scaler):
        X = df[FEATURES].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = scaler.transform(X)
        X = np.clip(X, -10, 10)
        if len(X) < WINDOW:
            pad = np.tile(X[0], (WINDOW - len(X), 1))
            X   = np.vstack([pad, X])
        X = X[-WINDOW:]
        return X.reshape(1, WINDOW, len(FEATURES))

    # ── Calibration physique ───────────────────────────────────────────
    def _physical_calibration(preds, df_feat, pred_dt):
        preds  = np.array(preds, dtype=float)
        n      = len(df_feat)
        month  = pred_dt.month
        window_sizes = [1, 6, 12, 24]
        wind_series  = df_feat["wind_speed"].values
        wind_trend   = float(np.polyfit(np.arange(min(12, n)), wind_series[-min(12, n):], 1)[0])
        mwd_sin_last = float(df_feat["mwd_sin"].iloc[-1]) if "mwd_sin" in df_feat.columns else 0.0
        mwd_cos_last = float(df_feat["mwd_cos"].iloc[-1]) if "mwd_cos" in df_feat.columns else 1.0
        mwd_coast_factor = max(0.8, 1.0 + 0.15 * (-mwd_sin_last))
        calibrated = np.empty(len(HORIZONS), dtype=float)
        for idx, (h_horizon, w_size) in enumerate(zip(HORIZONS, window_sizes)):
            raw_pred = float(preds[idx]) if idx < len(preds) else float(preds[-1])
            sub = df_feat.tail(w_size) if n >= w_size else df_feat
            mean_wind  = float(sub["wind_speed"].mean())
            max_hsv    = float(sub["MESURE"].max())
            recent_hsv = float(sub["MESURE"].iloc[-1])
            if month in [11, 12, 1, 2, 3]:
                winter_floor = 0.8 + 0.03 * h_horizon
                raw_pred = max(raw_pred, winter_floor)
                wind_mult = 1.0
                if mean_wind > 7:  wind_mult *= 1.08 + 0.005 * h_horizon
                if mean_wind > 10: wind_mult *= 1.15 + 0.008 * h_horizon
                if mean_wind > 14: wind_mult *= 1.25 + 0.010 * h_horizon
                if max_hsv > 2:    wind_mult *= 1.10 + 0.004 * h_horizon
                raw_pred *= wind_mult
            else:
                wind_mult = 1.0
                if mean_wind > 8:  wind_mult *= 1.06 + 0.003 * h_horizon
                if mean_wind > 12: wind_mult *= 1.12 + 0.005 * h_horizon
                raw_pred *= wind_mult
            raw_pred *= (1.0 + (mwd_coast_factor - 1.0) * (h_horizon / 24.0))
            if wind_trend > 0.3:
                raw_pred *= (1.0 + 0.02 * wind_trend * (h_horizon / 6.0))
            elif wind_trend < -0.3:
                raw_pred *= max(0.85, 1.0 - 0.015 * abs(wind_trend) * (h_horizon / 6.0))
            p25 = float(np.percentile(sub["MESURE"], 25))
            t   = h_horizon / 24.0
            floor = recent_hsv * 0.85 * (1 - t) + max(0.30, p25) * t
            raw_pred = max(raw_pred, floor)
            calibrated[idx] = np.clip(raw_pred, 0.05, 9.0)
        return calibrated

    # ── LAYOUT ────────────────────────────────────────────────────────
    col_cfg, col_res = st.columns([1, 1.6])

    with col_cfg:
        section("⚙️", "Configuration")

        if data_ok:
            plages_all  = q(f"""
                SELECT DISTINCT NOM_PLAGE,
                       FIRST(X) AS lon,
                       FIRST(Y) AS lat
                FROM {VIEW}
                GROUP BY NOM_PLAGE
                ORDER BY NOM_PLAGE
            """)
            plage_names = plages_all["NOM_PLAGE"].tolist()
        else:
            plages_all  = pd.DataFrame()
            plage_names = []

        selected_plage = st.selectbox("🏖️ Plage", plage_names)

        if not plages_all.empty and selected_plage:
            row         = plages_all[plages_all["NOM_PLAGE"] == selected_plage].iloc[0]
            lat_default = float(row["lat"])
            lon_default = float(row["lon"])
        else:
            lat_default = 36.75
            lon_default = 3.06

        lat = st.number_input("Latitude",  value=lat_default, format="%.4f")
        lon = st.number_input("Longitude", value=lon_default, format="%.4f")

        st.markdown("---")

        today        = datetime.utcnow().date()
        era5_cutoff  = today - timedelta(days=5)
        max_date     = today + timedelta(days=1)
        default_date = today + timedelta(days=1)

        pred_date = st.date_input(
            "📅 Date cible de prédiction",
            value=default_date,
            min_value=today - timedelta(days=30),
            max_value=max_date,
            help=(
                "• **Passé récent (≤ J-5)** : ERA5 seul\n"
                "• **Futur (J-4 à J+1)** : ERA5 + Open-Meteo Marine (gratuit, sans clé)"
            ),
        )
        pred_dt = datetime.combine(pred_date, datetime.min.time())

        days_from_now = (pred_date - today).days
        if pred_date <= era5_cutoff:
            st.info(f"📡 **Mode ERA5 pur** · Date historique")
            use_forecast = False
        elif pred_date <= today:
            st.warning(f"🔀 **Mode hybride** · ERA5 + Open-Meteo")
            use_forecast = True
        else:
            st.success(f"🚀 **Mode forecast** · ERA5 + Open-Meteo Marine\nPrédiction pour **{pred_date.strftime('%d/%m/%Y')}**")
            use_forecast = True

        st.markdown("---")

        cds_key = st.text_input(
            "🔑 CDS API KEY (ERA5)",
            type="password",
            help="Clé Copernicus CDS — https://cds.climate.copernicus.eu\nOptionnelle si Open-Meteo suffit.",
        )
        if use_forecast and not cds_key.strip():
            st.caption("ℹ️ Sans clé CDS, le contexte ERA5 sera remplacé par Open-Meteo.")

        #  Cherche dans plots/ directement
        safe_name  = selected_plage.replace(" ", "_") if selected_plage else ""
        base_dir   = os.path.dirname(LSTM_PATH)
        local_path = os.path.join(base_dir, "plots", f"{safe_name}_lstm.keras")
        has_local  = os.path.exists(local_path)
        if has_local:
            st.success(f"✅ Modèle fine-tuné disponible pour **{selected_plage}**")
        else:
            st.info("ℹ️ Modèle global utilisé pour cette plage")

        run_btn = st.button("🚀 Lancer prédiction", type="primary", use_container_width=True)

    with col_res:
        section("📊", "Résultats")
        results_area = st.empty()

    # ── PIPELINE PRINCIPAL ────────────────────────────────────────────
    if run_btn:
        try:
            now_utc      = datetime.utcnow()
            era5_cutoff  = now_utc - timedelta(days=5)
            window_start = pred_dt - timedelta(hours=WINDOW)
            window_end   = pred_dt

            progress = st.progress(0)
            status   = st.empty()

            status.info("⚙️ Chargement du scaler global...")
            progress.progress(8)
            scaler = load_scaler()

            df_era5     = None
            df_forecast = None
            sources_used = []

            era5_window_end   = min(window_end, era5_cutoff)
            era5_window_start = era5_window_end - timedelta(hours=WINDOW)

            if era5_window_end > era5_window_start and cds_key.strip():
                try:
                    status.info("🛰️ Téléchargement ERA5 (Copernicus CDS)...")
                    progress.progress(20)
                    os.environ["CDSAPI_KEY"] = cds_key.strip()
                    os.environ["CDSAPI_URL"] = "https://cds.climate.copernicus.eu/api"
                    c_era5  = cdsapi.Client(quiet=True)
                    tmp_dir = tempfile.mkdtemp()
                    df_era5 = _fetch_era5_window(
                        c_era5, lat, lon,
                        era5_window_start, era5_window_end,
                        tmp_dir
                    )
                    if not df_era5.empty:
                        sources_used.append(
                            f"✅ ERA5 ({len(df_era5)} pts · "
                            f"{era5_window_start.strftime('%d/%m %Hh')} → "
                            f"{era5_window_end.strftime('%d/%m %Hh')})"
                        )
                except Exception as e_era5:
                    st.warning(f"⚠️ ERA5 non disponible : {e_era5}")
                    df_era5 = None

            if use_forecast or df_era5 is None:
                try:
                    status.info("🌊 Récupération Open-Meteo Marine (forecast gratuit)...")
                    progress.progress(35)
                    om_start = window_start
                    om_end   = window_end + timedelta(hours=24)
                    df_forecast = _fetch_openmeteo_forecast(lat, lon, om_start, om_end)
                    if not df_forecast.empty:
                        sources_used.append(
                            f"✅ Open-Meteo Marine ({len(df_forecast)} pts · "
                            f"{om_start.strftime('%d/%m %Hh')} → "
                            f"{window_end.strftime('%d/%m %Hh')})"
                        )
                except Exception as e_om:
                    st.warning(f"⚠️ Open-Meteo Marine : {e_om}")
                    df_forecast = None

            if df_era5 is None and df_forecast is None:
                st.error("❌ Aucune source de données disponible.")
                st.stop()

            status.info("🔀 Fusion des sources de données (72h)...")
            progress.progress(50)
            df_raw = _build_context_window(df_era5, df_forecast, pred_dt, WINDOW)

            if len(df_raw) < 10:
                st.error(f"❌ Données insuffisantes : {len(df_raw)} pas de temps.")
                st.stop()

            for src in sources_used:
                st.caption(src)

            status.info("🔧 Construction des features (15 variables)...")
            progress.progress(62)
            df_feat = _build_features(df_raw, lat, lon)

            status.info("📐 Préparation du tenseur (1, 72, 15)...")
            progress.progress(72)
            X_tensor = _prepare_window(df_feat, scaler)

            status.info("🧠 Inférence LSTM (Transfer Learning)...")
            progress.progress(84)
            preds_raw, used_local, model_info = _run_inference(X_tensor, selected_plage)

            status.info("⚖️ Calibration physique par horizon...")
            progress.progress(94)
            preds = _physical_calibration(preds_raw, df_feat, pred_dt)
            preds = np.nan_to_num(preds, nan=1.5, posinf=5.0, neginf=0.5)
            preds = np.clip(preds, 0.05, 9.0)

            progress.progress(100)
            status.empty()

            # ── Affichage résultats ────────────────────────────────────
            with results_area.container():
                st.markdown(_model_badge(used_local, selected_plage), unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                st.success(f"✅ Prévision HSV — {selected_plage}")
                cols = st.columns(len(HORIZONS))
                for i, (h, col) in enumerate(zip(HORIZONS, cols)):
                    hsv             = float(preds[i])
                    color, label, emoji = _danger_level(hsv)
                    delta_str = "Référence" if i == 0 else f"{hsv - float(preds[0]):+.2f} m vs t+1h"
                    with col:
                        st.metric(f"t+{h}h", f"{hsv:.2f} m", delta=delta_str)
                        st.markdown(
                            f"<div style='text-align:center;color:{color};font-weight:700'>{emoji} {label}</div>",
                            unsafe_allow_html=True
                        )

                marker_colors   = [_danger_level(float(p))[0] for p in preds]
                horizon_labels  = []
                for h in HORIZONS:
                    dt_h = pred_dt + timedelta(hours=h)
                    horizon_labels.append(f"t+{h}h<br><sub>{dt_h.strftime('%d/%m %Hh')}</sub>")

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=[f"t+{h}h" for h in HORIZONS],
                    y=preds,
                    mode="lines+markers",
                    line=dict(width=4, color="#8b5cf6"),
                    marker=dict(size=14, color=marker_colors, line=dict(width=2, color="white")),
                    name="HSV (m)",
                    customdata=[[float(preds[i]), horizon_labels[i], _danger_level(float(preds[i]))[1]] for i in range(len(HORIZONS))],
                    hovertemplate="<b>%{customdata[1]}</b><br>HSV : <b>%{y:.2f} m</b><br>Niveau : %{customdata[2]}<extra></extra>",
                ))
                fig.add_hrect(y0=1.5, y1=9.0, fillcolor="rgba(239,68,68,0.08)", line_width=0,
                              annotation_text="⚠️ Danger", annotation_position="top left")
                fig.add_hrect(y0=1.0, y1=1.5, fillcolor="rgba(249,115,22,0.07)", line_width=0)
                fig.add_hline(y=1.5, line_dash="dash", line_color="red", line_width=2, annotation_text="Seuil danger 1.5m")
                fig.add_hline(y=1.0, line_dash="dot",  line_color="orange", line_width=1, annotation_text="Seuil modéré 1.0m")
                for i, (h, p) in enumerate(zip(HORIZONS, preds)):
                    fig.add_annotation(x=f"t+{h}h", y=float(p), text=f"<b>{float(p):.2f}m</b>",
                                       showarrow=False, yshift=16, font=dict(size=11, color=marker_colors[i]))

                model_label = '<span style="color:#7c3aed">🎯 Fine-tuné</span>' if used_local else '🌐 Global'
                src_label   = "ERA5 + Open-Meteo Marine" if use_forecast else "ERA5 Réanalyse"

                fig.update_layout(
                    title=dict(
                        text=(f"Prévision HSV — {selected_plage} | "
                              f"{pred_date.strftime('%d/%m/%Y')} | "
                              f"{model_label}"),
                        font=dict(size=14)
                    ),
                    yaxis_title="Hauteur Significative des Vagues (m)",
                    yaxis=dict(range=[0, max(max(preds) * 1.25, 2.0)]),
                    height=380,
                    showlegend=False,
                )
                apply_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("🔍 Données de contexte utilisées (72h)"):
                    n_pts     = len(df_feat)
                    ctx_start = pred_dt - timedelta(hours=WINDOW)
                    st.caption(
                        f"**Source** : {src_label} · **{n_pts} pas de temps** | "
                        f"{ctx_start.strftime('%d/%m %Hh')} → {pred_dt.strftime('%d/%m %Hh')} | "
                        f"Lat={lat:.4f} Lon={lon:.4f}"
                    )
                    display_cols = ["DATETIME", "MESURE", "wind_speed", "mwp", "mwd"]
                    display_cols = [c for c in display_cols if c in df_feat.columns]
                    df_display   = df_feat[display_cols].tail(72).copy()
                    if "mwd" in df_display.columns:
                        df_display["mwd"] = df_display["mwd"].round(1).astype(str) + "°"
                    st.dataframe(df_display, use_container_width=True)

                    if "mwd" in df_feat.columns:
                        st.markdown("**🧭 Rose des vagues (MWD) — 72h**")
                        mwd_vals = df_feat["mwd"].dropna().values
                        if len(mwd_vals) > 0:
                            bins = np.arange(0, 361, 22.5)
                            counts, _ = np.histogram(mwd_vals, bins=bins)
                            angles = bins[:-1] + 11.25
                            fig_rose = go.Figure(go.Barpolar(
                                r=counts, theta=angles, width=22.5,
                                marker_color="#8b5cf6",
                                marker_line_color="rgba(139,92,246,0.3)",
                                marker_line_width=1, opacity=0.85,
                            ))
                            fig_rose.update_layout(
                                polar=dict(
                                    radialaxis=dict(showticklabels=False, ticks=""),
                                    angularaxis=dict(
                                        tickmode="array",
                                        tickvals=[0,45,90,135,180,225,270,315],
                                        ticktext=["N","NE","E","SE","S","SO","O","NO"],
                                        direction="clockwise", rotation=90,
                                    ),
                                ),
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(12,24,41,0.6)",
                                font=dict(color="#94b8cc"),
                                showlegend=False, height=280,
                                margin=dict(l=10,r=10,t=20,b=10),
                            )
                            st.plotly_chart(fig_rose, use_container_width=True)
                        col_mwd1, col_mwd2, col_mwd3 = st.columns(3)
                        mwd_mean = float(df_feat["mwd"].mean())
                        mwd_last = float(df_feat["mwd"].iloc[-1])
                        col_mwd1.metric("📐 Dir. moyenne",  f"{mwd_mean:.1f}°")
                        col_mwd2.metric("📐 Dir. actuelle", f"{mwd_last:.1f}°")
                        def _mwd_label(d):
                            d = d % 360
                            if d < 22.5 or d >= 337.5: return "Nord ↑"
                            elif d < 67.5:  return "Nord-Est ↗"
                            elif d < 112.5: return "Est →"
                            elif d < 157.5: return "Sud-Est ↘"
                            elif d < 202.5: return "Sud ↓"
                            elif d < 247.5: return "Sud-Ouest ↙"
                            elif d < 292.5: return "Ouest ←"
                            else:           return "Nord-Ouest ↖"
                        col_mwd3.metric("🧭 Quadrant actuel", _mwd_label(mwd_last))

                with st.expander("⚙️ Détails techniques"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"""
**Modèle** : `{model_info}`
**Architecture** : LSTM Multi-Horizon
**Fenêtre** : {WINDOW}h
**Horizons** : +1h, +6h, +12h, +24h
                        """)
                    with c2:
                        st.markdown(f"""
**Features** : {len(FEATURES)}
**Source données** : {src_label}
**Points de contexte** : {n_pts}
**Vent moyen** : {df_feat['wind_speed'].mean():.1f} m/s
**HSV max (contexte)** : {df_feat['MESURE'].max():.2f} m
                        """)

                    comp_df = pd.DataFrame({
                        "Horizon":      [f"t+{h}h" for h in HORIZONS],
                        "Heure réelle": [(pred_dt + timedelta(hours=h)).strftime('%d/%m %H:%M') for h in HORIZONS],
                        "Brut (m)":     [f"{float(preds_raw[i]):.3f}" for i in range(len(HORIZONS))],
                        "Calibré (m)":  [f"{float(preds[i]):.3f}"     for i in range(len(HORIZONS))],
                        "Δ vs t+1h":    ["—"] + [f"{float(preds[i]) - float(preds[0]):+.3f} m" for i in range(1, len(HORIZONS))],
                        "Niveau":       [_danger_level(float(preds[i]))[2] + " " + _danger_level(float(preds[i]))[1] for i in range(len(HORIZONS))],
                    })
                    st.dataframe(comp_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"❌ Erreur pipeline : {e}")
            with st.expander("🐛 Traceback complet"):
                st.code(traceback.format_exc())
# ═══════════════════════════════════════════════════════════════════════════════
# PAGES EXISTANTES
# ═══════════════════════════════════════════════════════════════════════════════

elif page == T("home"):
    st.markdown(f"""
    <div class="hero">
        <h1>{T("hero_title")}</h1>
        <div class="sub">{T("hero_sub")}<br><b>M1</b> {T("hero_sub2")}</div>
        <div class="pills">
            <span class="pill pill-red">{T("drowning_alerts_pill")}</span>
            <span class="pill pill-blue">{T("desalination_pill")}</span>
            <span class="pill pill-green">{T("aquaculture_pill")}</span>
            <span class="pill pill-cyan">{T("marine_quality_pill")}</span>
            <span class="pill pill-purple">{T("two_models_pill")}</span>
            <span class="pill pill-amber">🔮 Prédiction Temps Réel M1 + M2</span>
        </div>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-card" style="border-left-color:#0ea5e9;">
            <div class="title" style="color:#38bdf8;">🔵 Modèle 1 — ERA5 (1985–2023)</div>
            <div class="threshold-row"><span class="threshold-key">Période</span><span class="threshold-val" style="color:#38bdf8;">1985 → 2023</span></div>
            <div class="threshold-row"><span class="threshold-key">Lignes</span><span class="threshold-val">~20 millions</span></div>
            <div class="threshold-row"><span class="threshold-key">Variables</span><span class="threshold-val">MESURE · wind_speed · mwp · mwd</span></div>
            <div class="threshold-row"><span class="threshold-key">Application</span><span class="threshold-val" style="color:#f87171;">⚠️ Alertes Noyades + Prédiction TR M1</span></div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="info-card" style="border-left-color:#8b5cf6;">
            <div class="title" style="color:#a78bfa;">🟣 Modèle 2 — ERA5 + CMEMS (1999–2023)</div>
            <div class="threshold-row"><span class="threshold-key">Période</span><span class="threshold-val" style="color:#a78bfa;">1999 → 2023</span></div>
            <div class="threshold-row"><span class="threshold-key">Lignes</span><span class="threshold-val">~12 millions</span></div>
            <div class="threshold-row"><span class="threshold-key">Dessalement</span><span class="threshold-val" style="color:#22d3ee;">salinity · spm · MESURE · mwp · wind_speed</span></div>
            <div class="threshold-row"><span class="threshold-key">Aquaculture</span><span class="threshold-val" style="color:#34d399;">o2 · sst · mwp · wind_speed · MESURE · spm</span></div>
        </div>""", unsafe_allow_html=True)

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

elif page == T("analysis"):
    if analysis_page == T("global_analysis"):
        badge = "🟣 M2" if is_m2 else "🔵 M1"
        page_header("#0ea5e9","📊","Analyse Globale",f"Distribution et tendances HSV — {badge}")
        wh = W(); show_kpis(wh)
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
            df_pl = q(f"""SELECT NOM_PLAGE,NOM_WILAYA,AVG(MESURE) AS avg_hsv,MAX(MESURE) AS max_hsv,
                       STDDEV(MESURE) AS std_hsv,COUNT(*) AS n,
                       SUM(CASE WHEN MESURE>=2 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_danger
                FROM {VIEW} {wh} GROUP BY NOM_PLAGE,NOM_WILAYA ORDER BY avg_hsv DESC LIMIT 30""")
            if not df_pl.empty:
                fig = go.Figure(go.Bar(x=df_pl["avg_hsv"],y=df_pl["NOM_PLAGE"],orientation="h",
                    marker_color=df_pl["avg_hsv"].apply(lambda v:"#ef4444" if v>=1.5 else "#f59e0b" if v>=1 else "#10b981"),
                    text=df_pl["avg_hsv"].map(lambda v:f"{v:.2f} m"),textposition="outside"))
                apply_theme(fig); fig.update_layout(title="Top 30 plages — HSV moyenne",xaxis_title="HSV (m)",
                    height=max(400,len(df_pl)*22),yaxis=dict(autorange="reversed",**PLOTLY_THEME["yaxis"]))
                st.plotly_chart(fig,use_container_width=True)

    elif analysis_page == T("summer_analysis"):
        page_header("#f97316","🏖️","Analyse Estivale","Juin · Juillet · Août")
        wh_ete = where_clause_with_extra("MONTH IN (6,7,8)")
        section("📊","KPIs Estivaux"); show_kpis(wh_ete)
        st.info("Contenu de l'analyse estivale — cf. version originale")
    else:
        st.info("👈 Sélectionnez un type d'analyse dans le menu de gauche.")

elif page == T("activities"):
    st.info(f"Page activités ({activity_page}) — cf. code original complet intégré.")

elif page == T("synthesis"):
    page_header("#8b5cf6","📋","Synthèse & Export","Tableaux récapitulatifs et téléchargement")
    wh = W()
    tab1,tab2,tab3 = st.tabs([T("synth_by_beach"),T("monthly_synth"),T("export")])
    with tab1:
        df_synth = q(f"""SELECT NOM_PLAGE,NOM_WILAYA,COUNT(*) AS n,
                       ROUND(AVG(MESURE),3) AS avg_hsv,ROUND(MAX(MESURE),2) AS max_hsv,
                       ROUND(STDDEV(MESURE),3) AS std_hsv,
                       ROUND(SUM(CASE WHEN MESURE>=2 THEN 1 ELSE 0 END)*100.0/COUNT(*),1) AS pct_danger
                FROM {VIEW} {wh} GROUP BY NOM_PLAGE,NOM_WILAYA ORDER BY avg_hsv DESC""")
        if not df_synth.empty:
            st.dataframe(df_synth,use_container_width=True,hide_index=True)
            st.download_button(T("download_csv"),df_synth.to_csv(index=False).encode("utf-8"),"synthese.csv","text/csv")

elif page == T("danger_map"):
    page_header("#ef4444","🗺️","Carte des Dangers","Cartographie interactive — côtes algériennes")
    wh = W()
    df_map = q(f"""SELECT NOM_PLAGE,NOM_WILAYA,FIRST(X) AS lon,FIRST(Y) AS lat,
                   ROUND(AVG(MESURE),3) AS avg_hsv,ROUND(MAX(MESURE),2) AS max_hsv,
                   ROUND(SUM(CASE WHEN MESURE>=2 THEN 1 ELSE 0 END)*100.0/COUNT(*),1) AS pct_danger
                FROM {VIEW} {wh} GROUP BY NOM_PLAGE,NOM_WILAYA HAVING lon IS NOT NULL AND lat IS NOT NULL""")
    if not df_map.empty:
        fig = px.scatter_mapbox(df_map,lat="lat",lon="lon",color="avg_hsv",size="avg_hsv",
            hover_name="NOM_PLAGE",hover_data={"NOM_WILAYA":True,"avg_hsv":":.3f","max_hsv":":.2f","pct_danger":":.1f","lon":False,"lat":False},
            color_continuous_scale="Blues",size_max=20,zoom=5,mapbox_style="carto-darkmatter",labels={"avg_hsv":"HSV Moy (m)"})
        apply_theme(fig); fig.update_layout(height=520,margin=dict(l=0,r=0,t=40,b=0),title="Côtes algériennes — HSV Moyenne")
        st.plotly_chart(fig,use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
model_info = "ERA5 + CMEMS (M2) · 1999–2023" if is_m2 else "ERA5 (M1) · 1985–2023"
lang_info  = {"fr":"Français","en":"English","ar":"العربية"}.get(st.session_state.get("lang","fr"),"Français")
st.markdown(f"""
<div style="text-align:center;padding:1rem 0;font-size:.75rem;color:var(--text-m);">
    Système HSV · Côtes Algériennes · {model_info} &nbsp;·&nbsp;
    LSTM + Transfer Learning &nbsp;·&nbsp;
    DuckDB + Streamlit + Plotly &nbsp;·&nbsp;
    Copernicus CDS ERA5 &nbsp;·&nbsp;
    🌐 {lang_info}
</div>""", unsafe_allow_html=True)
