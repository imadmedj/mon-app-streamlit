"""
╔══════════════════════════════════════════════════════════════════════════╗
║   DASHBOARD HSV — CÔTES ALGÉRIENNES                                     ║
║   Hauteur Significative des Vagues · Noyades · Dessalement · Aquaculture║
║   Dataset : data/lstm_final_clean  (~20 millions de lignes)             ║
║   Optimisé : DuckDB (requêtes SQL directement sur Parquet)              ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import duckdb
import os

st.set_page_config(
    page_title="HSV · Côtes Algériennes",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Syne:wght@600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-base:       #080f1a;
    --bg-surface:    #0c1829;
    --bg-card:       #0f2035;
    --bg-card-hover: #132640;
    --border-soft:   rgba(30,90,150,0.25);
    --border-mid:    rgba(30,110,180,0.4);
    --border-bright: rgba(14,157,232,0.6);
    --accent-1:      #0ea5e9;
    --accent-2:      #06b6d4;
    --accent-3:      #10b981;
    --warn:          #f59e0b;
    --danger:        #ef4444;
    --purple:        #8b5cf6;
    --text-h:        #f0f9ff;
    --text-p:        #94b8cc;
    --text-muted:    #4a7a96;
    --font-display:  'Syne', sans-serif;
    --font-body:     'DM Sans', sans-serif;
    --font-mono:     'JetBrains Mono', monospace;
    --r-sm: 8px; --r-md: 12px; --r-lg: 16px; --r-xl: 20px;
}

html, body, [class*="css"] { font-family: var(--font-body) !important; background: var(--bg-base) !important; color: var(--text-p) !important; }
.stApp { background: var(--bg-base) !important; }
.block-container { padding: 1.5rem 2rem 3rem !important; }

[data-testid="stSidebar"] { background: var(--bg-surface) !important; border-right: 1px solid var(--border-soft) !important; }
[data-testid="stSidebar"] * { color: var(--text-p) !important; }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 { font-family: var(--font-display) !important; color: var(--text-h) !important; }
[data-testid="stSidebar"] label { font-size: 0.68rem !important; text-transform: uppercase; letter-spacing: 0.12em; color: var(--text-muted) !important; font-weight: 600 !important; }
[data-testid="stSidebar"] .stSelectbox > div > div,[data-testid="stSidebar"] .stMultiSelect > div > div { background: var(--bg-card) !important; border: 1px solid var(--border-soft) !important; border-radius: var(--r-md) !important; color: var(--text-p) !important; }

[data-testid="stMetric"] { background: var(--bg-card) !important; border: 1px solid var(--border-soft) !important; border-radius: var(--r-lg) !important; padding: 1.1rem 1.3rem !important; position: relative; overflow: hidden; transition: border-color 0.2s, background 0.2s; }
[data-testid="stMetric"]:hover { border-color: var(--border-mid) !important; background: var(--bg-card-hover) !important; }
[data-testid="stMetric"]::after { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, var(--accent-1), var(--accent-2)); }
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: 0.7rem !important; text-transform: uppercase !important; letter-spacing: 0.12em !important; font-weight: 600 !important; }
[data-testid="stMetricValue"] { font-family: var(--font-display) !important; color: var(--text-h) !important; font-size: 1.7rem !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

.stTabs [data-baseweb="tab-list"] { background: var(--bg-card) !important; border: 1px solid var(--border-soft) !important; border-radius: var(--r-md) !important; padding: 4px !important; gap: 3px; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: var(--text-muted) !important; border-radius: var(--r-sm) !important; font-size: 0.8rem !important; font-weight: 600 !important; letter-spacing: 0.04em; padding: 7px 18px !important; border: none !important; transition: all 0.2s; }
.stTabs [aria-selected="true"] { background: var(--accent-1) !important; color: white !important; }

[data-testid="stDataFrame"] { border: 1px solid var(--border-soft) !important; border-radius: var(--r-lg) !important; overflow: hidden; }

.stDownloadButton > button, .stButton > button { background: var(--accent-1) !important; color: white !important; border: none !important; border-radius: var(--r-md) !important; font-weight: 600 !important; font-family: var(--font-body) !important; letter-spacing: 0.04em; padding: 0.5rem 1.4rem !important; transition: opacity 0.2s; }
.stDownloadButton > button:hover, .stButton > button:hover { opacity: 0.85 !important; }

hr { border: none; border-top: 1px solid var(--border-soft) !important; margin: 1.5rem 0 !important; }

.page-header { display: flex; align-items: center; gap: 12px; margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 1px solid var(--border-soft); }
.page-header-icon { width: 40px; height: 40px; border-radius: var(--r-md); display: flex; align-items: center; justify-content: center; font-size: 1.2rem; flex-shrink: 0; }
.page-header h1 { font-family: var(--font-display); font-size: 1.4rem; font-weight: 700; color: var(--text-h); margin: 0; line-height: 1.2; }
.page-header p { font-size: 0.8rem; color: var(--text-muted); margin: 2px 0 0; }

.section-title { font-family: var(--font-display); font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.16em; color: var(--accent-1); margin: 2rem 0 0.8rem; display: flex; align-items: center; gap: 8px; }
.section-title::after { content: ''; flex: 1; height: 1px; background: var(--border-soft); }

.stat-card { background: var(--bg-card); border: 1px solid var(--border-soft); border-radius: var(--r-lg); padding: 1.1rem 1.3rem; position: relative; overflow: hidden; transition: border-color 0.2s; }
.stat-card:hover { border-color: var(--border-mid); }
.stat-card .label { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.12em; color: var(--text-muted); font-weight: 600; margin-bottom: 4px; }
.stat-card .value { font-family: var(--font-display); font-size: 1.8rem; font-weight: 700; color: var(--text-h); line-height: 1; }
.stat-card .sub { font-size: 0.72rem; color: var(--text-muted); margin-top: 4px; }

.info-card { background: var(--bg-card); border: 1px solid var(--border-soft); border-left: 3px solid; border-radius: var(--r-md); padding: 0.9rem 1.1rem; margin-bottom: 0.7rem; }
.info-card .title { font-family: var(--font-display); font-size: 0.85rem; font-weight: 700; color: var(--text-h); margin-bottom: 0.5rem; }
.info-card .row { display: flex; justify-content: space-between; font-size: 0.78rem; padding: 3px 0; border-bottom: 1px solid var(--border-soft); }
.info-card .row:last-child { border-bottom: none; }
.info-card .row-key { color: var(--text-muted); }
.info-card .row-val { font-weight: 600; color: var(--text-h); }

.pill { display: inline-flex; align-items: center; padding: 3px 10px; border-radius: 20px; font-size: 0.7rem; font-weight: 600; letter-spacing: 0.04em; border: 1px solid; }
.pill-blue   { background: rgba(14,165,233,0.12); color: #38bdf8; border-color: rgba(14,165,233,0.3); }
.pill-green  { background: rgba(16,185,129,0.12); color: #34d399; border-color: rgba(16,185,129,0.3); }
.pill-red    { background: rgba(239,68,68,0.12);  color: #f87171; border-color: rgba(239,68,68,0.3); }
.pill-amber  { background: rgba(245,158,11,0.12); color: #fbbf24; border-color: rgba(245,158,11,0.3); }
.pill-purple { background: rgba(139,92,246,0.12); color: #a78bfa; border-color: rgba(139,92,246,0.3); }

.hero { background: var(--bg-surface); border: 1px solid var(--border-soft); border-radius: var(--r-xl); padding: 2.2rem 2rem 1.8rem; margin-bottom: 1.5rem; position: relative; overflow: hidden; }
.hero::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, var(--accent-1), var(--accent-2), var(--accent-3)); }
.hero h1 { font-family: var(--font-display); font-size: 1.9rem; font-weight: 800; color: var(--text-h); margin: 0 0 0.4rem; line-height: 1.2; }
.hero .sub { font-size: 0.85rem; color: var(--text-muted); line-height: 1.6; max-width: 680px; }
.hero .pills { margin-top: 1rem; display: flex; flex-wrap: wrap; gap: 8px; }

.threshold-row { display: flex; align-items: center; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid var(--border-soft); font-size: 0.8rem; }
.threshold-row:last-child { border-bottom: none; }
.threshold-key { color: var(--text-muted); }
.threshold-val { font-weight: 600; font-family: var(--font-mono); font-size: 0.78rem; }

.sidebar-logo { text-align: center; padding: 1.2rem 0 1.5rem; border-bottom: 1px solid var(--border-soft); margin-bottom: 1rem; }
.sidebar-logo .logo-icon { font-size: 2rem; }
.sidebar-logo .logo-title { font-family: var(--font-display); font-size: 1.1rem; font-weight: 800; color: var(--text-h); margin: 6px 0 2px; }
.sidebar-logo .logo-sub { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.12em; color: var(--text-muted); }

.synthesis-table-wrap { border: 1px solid var(--border-soft); border-radius: var(--r-lg); overflow: hidden; }
.synthesis-table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
.synthesis-table thead th { background: var(--bg-surface); color: var(--text-muted); font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600; padding: 10px 14px; text-align: left; border-bottom: 1px solid var(--border-soft); }
.synthesis-table tbody td { padding: 9px 14px; border-bottom: 1px solid var(--border-soft); color: var(--text-p); vertical-align: middle; }
.synthesis-table tbody tr:last-child td { border-bottom: none; }
.synthesis-table tbody tr:hover td { background: var(--bg-card-hover); }
.synthesis-table .val-critical { font-family: var(--font-mono); font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# CONSTANTES
# ═══════════════════════════════════════════════════════════════════════
DATA_PATH = "data/lstm_final_clean"

MOIS_LABELS = {1:"Janvier",2:"Février",3:"Mars",4:"Avril",5:"Mai",6:"Juin",
               7:"Juillet",8:"Août",9:"Septembre",10:"Octobre",11:"Novembre",12:"Décembre"}
MOIS_SHORT  = {1:"Jan",2:"Fév",3:"Mar",4:"Avr",5:"Mai",6:"Jun",
               7:"Jul",8:"Aoû",9:"Sep",10:"Oct",11:"Nov",12:"Déc"}

SEASON_COLORS = {'Hiver':'#818cf8','Printemps':'#34d399','Été':'#f87171','Automne':'#fb923c'}

DANGER_COLORS = {
    "Calme (<0.5m)":      "#10b981",
    "Faible (0.5–1.5m)":  "#f59e0b",
    "Modéré (1.5–2.5m)":  "#ef4444",
    "Agité (2.5–4m)":     "#8b5cf6",
    "Très agité (>4m)":   "#6d28d9",
}
ALERTE_COLORS = {
    'Calme (< 1 m)':      '#10b981',
    'Vigilance (1–2 m)':  '#f59e0b',
    'Danger (> 2 m)':     '#ef4444',
}

DISTANCE_LABELS = {
    1: "~1 km — Danger baigneurs",
    2: "~5 km — Zone étendue",
    3: "~10 km — Intermédiaire",
    4: "~20 km — Large",
}
DISTANCE_COLORS = {
    "~1 km — Danger baigneurs": "#ef4444",
    "~5 km — Zone étendue":     "#f59e0b",
    "~10 km — Intermédiaire":   "#0ea5e9",
    "~20 km — Large":           "#8b5cf6",
}

# ── Rose des vagues : mapping degrés → label cardinal ──────────────────
DIRS_MAP = {
      0.0: "N",   22.5: "NNE",  45.0: "NE",   67.5: "ENE",
     90.0: "E",  112.5: "ESE", 135.0: "SE",  157.5: "SSE",
    180.0: "S",  202.5: "SSO", 225.0: "SO",  247.5: "OSO",
    270.0: "O",  292.5: "ONO", 315.0: "NO",  337.5: "NNO",
    360.0: "N",
}

PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(12,24,41,0.6)",
    font=dict(color="#94b8cc", family="DM Sans", size=12),
    title_font=dict(color="#f0f9ff", family="Syne", size=14),
    xaxis=dict(gridcolor="rgba(30,90,150,0.2)", linecolor="rgba(30,90,150,0.3)",
               tickcolor="#4a7a96", zerolinecolor="rgba(30,90,150,0.2)"),
    yaxis=dict(gridcolor="rgba(30,90,150,0.2)", linecolor="rgba(30,90,150,0.3)",
               tickcolor="#4a7a96", zerolinecolor="rgba(30,90,150,0.2)"),
    legend=dict(bgcolor="rgba(12,24,41,0.8)", bordercolor="rgba(30,90,150,0.3)", borderwidth=1),
    margin=dict(l=20, r=20, t=50, b=30),
    hoverlabel=dict(bgcolor="#0f2035", bordercolor="rgba(30,90,150,0.5)", font_color="#f0f9ff"),
    colorway=["#0ea5e9","#06b6d4","#10b981","#f59e0b","#ef4444","#8b5cf6","#f97316"],
)

# ═══════════════════════════════════════════════════════════════════════
# DUCKDB — CONNEXION
# ═══════════════════════════════════════════════════════════════════════
data_ok = os.path.exists(DATA_PATH)

@st.cache_resource
def get_db_safe():
    con = duckdb.connect(database=":memory:", read_only=False)
    if not data_ok:
        return con
    parquet_files = []
    if os.path.isdir(DATA_PATH):
        for root, _, files in os.walk(DATA_PATH):
            for f in files:
                if f.endswith(".parquet"):
                    parquet_files.append(os.path.join(root, f))
    elif DATA_PATH.endswith(".parquet"):
        parquet_files = [DATA_PATH]
    else:
        for ext in [".parquet", ""]:
            test = DATA_PATH + ext
            if os.path.exists(test):
                parquet_files = [test]; break
    if not parquet_files:
        return con
    files_sql = ", ".join(f"'{f}'" for f in parquet_files)
    probe = con.execute(f"SELECT * FROM read_parquet([{files_sql}]) LIMIT 1").df()
    cols  = probe.columns.tolist()
    def _col(c, typ="DOUBLE"):
        return f"CAST({c} AS {typ})" if c in cols else f"NULL::{typ}"
    dist_expr = _col("DISTANCE","INTEGER") if "DISTANCE" in cols else "1::INTEGER"
    ws_expr   = _col("wind_speed")
    u10_expr  = _col("u10")
    v10_expr  = _col("v10")
    mwp_expr  = _col("mwp")
    mwd_expr  = _col("mwd")
    try:
        avg_sst = con.execute(f"SELECT AVG(CAST(sst AS DOUBLE)) FROM read_parquet([{files_sql}]) LIMIT 100000").fetchone()[0]
        sst_conv = "CAST(sst AS DOUBLE) - 273.15" if avg_sst and avg_sst > 100 else "CAST(sst AS DOUBLE)"
    except: sst_conv = "NULL::DOUBLE"
    try:
        avg_msl = con.execute(f"SELECT AVG(CAST(msl AS DOUBLE)) FROM read_parquet([{files_sql}]) LIMIT 100000").fetchone()[0]
        msl_conv = "CAST(msl AS DOUBLE) / 100.0" if avg_msl and avg_msl > 10000 else "CAST(msl AS DOUBLE)"
    except: msl_conv = "NULL::DOUBLE"

    con.execute(f"""
        CREATE OR REPLACE VIEW hsv AS
        SELECT
            CAST(NOM_PLAGE  AS VARCHAR)   AS NOM_PLAGE,
            CAST(NOM_WILAYA AS VARCHAR)   AS NOM_WILAYA,
            CAST(DATETIME   AS TIMESTAMP) AS DATETIME,
            CAST(X AS DOUBLE)             AS X,
            CAST(Y AS DOUBLE)             AS Y,
            {dist_expr}                   AS DISTANCE,
            CAST(MESURE AS DOUBLE)        AS MESURE,
            {u10_expr}                    AS u10,
            {v10_expr}                    AS v10,
            {ws_expr}                     AS wind_speed,
            {mwp_expr}                    AS mwp,
            {mwd_expr}                    AS mwd,
            ({sst_conv})                  AS sst,
            ({msl_conv})                  AS msl,
            YEAR(CAST(DATETIME AS TIMESTAMP))      AS YEAR,
            MONTH(CAST(DATETIME AS TIMESTAMP))     AS MONTH,
            DAY(CAST(DATETIME AS TIMESTAMP))       AS DAY,
            HOUR(CAST(DATETIME AS TIMESTAMP))      AS HOUR,
            DAYOFWEEK(CAST(DATETIME AS TIMESTAMP)) AS WEEKDAY,
            CASE MONTH(CAST(DATETIME AS TIMESTAMP))
                WHEN 12 THEN 'Hiver'    WHEN 1 THEN 'Hiver'     WHEN 2  THEN 'Hiver'
                WHEN 3  THEN 'Printemps' WHEN 4 THEN 'Printemps' WHEN 5  THEN 'Printemps'
                WHEN 6  THEN 'Été'      WHEN 7 THEN 'Été'       WHEN 8  THEN 'Été'
                WHEN 9  THEN 'Automne'  WHEN 10 THEN 'Automne'  WHEN 11 THEN 'Automne'
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
            CASE WHEN CAST(MESURE AS DOUBLE) < 1.2
                      AND ({sst_conv}) BETWEEN 16 AND 24
                      AND {mwp_expr} < 8
                 THEN TRUE ELSE FALSE END AS AQUA_OK,
            CASE WHEN ({sst_conv}) BETWEEN 16 AND 26
                      AND CAST(MESURE AS DOUBLE) <= 3.0
                      AND {ws_expr} <= 10.0
                      AND ({msl_conv}) >= 1005.0
                 THEN TRUE ELSE FALSE END AS DESSAL_OK
        FROM read_parquet([{files_sql}])
    """)
    return con

con = get_db_safe()

@st.cache_data(show_spinner=False)
def get_lists():
    if not data_ok: return [], [], []
    wilayas = con.execute("SELECT DISTINCT NOM_WILAYA FROM hsv ORDER BY NOM_WILAYA").df()["NOM_WILAYA"].tolist()
    plages  = con.execute("SELECT DISTINCT NOM_PLAGE  FROM hsv ORDER BY NOM_PLAGE").df()["NOM_PLAGE"].tolist()
    years   = con.execute("SELECT DISTINCT YEAR FROM hsv ORDER BY YEAR").df()["YEAR"].tolist()
    return wilayas, plages, years

all_wilayas, all_plages, all_years = get_lists()

def q(sql: str) -> pd.DataFrame:
    return con.execute(sql).df()

def has_col(col: str) -> bool:
    try:
        r = con.execute(f"SELECT COUNT(*) FROM hsv WHERE {col} IS NOT NULL LIMIT 1").fetchone()
        return r[0] > 0
    except: return False

# ═══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="logo-icon">🌊</div>
        <div class="logo-title">HSV Algérie</div>
        <div class="logo-sub">Côtes · 1985–2023 · ERA5</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.selectbox(
        "Navigation",
        ["🏠 Accueil",
         "📊 Analyse Globale",
         "🏖️ Analyse Été",
         "🏊 Alertes Noyades",
         "💧 Dessalement SWRO",
         "🐟 Aquaculture",
         "📋 Synthèse & Export",
         "🗺️ Carte des Dangers"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**FILTRES TEMPORELS**")
    year_filter  = st.multiselect("Année",  all_years, default=[], placeholder="Toutes...")
    month_filter = st.multiselect("Mois",   list(range(1,13)), format_func=lambda x: MOIS_LABELS[x], default=[], placeholder="Tous...")
    day_filter   = st.multiselect("Jour",   list(range(1,32)), format_func=lambda x: f"{x:02d}", default=[], placeholder="Tous...")
    hour_filter  = st.multiselect("Heure",  list(range(0,24)), format_func=lambda x: f"{x:02d}h00", default=[], placeholder="Toutes...")
    st.markdown("**FILTRES GÉOGRAPHIQUES**")
    wilaya_filter = st.multiselect("Wilaya", all_wilayas, default=[], placeholder="Toutes...")

    if wilaya_filter and data_ok:
        wil_in = ",".join(f"'{w}'" for w in wilaya_filter)
        plages_dispo = q(f"SELECT DISTINCT NOM_PLAGE FROM hsv WHERE NOM_WILAYA IN ({wil_in}) ORDER BY NOM_PLAGE")["NOM_PLAGE"].tolist()
    else:
        plages_dispo = all_plages

    plage_filter = st.multiselect("Plage", plages_dispo, default=[], placeholder="Toutes...")
    nb_sel = len(plage_filter) if plage_filter else len(plages_dispo)
    if data_ok:
        st.info(f"📍 {nb_sel} / {len(plages_dispo)} plage(s)")

    st.markdown("---")
    cols_ok = [c for c in ["sst","mwp","msl","mwd","wind_speed"] if data_ok and has_col(c)]
    if cols_ok:
        st.success(f"✅ Variables ERA5\n{', '.join(c.upper() for c in cols_ok)}")
    elif data_ok:
        st.warning("⚠️ Variables ERA5 non trouvées")

# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════
def where_clause(extra: str = "") -> str:
    conds = []
    if year_filter:   conds.append(f"YEAR IN ({','.join(str(y) for y in year_filter)})")
    if month_filter:  conds.append(f"MONTH IN ({','.join(str(m) for m in month_filter)})")
    if day_filter:    conds.append(f"DAY IN ({','.join(str(d) for d in day_filter)})")
    if hour_filter:   conds.append(f"HOUR IN ({','.join(str(h) for h in hour_filter)})")
    if plage_filter:
        pl_in = ",".join(f"'{p}'" for p in plage_filter)
        conds.append(f"NOM_PLAGE IN ({pl_in})")
    elif wilaya_filter:
        wil_in2 = ",".join(f"'{w}'" for w in wilaya_filter)
        conds.append(f"NOM_WILAYA IN ({wil_in2})")
    if extra: conds.append(extra)
    return ("WHERE " + " AND ".join(conds)) if conds else ""

W = where_clause

def apply_theme(fig):
    fig.update_layout(**PLOTLY_THEME)
    return fig

# ─── FIX : helper _ax() pour axes Plotly (évite NameError) ────────────
def _ax():
    """Retourne les paramètres d'axe cohérents avec le thème sombre."""
    return dict(
        gridcolor="rgba(30,90,150,0.2)",
        linecolor="rgba(30,90,150,0.3)",
        tickcolor="#4a7a96",
        zerolinecolor="rgba(30,90,150,0.2)",
        tickfont=dict(color="#94b8cc"),
    )

def section(icon: str, title: str):
    st.markdown(f'<div class="section-title">{icon} {title}</div>', unsafe_allow_html=True)

def page_header(icon_bg: str, icon: str, title: str, subtitle: str):
    st.markdown(f"""
    <div class="page-header">
        <div class="page-header-icon" style="background:{icon_bg}20;border:1px solid {icon_bg}40;">{icon}</div>
        <div><h1>{title}</h1><p>{subtitle}</p></div>
    </div>
    """, unsafe_allow_html=True)

def show_kpis(wh=""):
    if not data_ok: st.warning("Aucune donnée disponible."); return
    with st.spinner("Calcul des indicateurs..."):
        r = q(f"""
            SELECT COUNT(*) AS total, AVG(MESURE) AS avg_hsv, MAX(MESURE) AS max_hsv,
                   STDDEV(MESURE) AS std_hsv,
                   SUM(CASE WHEN MESURE>=1.5 THEN 1 ELSE 0 END) AS n_15,
                   SUM(CASE WHEN MESURE>=2.5 THEN 1 ELSE 0 END) AS n_25,
                   SUM(CASE WHEN MESURE>=4.0 THEN 1 ELSE 0 END) AS n_40,
                   PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY MESURE) AS p95
            FROM hsv {wh}
        """)
    if r.empty or r["total"].iloc[0] == 0: st.warning("Aucune donnée pour les filtres sélectionnés."); return
    total=r["total"].iloc[0]; avg_h=r["avg_hsv"].iloc[0]; max_h=r["max_hsv"].iloc[0]
    std_h=r["std_hsv"].iloc[0]; n15=r["n_15"].iloc[0]; n25=r["n_25"].iloc[0]; n40=r["n_40"].iloc[0]; p95=r["p95"].iloc[0]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("📦 Mesures",       f"{int(total):,}")
    c2.metric("🌊 HSV Moyenne",   f"{avg_h:.3f} m", f"± {std_h:.3f} m")
    c3.metric("⚠️ HSV Maximum",   f"{max_h:.2f} m")
    c4.metric("📈 Percentile 95", f"{p95:.2f} m")
    c5,c6,c7,c8 = st.columns(4)
    c5.metric("🟡 ≥ 1.5 m", f"{int(n15):,}", f"{n15/total*100:.1f}%")
    c6.metric("🟠 ≥ 2.5 m", f"{int(n25):,}", f"{n25/total*100:.1f}%")
    c7.metric("🔴 ≥ 4.0 m", f"{int(n40):,}", f"{n40/total*100:.1f}%")
    c8.metric("📐 Écart-type",    f"{std_h:.3f} m")


# ═══════════════════════════════════════════════════════════════════════
# PAGE : ACCUEIL
# ═══════════════════════════════════════════════════════════════════════
if page == "🏠 Accueil":
    st.markdown("""
    <div class="hero">
        <h1>Système d'Analyse des Vagues Côtières — Algérie</h1>
        <div class="sub">
            Prévision de la Hauteur Significative des Vagues (HSV) par apprentissage profond LSTM + Transfer Learning.<br>
            Données ERA5 · 60 plages · 13 wilayas côtières · 38 ans d'observations (1985–2023).
        </div>
        <div class="pills">
            <span class="pill pill-red">Alertes noyades</span>
            <span class="pill pill-blue">Dessalement SWRO</span>
            <span class="pill pill-green">Aquaculture marine</span>
            <span class="pill pill-purple">Analyse HSV / ERA5</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    section("📊", "Statistiques globales du dataset")
    if data_ok: show_kpis()
    else: st.error("❌ Dataset introuvable : `data/lstm_final_clean`")

    st.markdown("---")
    section("🎯", "Modules applicatifs")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-card" style="border-left-color:#ef4444;">
            <div class="title" style="color:#f87171;">🏊 Alertes Noyades</div>
            <div class="threshold-row"><span class="threshold-key">Seuil vigilance</span><span class="threshold-val" style="color:#fbbf24;">HSV &gt; 1.0 m</span></div>
            <div class="threshold-row"><span class="threshold-key">Seuil danger</span><span class="threshold-val" style="color:#f87171;">HSV &gt; 2.0 m · MWP &gt; 8 s</span></div>
            <div class="threshold-row"><span class="threshold-key">Vent critique</span><span class="threshold-val" style="color:#f87171;">wind_speed &gt; 10 m/s</span></div>
            <div class="threshold-row"><span class="threshold-key">Saison critique</span><span class="threshold-val">Juin – Août</span></div>
            <div class="threshold-row"><span class="threshold-key">Wilayas à risque</span><span class="threshold-val">Tlemcen · Aïn Témouchent</span></div>
        </div>
        <div class="info-card" style="border-left-color:#0ea5e9;">
            <div class="title" style="color:#38bdf8;">💧 Dessalement SWRO</div>
            <div class="threshold-row"><span class="threshold-key">SST optimale</span><span class="threshold-val" style="color:#38bdf8;">16 °C – 26 °C</span></div>
            <div class="threshold-row"><span class="threshold-key">Vent fort</span><span class="threshold-val" style="color:#fbbf24;">Alerte &gt; 10 m/s (turbidité)</span></div>
            <div class="threshold-row"><span class="threshold-key">Pression critique</span><span class="threshold-val" style="color:#f87171;">MSL &lt; 1005 hPa</span></div>
            <div class="threshold-row"><span class="threshold-key">HSV prise d'eau</span><span class="threshold-val" style="color:#fbbf24;">Alerte &gt; 3.0 m</span></div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="info-card" style="border-left-color:#10b981;">
            <div class="title" style="color:#34d399;">🐟 Aquaculture Marine</div>
            <div class="threshold-row"><span class="threshold-key">HSV favorable</span><span class="threshold-val" style="color:#34d399;">&lt; 1.2 m (sécurité cages)</span></div>
            <div class="threshold-row"><span class="threshold-key">SST favorable</span><span class="threshold-val" style="color:#34d399;">16 °C – 24 °C</span></div>
            <div class="threshold-row"><span class="threshold-key">MWP favorable</span><span class="threshold-val">&lt; 8 s</span></div>
            <div class="threshold-row"><span class="threshold-key">Vent favorable</span><span class="threshold-val" style="color:#34d399;">&lt; 8 m/s (stabilité cages)</span></div>
            <div class="threshold-row"><span class="threshold-key">Direction favorable</span><span class="threshold-val">MWD hors secteur N–NO</span></div>
        </div>
        <div class="info-card" style="border-left-color:#8b5cf6;">
            <div class="title" style="color:#a78bfa;">🌊 Analyse HSV · ERA5</div>
            <div class="threshold-row"><span class="threshold-key">Observations</span><span class="threshold-val" style="color:#a78bfa;">~20 millions (horaires)</span></div>
            <div class="threshold-row"><span class="threshold-key">Plages · Wilayas</span><span class="threshold-val">60 plages · 13 wilayas</span></div>
            <div class="threshold-row"><span class="threshold-key">Période</span><span class="threshold-val">1985 → 2023 (38 ans)</span></div>
            <div class="threshold-row"><span class="threshold-key">Variables</span><span class="threshold-val">HSV · SST · MWP · MWD · MSL · Vent</span></div>
        </div>
        """, unsafe_allow_html=True)

    if data_ok:
        st.markdown("---")
        section("📅", "Évolution annuelle de la HSV")
        @st.cache_data(show_spinner=False)
        def accueil_annual():
            return q("SELECT YEAR, AVG(MESURE) AS avg_hsv, MAX(MESURE) AS max_hsv, STDDEV(MESURE) AS std_hsv FROM hsv GROUP BY YEAR ORDER BY YEAR")
        df_yr = accueil_annual()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_yr["YEAR"], y=df_yr["avg_hsv"]+df_yr["std_hsv"], fill=None, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=df_yr["YEAR"], y=df_yr["avg_hsv"]-df_yr["std_hsv"], fill="tonexty", mode="lines", line=dict(width=0), fillcolor="rgba(14,165,233,0.1)", showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=df_yr["YEAR"], y=df_yr["avg_hsv"], name="HSV Moyenne", mode="lines+markers", line=dict(color="#0ea5e9", width=2.5), marker=dict(size=5, color="#06b6d4")))
        fig.add_trace(go.Scatter(x=df_yr["YEAR"], y=df_yr["max_hsv"], name="HSV Maximum", mode="lines", line=dict(color="#ef4444", width=1.5, dash="dot")))
        fig.add_hline(y=1.5, line_dash="dash", line_color="#f59e0b", annotation_text="Seuil danger 1.5 m", annotation_font_color="#f59e0b", annotation_font_size=10)
        apply_theme(fig)
        fig.update_layout(title="Évolution annuelle de la HSV — Côtes algériennes", xaxis_title="Année", yaxis_title="HSV (m)", height=380)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        section("📋", "Tableau de synthèse — seuils critiques")
        st.markdown("""
        <div class="synthesis-table-wrap">
        <table class="synthesis-table">
          <thead><tr><th>Application</th><th>Variable</th><th>Seuil critique</th><th>Mois à risque</th><th>Wilayas exposées</th></tr></thead>
          <tbody>
            <tr><td><span class="pill pill-red">Noyades</span></td><td>HSV (m)</td><td class="val-critical" style="color:#f87171;">&gt; 2.0 m</td><td>Nov – Fév</td><td>Tlemcen · Aïn Témouchent</td></tr>
            <tr><td><span class="pill pill-red">Noyades</span></td><td>Vent (m/s)</td><td class="val-critical" style="color:#fbbf24;">&gt; 10 m/s</td><td>Nov – Mar</td><td>Tlemcen · Oran</td></tr>
            <tr><td><span class="pill pill-red">Noyades</span></td><td>MWP (s)</td><td class="val-critical" style="color:#fbbf24;">&gt; 8 s</td><td>Nov – Fév</td><td>Tlemcen · Oran</td></tr>
            <tr><td><span class="pill pill-blue">Dessalement</span></td><td>SST (°C)</td><td class="val-critical" style="color:#38bdf8;">&lt; 16 °C ou &gt; 26 °C</td><td>Jan–Mar · Août</td><td>Toute la côte</td></tr>
            <tr><td><span class="pill pill-blue">Dessalement</span></td><td>Vent (m/s)</td><td class="val-critical" style="color:#fbbf24;">&gt; 10 m/s (turbidité)</td><td>Déc – Fév</td><td>Toute la côte</td></tr>
            <tr><td><span class="pill pill-blue">Dessalement</span></td><td>MSL (hPa)</td><td class="val-critical" style="color:#f87171;">&lt; 1005 hPa</td><td>Déc – Fév</td><td>Toute la côte</td></tr>
            <tr><td><span class="pill pill-green">Aquaculture</span></td><td>HSV (m)</td><td class="val-critical" style="color:#34d399;">&gt; 1.2 m</td><td>Nov – Mar</td><td>Tlemcen · Oran</td></tr>
            <tr><td><span class="pill pill-green">Aquaculture</span></td><td>Vent (m/s)</td><td class="val-critical" style="color:#34d399;">&gt; 8 m/s</td><td>Nov – Mar</td><td>Tlemcen · Oran</td></tr>
            <tr><td><span class="pill pill-green">Aquaculture</span></td><td>SST (°C)</td><td class="val-critical" style="color:#34d399;">&lt; 16 °C ou &gt; 24 °C</td><td>Jan–Mar · Août–Sep</td><td>Toute la côte</td></tr>
          </tbody>
        </table>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# PAGE : ANALYSE GLOBALE
# ═══════════════════════════════════════════════════════════════════════
elif page == "📊 Analyse Globale":
    page_header("#0ea5e9","📊","Analyse Globale","Distribution et tendances de la HSV — toutes plages")
    wh = W()
    show_kpis(wh)

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Série Temporelle","🗂️ Distribution","📅 Saisonnalité","🏖️ Par Plage"])

    with tab1:
        section("📈","Évolution annuelle de la HSV")
        with st.spinner():
            df_ann = q(f"SELECT YEAR, AVG(MESURE) AS avg_hsv, MAX(MESURE) AS max_hsv, STDDEV(MESURE) AS std_hsv FROM hsv {wh} GROUP BY YEAR ORDER BY YEAR")
        if not df_ann.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_ann["YEAR"], y=df_ann["avg_hsv"]+df_ann["std_hsv"], fill=None, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=df_ann["YEAR"], y=df_ann["avg_hsv"]-df_ann["std_hsv"], fill="tonexty", mode="lines", line=dict(width=0), fillcolor="rgba(14,165,233,0.1)", showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=df_ann["YEAR"], y=df_ann["avg_hsv"], name="Moyenne", mode="lines+markers", line=dict(color="#0ea5e9", width=2)))
            fig.add_trace(go.Scatter(x=df_ann["YEAR"], y=df_ann["max_hsv"], name="Maximum", mode="lines", line=dict(color="#ef4444", width=1.5, dash="dot")))
            for seuil, color, label in [(1.5,"#f59e0b","Vigilance 1.5 m"),(2.5,"#ef4444","Danger 2.5 m")]:
                fig.add_hline(y=seuil, line_dash="dash", line_color=color, annotation_text=label, annotation_font_color=color, annotation_font_size=10)
            apply_theme(fig); fig.update_layout(title="HSV annuelle", xaxis_title="Année", yaxis_title="HSV (m)", height=380)
            st.plotly_chart(fig, use_container_width=True)

        section("📆","Évolution mensuelle moyenne")
        with st.spinner():
            df_mo = q(f"SELECT MONTH, AVG(MESURE) AS avg_hsv, MAX(MESURE) AS max_hsv, STDDEV(MESURE) AS std_hsv FROM hsv {wh} GROUP BY MONTH ORDER BY MONTH")
        if not df_mo.empty:
            df_mo["MOIS_LABEL"] = df_mo["MONTH"].map(MOIS_SHORT)
            fig2 = go.Figure(go.Bar(x=df_mo["MOIS_LABEL"], y=df_mo["avg_hsv"], name="HSV Moyenne",
                marker_color=df_mo["avg_hsv"].apply(lambda v: "#ef4444" if v>=2 else "#f59e0b" if v>=1 else "#10b981"),
                error_y=dict(type="data", array=df_mo["std_hsv"], visible=True, color="rgba(148,184,204,0.4)")))
            apply_theme(fig2); fig2.update_layout(title="HSV moyenne par mois", xaxis_title="Mois", yaxis_title="HSV (m)", height=340)
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        section("🗂️","Distribution de la HSV")
        col_a, col_b = st.columns(2)
        with col_a:
            with st.spinner():
                df_dist = q(f"SELECT ROUND(MESURE,1) AS h, COUNT(*) AS n FROM hsv {wh} GROUP BY h ORDER BY h")
            if not df_dist.empty:
                fig = go.Figure(go.Bar(x=df_dist["h"], y=df_dist["n"], marker_color="#0ea5e9", marker_line_width=0))
                for s, c in [(1.5,"#f59e0b"),(2.5,"#ef4444")]:
                    fig.add_vline(x=s, line_color=c, line_dash="dash", annotation_text=f"{s} m", annotation_font_color=c)
                apply_theme(fig); fig.update_layout(title="Histogramme HSV", xaxis_title="HSV (m)", yaxis_title="Nombre", height=340)
                st.plotly_chart(fig, use_container_width=True)
        with col_b:
            with st.spinner():
                df_niv = q(f"SELECT NIVEAU, COUNT(*) AS n FROM hsv {wh} GROUP BY NIVEAU")
            if not df_niv.empty:
                ordre = ["Calme (<0.5m)","Faible (0.5–1.5m)","Modéré (1.5–2.5m)","Agité (2.5–4m)","Très agité (>4m)"]
                df_niv["NIVEAU"] = pd.Categorical(df_niv["NIVEAU"], categories=ordre, ordered=True)
                df_niv = df_niv.sort_values("NIVEAU")
                fig = go.Figure(go.Pie(labels=df_niv["NIVEAU"], values=df_niv["n"], hole=0.55,
                    marker_colors=[DANGER_COLORS.get(n,"#666") for n in df_niv["NIVEAU"]]))
                apply_theme(fig); fig.update_layout(title="Répartition par niveau d'agitation", height=340)
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        section("📅","Saisonnalité")
        col1, col2 = st.columns(2)
        with col1:
            with st.spinner():
                df_seas = q(f"SELECT SEASON, AVG(MESURE) AS avg_hsv, COUNT(*) AS n FROM hsv {wh} GROUP BY SEASON")
            if not df_seas.empty:
                ord_s = ['Hiver','Printemps','Été','Automne']
                df_seas["SEASON"] = pd.Categorical(df_seas["SEASON"], categories=ord_s, ordered=True)
                df_seas = df_seas.sort_values("SEASON")
                fig = go.Figure(go.Bar(x=df_seas["SEASON"], y=df_seas["avg_hsv"],
                    marker_color=[SEASON_COLORS.get(s,"#0ea5e9") for s in df_seas["SEASON"]],
                    text=df_seas["avg_hsv"].map(lambda v: f"{v:.2f} m"), textposition="outside"))
                apply_theme(fig); fig.update_layout(title="HSV moyenne par saison", yaxis_title="HSV (m)", height=340)
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            with st.spinner():
                df_hr = q(f"SELECT HOUR, AVG(MESURE) AS avg_hsv FROM hsv {wh} GROUP BY HOUR ORDER BY HOUR")
            if not df_hr.empty:
                fig = go.Figure(go.Scatter(x=df_hr["HOUR"], y=df_hr["avg_hsv"], mode="lines+markers", fill="tozeroy",
                    line=dict(color="#06b6d4", width=2), fillcolor="rgba(6,182,212,0.1)"))
                apply_theme(fig); fig.update_layout(title="Cycle diurne de la HSV", xaxis_title="Heure", yaxis_title="HSV (m)", height=340)
                st.plotly_chart(fig, use_container_width=True)

    with tab4:
        section("🏖️","Classement des plages par HSV moyenne")
        with st.spinner():
            df_pl = q(f"""
                SELECT NOM_PLAGE, NOM_WILAYA,
                       AVG(MESURE) AS avg_hsv, MAX(MESURE) AS max_hsv, STDDEV(MESURE) AS std_hsv,
                       COUNT(*) AS n,
                       SUM(CASE WHEN MESURE>=2 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_danger
                FROM hsv {wh} GROUP BY NOM_PLAGE, NOM_WILAYA ORDER BY avg_hsv DESC LIMIT 30
            """)
        if not df_pl.empty:
            fig = go.Figure(go.Bar(x=df_pl["avg_hsv"], y=df_pl["NOM_PLAGE"], orientation="h",
                marker_color=df_pl["avg_hsv"].apply(lambda v: "#ef4444" if v>=1.5 else "#f59e0b" if v>=1 else "#10b981"),
                text=df_pl["avg_hsv"].map(lambda v: f"{v:.2f} m"), textposition="outside"))
            apply_theme(fig); fig.update_layout(title="Top 30 plages — HSV moyenne", xaxis_title="HSV (m)",
                height=max(400, len(df_pl)*22), yaxis=dict(autorange="reversed", **PLOTLY_THEME["yaxis"]))
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════
# PAGE : ANALYSE ÉTÉ
# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════
elif page == "🏖️ Analyse Été":
    page_header(
        "#f97316", "🏖️", "Analyse Estivale",
        "Juin · Juillet · Août — Risques HSV, vent et direction"
    )

    if not data_ok:
        st.error("❌ Dataset introuvable.")
        st.stop()

    wh_ete = "WHERE MONTH IN (6,7,8)"

    section("📊", "KPIs Estivaux")
    show_kpis(wh_ete)

    with st.spinner():
        r_vent = q(f"""
            SELECT AVG(wind_speed) AS avg_ws, MAX(wind_speed) AS max_ws,
                   SUM(CASE WHEN wind_speed>=10 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_fort,
                   AVG(mwp) AS avg_mwp, AVG(mwd) AS avg_mwd
            FROM hsv {wh_ete}
        """)

    if not r_vent.empty:
        section("💨", "Indicateurs vent et vagues — été")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💨 Vent moyen",    f"{r_vent['avg_ws'].iloc[0]:.2f} m/s")
        c2.metric("🌪️ Vent max",      f"{r_vent['max_ws'].iloc[0]:.1f} m/s")
        c3.metric("⚡ Vent fort ≥10", f"{r_vent['pct_fort'].iloc[0]:.1f}%")
        c4.metric("🌊 MWP moyen",     f"{r_vent['avg_mwp'].iloc[0]:.1f} s")

    # ── Fonction couleur Risk Score (définie avant les tabs pour être accessible partout) ──
    def _risk_color(score):
        if score >= 1.0:   return "#ef4444"   # rouge — très élevé
        elif score >= 0.7: return "#f97316"   # orange — élevé
        elif score >= 0.4: return "#f59e0b"   # jaune — modéré
        else:              return "#22c55e"   # vert — faible

    tab1, tab2, tab3, tab4 = st.tabs(["📅 Évolution", "🌬️ Vent & Direction", "🏆 Wilayas", "🏖️ Plages"])

    # ════════════════════════════════════════════════════════════════════
    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            section("📅", "HSV annuel été")
            df_ye = q(f"""
                SELECT YEAR, AVG(MESURE) AS avg_hsv, MAX(MESURE) AS max_hsv
                FROM hsv {wh_ete} GROUP BY YEAR ORDER BY YEAR
            """)
            if not df_ye.empty:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=df_ye["YEAR"], y=df_ye["avg_hsv"], name="HSV moyen",
                    marker_color="#0ea5e9",
                    text=df_ye["avg_hsv"].map(lambda v: f"{v:.3f}"), textposition="outside"
                ))
                fig.add_trace(go.Scatter(
                    x=df_ye["YEAR"], y=df_ye["max_hsv"], name="HSV max",
                    mode="lines+markers", line=dict(color="#ef4444", width=2)
                ))
                fig.add_hline(y=1.5, line_dash="dash", line_color="#f59e0b",
                    annotation_text="Seuil 1.5 m", annotation_position="top right")
                apply_theme(fig)
                fig.update_layout(
                    title="HSV moyen et maximum par année (été)",
                    xaxis=dict(title="Année", **_ax()),
                    yaxis=dict(title="HSV (m)", **_ax()),
                    legend=dict(orientation="h", y=1.08), height=360
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            section("🗓️", "HSV mensuel été")
            df_me = q(f"""
                SELECT MONTH, AVG(MESURE) AS avg_hsv
                FROM hsv {wh_ete} GROUP BY MONTH ORDER BY MONTH
            """)
            if not df_me.empty:
                df_me["MOIS"] = df_me["MONTH"].map({6: "Juin", 7: "Juillet", 8: "Août"})
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=df_me["MOIS"], y=df_me["avg_hsv"],
                    marker_color=["#0ea5e9", "#38bdf8", "#7dd3fc"],
                    text=df_me["avg_hsv"].map(lambda v: f"{v:.3f} m"), textposition="outside"
                ))
                apply_theme(fig)
                fig.update_layout(
                    title="HSV moyen par mois d'été",
                    xaxis=dict(title="Mois", **_ax()),
                    yaxis=dict(title="HSV (m)", **_ax()),
                    showlegend=False, height=360
                )
                st.plotly_chart(fig, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════
    with tab2:
        section("💨", "Vent été")
        df_ws_ete = q(f"""
            SELECT MONTH, AVG(wind_speed) AS avg_ws, MAX(wind_speed) AS max_ws,
                   AVG(u10) AS avg_u10, AVG(v10) AS avg_v10
            FROM hsv {wh_ete} GROUP BY MONTH ORDER BY MONTH
        """)
        col1, col2 = st.columns(2)

        with col1:
            if not df_ws_ete.empty:
                df_ws_ete["MOIS"] = df_ws_ete["MONTH"].map({6: "Juin", 7: "Juillet", 8: "Août"})
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=df_ws_ete["MOIS"], y=df_ws_ete["avg_ws"], name="Vent moyen (m/s)",
                    marker_color="#06b6d4",
                    text=df_ws_ete["avg_ws"].map(lambda v: f"{v:.1f}"), textposition="outside"
                ))
                fig.add_trace(go.Scatter(
                    x=df_ws_ete["MOIS"], y=df_ws_ete["max_ws"], name="Vent max (m/s)",
                    mode="lines+markers", line=dict(color="#ef4444", width=2)
                ))
                fig.add_hline(y=10, line_dash="dash", line_color="#f59e0b",
                    annotation_text="Seuil fort 10 m/s", annotation_position="top right")
                apply_theme(fig)
                fig.update_layout(
                    title="Vitesse du vent par mois (été)",
                    xaxis=dict(title="Mois", **_ax()),
                    yaxis=dict(title="Vent (m/s)", **_ax()),
                    legend=dict(orientation="h", y=1.08), height=360
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            section("🧭", "Composantes U10 / V10")
            if not df_ws_ete.empty:
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x=df_ws_ete["MOIS"], y=df_ws_ete["avg_u10"],
                    name="U10 — composante zonale", marker_color="#8b5cf6",
                    text=df_ws_ete["avg_u10"].map(lambda v: f"{v:.2f}"), textposition="outside"
                ))
                fig2.add_trace(go.Bar(
                    x=df_ws_ete["MOIS"], y=df_ws_ete["avg_v10"],
                    name="V10 — composante méridionale", marker_color="#06b6d4",
                    text=df_ws_ete["avg_v10"].map(lambda v: f"{v:.2f}"), textposition="outside"
                ))
                apply_theme(fig2)
                fig2.update_layout(
                    title="Composantes U10 / V10 moyennes (été)",
                    xaxis=dict(title="Mois", **_ax()),
                    yaxis=dict(title="Composante (m/s)", **_ax()),
                    barmode="group", legend=dict(orientation="h", y=1.08), height=360
                )
                st.plotly_chart(fig2, use_container_width=True)

        section("🌊", "Rose des vagues (été)")
        df_mwd = q(f"""
            SELECT FLOOR(mwd / 22.5) * 22.5 AS dir_bin, COUNT(*) AS n, AVG(MESURE) AS avg_hsv
            FROM hsv WHERE MONTH IN (6,7,8) AND mwd IS NOT NULL
            GROUP BY dir_bin ORDER BY dir_bin
        """)
        if not df_mwd.empty:
            df_mwd["dir_label"] = df_mwd["dir_bin"].map(
                lambda v: DIRS_MAP.get(round(v / 22.5) * 22.5, f"{v:.0f}°")
            )
            fig3 = go.Figure(go.Barpolar(
                r=df_mwd["n"], theta=df_mwd["dir_bin"], width=22,
                text=df_mwd["dir_label"],
                hovertemplate="<b>Direction : %{text}</b><br>Occurrences : %{r}<br>HSV moy : %{customdata:.3f} m<extra></extra>",
                customdata=df_mwd["avg_hsv"],
                marker=dict(
                    color=df_mwd["avg_hsv"], colorscale="RdYlGn_r",
                    colorbar=dict(title=dict(text="HSV moy (m)", side="right"), thickness=14),
                    showscale=True
                )
            ))
            apply_theme(fig3)
            fig3.update_layout(
                title="Rose des vagues — direction et intensité HSV (été)",
                polar=dict(
                    bgcolor="rgba(4,18,32,0.6)",
                    angularaxis=dict(
                        tickmode="array", tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                        ticktext=["N", "NE", "E", "SE", "S", "SO", "O", "NO"],
                        direction="clockwise", rotation=90,
                        tickfont=dict(color="#94b8cc", size=12), linecolor="#1e3a4f"
                    ),
                    radialaxis=dict(
                        tickfont=dict(color="#94b8cc", size=10),
                        gridcolor="#1e3a4f", linecolor="#1e3a4f",
                        title=dict(text="Nb observations", font=dict(color="#94b8cc"))
                    )
                ),
                height=480
            )
            st.plotly_chart(fig3, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════
    with tab3:
        section("🏆", "Classement wilayas — été (Risk Score)")

        df_rank = q(f"""
            SELECT NOM_WILAYA,
                   ROUND(AVG(MESURE),4)     AS avg_hsv,
                   ROUND(MAX(MESURE),4)     AS max_hsv,
                   ROUND(AVG(wind_speed),4) AS avg_ws,
                   ROUND(AVG(mwp),4)        AS avg_mwp,
                   ROUND(SUM(CASE WHEN MESURE>=2 THEN 1 ELSE 0 END)*100.0/COUNT(*),2) AS pct_danger,
                   COUNT(*) AS total
            FROM hsv {wh_ete}
            GROUP BY NOM_WILAYA
        """)

        if not df_rank.empty:
            # Protection division par zéro sur MWP
            df_rank["avg_mwp"] = df_rank["avg_mwp"].replace(0, None)
            df_rank["avg_mwp"] = df_rank["avg_mwp"].fillna(df_rank["avg_mwp"].median())

            # ── Calcul Risk Score ──
            # RISK = 0.4*HSV_moy + 0.2*HSV_max + 0.2*(%danger/100) + 0.1*vent_moy + 0.1*(1/MWP)
            df_rank["risk_score"] = (
                0.4 * df_rank["avg_hsv"]
              + 0.2 * df_rank["max_hsv"]
              + 0.2 * (df_rank["pct_danger"] / 100)
              + 0.1 * df_rank["avg_ws"]
              + 0.1 * (1 / df_rank["avg_mwp"])
            ).round(4)

            df_rank = df_rank.sort_values("risk_score", ascending=False).head(15).reset_index(drop=True)

            bar_colors_rank = df_rank["risk_score"].apply(_risk_color)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_rank["NOM_WILAYA"],
                y=df_rank["risk_score"],
                name="Risk Score",
                marker_color=bar_colors_rank,
                text=df_rank["risk_score"].map(lambda v: f"{v:.4f}"),
                textposition="outside",
                customdata=df_rank[["avg_hsv", "max_hsv", "avg_ws", "avg_mwp", "pct_danger"]].values,
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "⚠️ Risk Score : %{y:.4f}<br>"
                    "HSV moy : %{customdata[0]:.3f} m<br>"
                    "HSV max : %{customdata[1]:.2f} m<br>"
                    "Vent moy : %{customdata[2]:.2f} m/s<br>"
                    "MWP moy : %{customdata[3]:.1f} s<br>"
                    "% Danger : %{customdata[4]:.1f}%<extra></extra>"
                )
            ))

            fig.add_hline(y=0.4, line_dash="dot",  line_color="#22c55e",
                annotation_text="Faible",  annotation_position="top right")
            fig.add_hline(y=0.7, line_dash="dot",  line_color="#f59e0b",
                annotation_text="Modéré",  annotation_position="top right")
            fig.add_hline(y=1.0, line_dash="dash", line_color="#ef4444",
                annotation_text="Élevé",   annotation_position="top right")

            apply_theme(fig)
            fig.update_layout(
                title="Risk Score par wilaya (été — Top 15)",
                xaxis=dict(title="Wilaya", tickangle=-35, **_ax()),
                yaxis=dict(title="Risk Score", **_ax()),
                showlegend=False, height=420
            )
            st.plotly_chart(fig, use_container_width=True)

            # Légende niveaux de risque
            st.markdown("""
            <div style="display:flex;gap:18px;flex-wrap:wrap;margin:4px 0 12px 0;font-size:0.85rem;">
              <span style="color:#22c55e">🟢 <b>Faible</b> &lt; 1.0</span>
              <span style="color:#f59e0b">🟡 <b>Modéré</b> 1.0 – 1.4</span>
              <span style="color:#ef4444">🔴 <b>élevé</b> ≥ 1.4</span>
            </div>
            """, unsafe_allow_html=True)

            # Tableau récapitulatif
            df_display_rank = df_rank[["NOM_WILAYA", "avg_hsv", "max_hsv", "avg_ws", "avg_mwp", "pct_danger", "risk_score"]].copy()
            df_display_rank.columns = ["Wilaya", "HSV Moy (m)", "HSV Max (m)", "Vent Moy (m/s)", "MWP Moy (s)", "% Danger", "⚠️ Risk Score"]
            st.dataframe(df_display_rank, use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════════════════════════
    with tab4:
        section("🏖️", "Classement plages — été (Risk Score)")

        col_sort, col_n = st.columns([3, 1])
        with col_sort:
            sort_ete = st.selectbox(
                "Trier par",
                ["Risk Score ↓", "HSV Moyenne ↓", "HSV Maximum ↓", "% Danger ↓", "Vent Moyen ↓"],
                key="sort_ete_plage"
            )
        with col_n:
            top_n_ete = st.number_input("Top N", min_value=5, max_value=56, value=20, step=5, key="topn_ete")

        df_pl = q(f"""
            SELECT NOM_PLAGE, NOM_WILAYA,
                   ROUND(AVG(MESURE),4)     AS avg_hsv,
                   ROUND(MAX(MESURE),4)     AS max_hsv,
                   ROUND(AVG(wind_speed),4) AS avg_ws,
                   ROUND(AVG(mwp),4)        AS avg_mwp,
                   ROUND(SUM(CASE WHEN MESURE>=2 THEN 1 ELSE 0 END)*100.0/COUNT(*),2) AS pct_danger,
                   COUNT(*) AS total
            FROM hsv {wh_ete}
            GROUP BY NOM_PLAGE, NOM_WILAYA
        """)

        if not df_pl.empty:
            # Protection division par zéro sur MWP
            df_pl["avg_mwp"] = df_pl["avg_mwp"].replace(0, None)
            df_pl["avg_mwp"] = df_pl["avg_mwp"].fillna(df_pl["avg_mwp"].median())

            # ── Calcul Risk Score ──
            # RISK = 0.4*HSV_moy + 0.2*HSV_max + 0.2*(%danger/100) + 0.1*vent_moy + 0.1*(1/MWP)
            df_pl["risk_score"] = (
                0.4 * df_pl["avg_hsv"]
              + 0.2 * df_pl["max_hsv"]
              + 0.2 * (df_pl["pct_danger"] / 100)
              + 0.1 * df_pl["avg_ws"]
              + 0.1 * (1 / df_pl["avg_mwp"])
            ).round(4)

            # Tri selon choix utilisateur
            sort_col_ete = {
                "Risk Score ↓":  "risk_score",
                "HSV Moyenne ↓": "avg_hsv",
                "HSV Maximum ↓": "max_hsv",
                "% Danger ↓":    "pct_danger",
                "Vent Moyen ↓":  "avg_ws",
            }[sort_ete]

            df_pl = df_pl.sort_values(sort_col_ete, ascending=False).head(top_n_ete).reset_index(drop=True)

            bar_colors_pl = df_pl["risk_score"].apply(_risk_color)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_pl["risk_score"],
                y=df_pl["NOM_PLAGE"],
                orientation="h",
                marker_color=bar_colors_pl,
                text=df_pl["risk_score"].map(lambda v: f"{v:.4f}"),
                textposition="outside",
                customdata=df_pl[["NOM_WILAYA", "avg_hsv", "max_hsv", "avg_ws", "avg_mwp", "pct_danger"]].values,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Wilaya : %{customdata[0]}<br>"
                    "⚠️ Risk Score : %{x:.4f}<br>"
                    "HSV moy : %{customdata[1]:.3f} m<br>"
                    "HSV max : %{customdata[2]:.2f} m<br>"
                    "Vent moy : %{customdata[3]:.2f} m/s<br>"
                    "MWP moy : %{customdata[4]:.1f} s<br>"
                    "% Danger : %{customdata[5]:.1f}%<extra></extra>"
                )
            ))

            fig.add_vline(x=0.4, line_dash="dot",  line_color="#22c55e",
                annotation_text="Faible",  annotation_position="top")
            fig.add_vline(x=0.7, line_dash="dot",  line_color="#f59e0b",
                annotation_text="Modéré",  annotation_position="top")
            fig.add_vline(x=1.0, line_dash="dash", line_color="#ef4444",
                annotation_text="Élevé",   annotation_position="top")

            apply_theme(fig)
            fig.update_layout(
                title=f"Top {top_n_ete} plages — {sort_ete.replace(' ↓', '')} (été)",
                xaxis=dict(title="Risk Score", **_ax()),
                yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
                height=max(380, top_n_ete * 22),
                margin=dict(l=0, r=90, t=50, b=0),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

            # Légende niveaux de risque
            st.markdown("""
            <div style="display:flex;gap:18px;flex-wrap:wrap;margin:4px 0 12px 0;font-size:0.85rem;">
              <span style="color:#22c55e">🟢 <b>Faible</b> &lt; 1.0</span>
              <span style="color:#f59e0b">🟡 <b>Modéré</b> 1.0 – 1.4</span>
              <span style="color:#ef4444">🔴 <b>élevé</b> ≥ 1.4</span>
            </div>
            """, unsafe_allow_html=True)

            # Tableau récapitulatif
            df_display_pl = df_pl[["NOM_PLAGE", "NOM_WILAYA", "avg_hsv", "max_hsv", "avg_ws", "avg_mwp", "pct_danger", "risk_score"]].copy()
            df_display_pl.columns = ["Plage", "Wilaya", "HSV Moy (m)", "HSV Max (m)", "Vent Moy (m/s)", "MWP Moy (s)", "% Danger", "⚠️ Risk Score"]
            st.dataframe(df_display_pl, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════
# PAGE : ALERTES NOYADES
# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════
# PAGE : ALERTES NOYADES
# ═══════════════════════════════════════════════════════════════════════
elif page == "🏊 Alertes Noyades":
    page_header("#ef4444","🏊","Alertes Noyades","HSV · Vent · Direction — seuils critiques de sécurité balnéaire")
    st.markdown("""
    <div class="info-card" style="border-left-color:#ef4444; background:rgba(239,68,68,0.06);">
        <div class="title" style="color:#f87171;">Seuils de référence — sécurité baignade</div>
        <div class="threshold-row"><span class="threshold-key">🟢 Calme</span><span class="threshold-val" style="color:#34d399;">HSV &lt; 1.0 m · Vent &lt; 5 m/s — Baignade sûre</span></div>
        <div class="threshold-row"><span class="threshold-key">🟡 Vigilance</span><span class="threshold-val" style="color:#fbbf24;">HSV 1–2 m · Vent 5–10 m/s — Prudence recommandée</span></div>
        <div class="threshold-row"><span class="threshold-key">🔴 Danger</span><span class="threshold-val" style="color:#f87171;">HSV &gt; 2 m ou Vent &gt; 10 m/s ou MWP &gt; 8 s</span></div>
        <div class="threshold-row"><span class="threshold-key">📐 Risk Score</span><span class="threshold-val" style="color:#a78bfa;">(HSV/3)×0.5 + (Vent/15)×0.3 + (MWP/10)×0.2</span></div>
    </div>
    """, unsafe_allow_html=True)

    wh = W()
    show_kpis(wh)

    with st.spinner():
        r_ws = q(f"""
            SELECT AVG(wind_speed) AS avg_ws, MAX(wind_speed) AS max_ws,
                   SUM(CASE WHEN wind_speed>=10 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_fort,
                   SUM(CASE WHEN wind_speed>=15 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_tempete
            FROM hsv {wh}
        """)
    if not r_ws.empty:
        section("💨", "Indicateurs vent")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💨 Vent Moyen",       f"{r_ws['avg_ws'].iloc[0]:.2f} m/s")
        c2.metric("🌪️ Vent Maximum",     f"{r_ws['max_ws'].iloc[0]:.1f} m/s")
        c3.metric("⚡ % Vent Fort ≥10",  f"{r_ws['pct_fort'].iloc[0]:.1f}%")
        c4.metric("🌩️ % Tempête ≥15",   f"{r_ws['pct_tempete'].iloc[0]:.1f}%")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Alertes","🌬️ Vent & MWD","📡 Distance","🏖️ Par Plage"])

    # ─────────────────────────────────────────────────────────────────
    # TAB 1 — ALERTES
    # ─────────────────────────────────────────────────────────────────
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            section("📊","Répartition globale des niveaux d'alerte")
            with st.spinner():
                df_al = q(f"SELECT ALERTE, COUNT(*) AS n FROM hsv {wh} GROUP BY ALERTE")
            if not df_al.empty:
                ord_al = ['Calme (< 1 m)','Vigilance (1–2 m)','Danger (> 2 m)']
                df_al["ALERTE"] = pd.Categorical(df_al["ALERTE"], categories=ord_al, ordered=True)
                df_al = df_al.sort_values("ALERTE")
                total_al = df_al["n"].sum()
                for _, row in df_al.iterrows():
                    pct   = row["n"] / total_al * 100
                    color = ALERTE_COLORS.get(row["ALERTE"], "#666")
                    st.markdown(f"""
                    <div style="background:var(--bg-card);border:1px solid var(--border-soft);
                         border-left:3px solid {color};border-radius:8px;padding:10px 14px;
                         margin-bottom:8px;display:flex;justify-content:space-between;">
                        <span>{row['ALERTE']}</span>
                        <b style="color:{color};">{pct:.1f}%</b>
                    </div>""", unsafe_allow_html=True)

        with col2:
            section("📈","Évolution annuelle % danger")
            with st.spinner():
                df_ann_al = q(f"""
                    SELECT YEAR,
                           SUM(CASE WHEN MESURE>=2 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_danger,
                           SUM(CASE WHEN MESURE>=1 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_vig,
                           SUM(CASE WHEN wind_speed>=10 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_vent_fort
                    FROM hsv {wh} GROUP BY YEAR ORDER BY YEAR
                """)
            if not df_ann_al.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_ann_al["YEAR"], y=df_ann_al["pct_vig"],
                    name="Vigilance HSV", line=dict(color="#f59e0b")))
                fig.add_trace(go.Scatter(x=df_ann_al["YEAR"], y=df_ann_al["pct_danger"],
                    name="Danger HSV", line=dict(color="#ef4444"),
                    fill="tozeroy", fillcolor="rgba(239,68,68,0.08)"))
                fig.add_trace(go.Scatter(x=df_ann_al["YEAR"], y=df_ann_al["pct_vent_fort"],
                    name="Vent ≥10 m/s", line=dict(color="#06b6d4", dash="dot")))
                apply_theme(fig)
                fig.update_layout(title="% annuel alertes HSV + vent",
                    xaxis_title="Année", yaxis_title="%", height=360)
                st.plotly_chart(fig, use_container_width=True)

        if has_col("mwp"):
            section("⏱️","Période des vagues (MWP) — corrélé au risque de noyade")
            with st.spinner():
                df_mwp = q(f"""
                    SELECT ROUND(mwp,0) AS mwp_r, COUNT(*) AS n
                    FROM hsv {wh}
                    WHERE mwp IS NOT NULL
                    GROUP BY mwp_r ORDER BY mwp_r
                """)
            if not df_mwp.empty:
                fig = go.Figure(go.Bar(
                    x=df_mwp["mwp_r"], y=df_mwp["n"],
                    marker_color=df_mwp["mwp_r"].apply(
                        lambda v: "#ef4444" if v > 8 else "#f59e0b" if v > 6 else "#10b981"
                    )
                ))
                fig.add_vline(x=8, line_dash="dash", line_color="#ef4444", annotation_text="Danger >8s")
                apply_theme(fig)
                fig.update_layout(title="Distribution MWP",
                    xaxis_title="MWP (s)", yaxis_title="Nombre", height=300)
                st.plotly_chart(fig, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────
    # TAB 2 — VENT & MWD
    # ─────────────────────────────────────────────────────────────────
    with tab2:
        section("💨", "Vent mensuel et corrélation avec HSV")
        col1, col2 = st.columns(2)
        with col1:
            with st.spinner():
                df_ws_m = q(f"""
                    SELECT MONTH,
                           AVG(wind_speed) AS avg_ws, MAX(wind_speed) AS max_ws,
                           AVG(u10) AS avg_u10, AVG(v10) AS avg_v10,
                           SUM(CASE WHEN wind_speed>=10 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_fort,
                           AVG(MESURE) AS avg_hsv
                    FROM hsv {wh} GROUP BY MONTH ORDER BY MONTH
                """)
            if not df_ws_m.empty:
                df_ws_m["MOIS_L"] = df_ws_m["MONTH"].map(MOIS_SHORT)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_ws_m["MOIS_L"], y=df_ws_m["avg_ws"],
                    name="Vent moy (m/s)", mode="lines+markers",
                    line=dict(color="#06b6d4", width=2)))
                fig.add_trace(go.Bar(x=df_ws_m["MOIS_L"], y=df_ws_m["pct_fort"],
                    name="% vent ≥10 m/s", marker_color="rgba(239,68,68,0.5)", yaxis="y2"))
                fig.add_hline(y=10, line_dash="dash", line_color="#f59e0b",
                    annotation_text="Seuil alerte 10 m/s")
                apply_theme(fig)
                fig.update_layout(
                    title="Vent mensuel — vitesse et % fort",
                    yaxis=dict(title="Vent moy (m/s)", **PLOTLY_THEME["yaxis"]),
                    yaxis2=dict(title="% vent ≥10 m/s", overlaying="y", side="right",
                                gridcolor="rgba(0,0,0,0)", tickcolor="#4a7a96",
                                tickfont=dict(color="#94b8cc")),
                    height=360
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            section("🧮", "Composantes U10 / V10 mensuelles")
            if not df_ws_m.empty:
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(x=df_ws_m["MOIS_L"], y=df_ws_m["avg_u10"],
                    name="U10 (zonal)", marker_color="#0ea5e9"))
                fig2.add_trace(go.Bar(x=df_ws_m["MOIS_L"], y=df_ws_m["avg_v10"],
                    name="V10 (méridional)", marker_color="#8b5cf6"))
                fig2.add_hline(y=0, line_color="rgba(148,184,204,0.3)")
                apply_theme(fig2)
                fig2.update_layout(barmode="group", title="U10 / V10 par mois",
                    yaxis_title="m/s", height=340)
                st.plotly_chart(fig2, use_container_width=True)

        section("🧭", "Rose des vagues — Direction MWD et niveau de danger")
        col3, col4 = st.columns(2)
        with col3:
            with st.spinner():
                df_mwd = q(f"""
                    SELECT FLOOR(mwd/22.5)*22.5 AS dir_bin,
                           COUNT(*) AS n,
                           AVG(MESURE) AS avg_hsv,
                           AVG(wind_speed) AS avg_ws
                    FROM hsv {wh}
                    WHERE mwd IS NOT NULL
                    GROUP BY dir_bin ORDER BY dir_bin
                """)
            if not df_mwd.empty:
                fig3 = go.Figure(go.Barpolar(
                    r=df_mwd["n"], theta=df_mwd["dir_bin"], width=22,
                    marker_color=df_mwd["avg_hsv"],
                    marker_colorscale=[[0,"#10b981"],[0.5,"#f59e0b"],[1,"#ef4444"]],
                    marker_showscale=True,
                    marker_colorbar=dict(title="HSV moy (m)", tickfont=dict(color="#94b8cc"))
                ))
                apply_theme(fig3)
                fig3.update_layout(
                    title="Rose des vagues — colorée par HSV",
                    polar=dict(
                        bgcolor="rgba(12,24,41,0.8)",
                        angularaxis=dict(
                            tickmode="array",
                            tickvals=[0,45,90,135,180,225,270,315],
                            ticktext=["N","NE","E","SE","S","SO","O","NO"],
                            direction="clockwise", rotation=90,
                            gridcolor="rgba(30,90,150,0.3)",
                            tickfont=dict(color="#94b8cc")
                        ),
                        radialaxis=dict(gridcolor="rgba(30,90,150,0.2)",
                                        tickfont=dict(color="#94b8cc"))
                    ), height=400
                )
                st.plotly_chart(fig3, use_container_width=True)

        with col4:
            section("📊", "HSV et vent par secteur directionnel")
            if not df_mwd.empty:
                df_mwd["sector"] = pd.cut(df_mwd["dir_bin"],
                    bins=[0,90,180,270,360],
                    labels=["N–E (0–90°)","E–S (90–180°)","S–O (180–270°)","O–N (270–360°)"],
                    include_lowest=True)
                df_sec = df_mwd.groupby("sector", observed=True).agg(
                    avg_hsv=("avg_hsv","mean"),
                    avg_ws=("avg_ws","mean"),
                    n=("n","sum")
                ).reset_index()
                fig4 = go.Figure()
                fig4.add_trace(go.Bar(
                    x=df_sec["sector"].astype(str), y=df_sec["avg_hsv"],
                    name="HSV moy (m)",
                    marker_color=["#0ea5e9","#06b6d4","#8b5cf6","#f59e0b"]
                ))
                fig4.add_trace(go.Scatter(
                    x=df_sec["sector"].astype(str), y=df_sec["avg_ws"],
                    name="Vent moy (m/s)", mode="lines+markers",
                    line=dict(color="#ef4444", width=2), yaxis="y2"
                ))
                apply_theme(fig4)
                fig4.update_layout(
                    title="HSV et vent par secteur directionnel",
                    yaxis=dict(title="HSV (m)", **PLOTLY_THEME["yaxis"]),
                    yaxis2=dict(title="Vent (m/s)", overlaying="y", side="right",
                                gridcolor="rgba(0,0,0,0)", tickcolor="#4a7a96",
                                tickfont=dict(color="#94b8cc")),
                    height=360
                )
                st.plotly_chart(fig4, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────
    # TAB 3 — DISTANCE  ← MODIFIÉ : risk_score basé sur MESURE
    # ─────────────────────────────────────────────────────────────────
    with tab3:
        section("📡", "Analyse par distance à la côte — zones de danger")
        st.markdown("""
        <div class="info-card" style="border-left-color:#ef4444;">
            <div class="title" style="color:#f87171;">Zones de risque par distance</div>
            <div class="threshold-row"><span class="threshold-key" style="color:#ef4444;">● Dist. 1 — ~1 km</span><span class="threshold-val">Frontière terre-mer · Danger direct baigneurs</span></div>
            <div class="threshold-row"><span class="threshold-key" style="color:#f59e0b;">● Dist. 2 — ~5 km</span><span class="threshold-val">Zone de baignade étendue</span></div>
            <div class="threshold-row"><span class="threshold-key" style="color:#0ea5e9;">● Dist. 3 — ~10 km</span><span class="threshold-val">Zone intermédiaire</span></div>
            <div class="threshold-row"><span class="threshold-key" style="color:#8b5cf6;">● Dist. 4 — ~20 km</span><span class="threshold-val">Conditions du large</span></div>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner():
            df_dist_al = q(f"""
                SELECT DISTANCE,
                       AVG(MESURE)      AS avg_hsv,  MAX(MESURE)      AS max_hsv,
                       AVG(wind_speed)  AS avg_ws,   MAX(wind_speed)  AS max_ws,
                       AVG(mwp)         AS avg_mwp,
                       SUM(CASE WHEN MESURE>=2 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_danger,
                       SUM(CASE WHEN wind_speed>=10 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_vent_fort,
                       (
                           AVG(MESURE)/3 * 0.5 +
                           AVG(wind_speed)/15 * 0.3 +
                           AVG(mwp)/10 * 0.2
                       ) AS risk_score,
                       COUNT(*) AS n
                FROM hsv {wh} GROUP BY DISTANCE ORDER BY DISTANCE
            """)

        if not df_dist_al.empty:
            df_dist_al["DIST_LABEL"] = df_dist_al["DISTANCE"].map(DISTANCE_LABELS)
            dist_colors_list = [list(DISTANCE_COLORS.values())[i % 4]
                                for i in range(len(df_dist_al))]

            # ── Stat cards avec risk_score ──────────────────────────
            cols_d = st.columns(len(df_dist_al))
            for i, (_, row) in enumerate(df_dist_al.iterrows()):
                c = list(DISTANCE_COLORS.values())[i % 4]
                rs = row["risk_score"] if row["risk_score"] is not None else 0.0
                rs_color = "#ef4444" if rs > 0.5 else "#f59e0b" if rs > 0.25 else "#10b981"
                rs_label = "🔴 Danger" if rs > 0.5 else "🟡 Vigilance" if rs > 0.25 else "🟢 Calme"
                with cols_d[i]:
                    st.markdown(f"""
                    <div class="stat-card" style="border-top:2px solid {c};">
                        <div class="label">{row['DIST_LABEL']}</div>
                        <div class="value" style="color:{c};font-size:1.3rem;">{row['avg_hsv']:.2f} m</div>
                        <div class="sub">% Danger: {row['pct_danger']:.1f}%</div>
                        <div class="sub">Vent moy: {row['avg_ws']:.1f} m/s · % fort: {row['pct_vent_fort']:.1f}%</div>
                        <div class="sub">MWP moy: {row['avg_mwp']:.1f} s</div>
                        <div class="sub" style="color:{rs_color};">⚠️ Risk Score: {rs:.3f} — {rs_label}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # ── Graphiques HSV & vent ───────────────────────────────
            col_a, col_b = st.columns(2)
            with col_a:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=df_dist_al["DIST_LABEL"], y=df_dist_al["avg_hsv"],
                    name="HSV moy (m)", marker_color=dist_colors_list,
                    text=df_dist_al["avg_hsv"].map(lambda v: f"{v:.2f}"),
                    textposition="outside"
                ))
                fig.add_trace(go.Scatter(
                    x=df_dist_al["DIST_LABEL"], y=df_dist_al["pct_danger"],
                    name="% Danger", mode="lines+markers",
                    line=dict(color="#ef4444", width=2), yaxis="y2"
                ))
                apply_theme(fig)
                fig.update_layout(
                    title="HSV et % danger par zone",
                    yaxis=dict(title="HSV (m)", **PLOTLY_THEME["yaxis"]),
                    yaxis2=dict(title="% Danger", overlaying="y", side="right",
                                gridcolor="rgba(0,0,0,0)", tickcolor="#4a7a96",
                                tickfont=dict(color="#94b8cc")),
                    height=340
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_b:
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x=df_dist_al["DIST_LABEL"], y=df_dist_al["avg_ws"],
                    name="Vent moy (m/s)", marker_color=dist_colors_list,
                    text=df_dist_al["avg_ws"].map(lambda v: f"{v:.1f}"),
                    textposition="outside"
                ))
                fig2.add_trace(go.Scatter(
                    x=df_dist_al["DIST_LABEL"], y=df_dist_al["pct_vent_fort"],
                    name="% Vent fort", mode="lines+markers",
                    line=dict(color="#f59e0b", width=2), yaxis="y2"
                ))
                apply_theme(fig2)
                fig2.update_layout(
                    title="Vent et % vent fort par zone",
                    yaxis=dict(title="Vent moy (m/s)", **PLOTLY_THEME["yaxis"]),
                    yaxis2=dict(title="% Vent fort", overlaying="y", side="right",
                                gridcolor="rgba(0,0,0,0)", tickcolor="#4a7a96",
                                tickfont=dict(color="#94b8cc")),
                    height=340
                )
                st.plotly_chart(fig2, use_container_width=True)

            # ── Graphique Risk Score par zone de distance ───────────
            section("⚠️", "Risk Score Noyades par zone de distance")
            df_dist_al["rs_clean"] = df_dist_al["risk_score"].fillna(0.0)
            fig_risk = go.Figure(go.Bar(
                x=df_dist_al["DIST_LABEL"],
                y=df_dist_al["rs_clean"],
                marker_color=df_dist_al["rs_clean"].apply(
                    lambda v: "#ef4444" if v > 0.5 else "#f59e0b" if v > 0.25 else "#10b981"
                ),
                text=df_dist_al["rs_clean"].map(lambda v: f"{v:.3f}"),
                textposition="outside"
            ))
            fig_risk.add_hline(y=0.5,  line_dash="dash", line_color="#ef4444",
                annotation_text="Seuil Danger 0.5")
            fig_risk.add_hline(y=0.25, line_dash="dash", line_color="#f59e0b",
                annotation_text="Seuil Vigilance 0.25")
            apply_theme(fig_risk)
            fig_risk.update_layout(
                title="Risk Score Noyades par zone de distance — (HSV/3)×0.5 + (Vent/15)×0.3 + (MWP/10)×0.2",
                yaxis_title="Risk Score",
                height=340
            )
            st.plotly_chart(fig_risk, use_container_width=True)

            # ── HSV mensuelle par zone ──────────────────────────────
            section("📅", "HSV mensuelle par zone de distance")
            with st.spinner():
                df_dist_mo = q(f"""
                    SELECT DISTANCE, MONTH,
                           AVG(MESURE) AS avg_hsv,
                           AVG(wind_speed) AS avg_ws
                    FROM hsv {wh}
                    GROUP BY DISTANCE, MONTH ORDER BY DISTANCE, MONTH
                """)
            if not df_dist_mo.empty:
                df_dist_mo["DIST_LABEL"] = df_dist_mo["DISTANCE"].map(DISTANCE_LABELS)
                df_dist_mo["MOIS_L"]     = df_dist_mo["MONTH"].map(MOIS_SHORT)
                fig3 = px.line(df_dist_mo, x="MOIS_L", y="avg_hsv", color="DIST_LABEL",
                    color_discrete_map=DISTANCE_COLORS, markers=True,
                    labels={"avg_hsv":"HSV (m)","MOIS_L":"Mois","DIST_LABEL":"Zone"})
                fig3.add_hline(y=2.0, line_dash="dash", line_color="#ef4444",
                    annotation_text="Danger 2.0 m")
                apply_theme(fig3)
                fig3.update_layout(title="HSV mensuelle par zone de distance", height=340)
                st.plotly_chart(fig3, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────
    # TAB 4 — PAR PLAGE  ← MODIFIÉ : risk_score basé sur MESURE (pas hsv_avg)
    # ─────────────────────────────────────────────────────────────────
    with tab4:
        section("🏖️","Plages dangereuses — Risk Score Noyades")
        with st.spinner():
            df_pl_al = q(f"""
                SELECT NOM_PLAGE, NOM_WILAYA,
                       AVG(MESURE)      AS avg_hsv,  MAX(MESURE)      AS max_hsv,
                       AVG(wind_speed)  AS avg_ws,   MAX(wind_speed)  AS max_ws,
                       AVG(mwp)         AS avg_mwp,  MAX(mwp)         AS max_mwp,
                       AVG(mwd)         AS avg_mwd,
                       SUM(CASE WHEN MESURE>=2 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_danger,
                       SUM(CASE WHEN wind_speed>=10 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_vent_fort
                FROM hsv {wh}
                GROUP BY NOM_PLAGE, NOM_WILAYA
                ORDER BY pct_danger DESC
                LIMIT 30
            """)

        if not df_pl_al.empty:
            # ── Remplacement des NaN MWP par la médiane ─────────────
            df_pl_al["avg_mwp"] = df_pl_al["avg_mwp"].fillna(df_pl_al["avg_mwp"].median())

            # ── Risk Score basé sur MESURE (avg_hsv = AVG(MESURE)) ──
            df_pl_al["risk_score"] = (
                (df_pl_al["avg_hsv"] / 3.0)  * 0.5
              + (df_pl_al["avg_ws"]  / 15.0) * 0.3
              + (df_pl_al["avg_mwp"] / 10.0) * 0.2
            ).round(4)

            col_a, col_b = st.columns(2)
            with col_a:
                fig = px.scatter(df_pl_al, x="avg_hsv", y="avg_ws",
                    size="risk_score", color="NOM_WILAYA", hover_name="NOM_PLAGE",
                    labels={"avg_hsv":"HSV moy (m)","avg_ws":"Vent moy (m/s)"})
                fig.add_vline(x=2.0, line_dash="dash", line_color="#ef4444",
                    annotation_text="HSV 2m")
                fig.add_hline(y=10,  line_dash="dash", line_color="#f59e0b",
                    annotation_text="Vent 10 m/s")
                apply_theme(fig)
                fig.update_layout(title="Risque HSV × Vent par plage", height=420)
                st.plotly_chart(fig, use_container_width=True)

            with col_b:
                fig2 = px.scatter(df_pl_al, x="avg_mwp", y="avg_mwd",
                    size="avg_hsv", color="NOM_WILAYA", hover_name="NOM_PLAGE",
                    labels={"avg_mwp":"MWP moy (s)","avg_mwd":"Direction MWD (°)"})
                fig2.add_vline(x=8, line_dash="dash", line_color="#ef4444",
                    annotation_text="MWP 8s")
                apply_theme(fig2)
                fig2.update_layout(title="MWP × Direction MWD par plage", height=420)
                st.plotly_chart(fig2, use_container_width=True)

            # ── Tableau final ────────────────────────────────────────
            df_show = df_pl_al[[
                "NOM_PLAGE","NOM_WILAYA",
                "avg_hsv","max_hsv",
                "avg_ws","max_ws",
                "avg_mwp","max_mwp","avg_mwd",
                "pct_danger","pct_vent_fort","risk_score"
            ]].round(3)
            df_show.columns = [
                "Plage","Wilaya",
                "HSV Moy","HSV Max",
                "Vent Moy","Vent Max",
                "MWP Moy","MWP Max","MWD Moy",
                "% Danger","% Vent Fort","⚠️ Risk Score"
            ]
            st.dataframe(df_show, use_container_width=True, hide_index=True)
# ═══════════════════════════════════════════════════════════════════════
# PAGE : DESSALEMENT SWRO
# ═══════════════════════════════════════════════════════════════════════
elif page == "💧 Dessalement SWRO":
    page_header("#0ea5e9","💧","Dessalement SWRO","Conditions marines pour les stations d'osmose inverse · SST · Vent · MSL")

    st.markdown("""
    <div class="info-card" style="border-left-color:#0ea5e9; background:rgba(14,165,233,0.06);">
        <div class="title" style="color:#38bdf8;">Paramètres clés — dessalement par osmose inverse</div>
        <div class="threshold-row"><span class="threshold-key">SST optimale</span><span class="threshold-val" style="color:#38bdf8;">16 °C – 26 °C (efficacité membranaire)</span></div>
        <div class="threshold-row"><span class="threshold-key">HSV prise d'eau</span><span class="threshold-val" style="color:#fbbf24;">Alerte si HSV &gt; 3.0 m (colmatage)</span></div>
        <div class="threshold-row"><span class="threshold-key">Vent fort</span><span class="threshold-val" style="color:#f87171;">Alerte si vent &gt; 10 m/s (turbidité et agitation)</span></div>
        <div class="threshold-row"><span class="threshold-key">Pression atmosphérique</span><span class="threshold-val" style="color:#f87171;">Critique si MSL &lt; 1005 hPa (tempête)</span></div>
        <div class="threshold-row"><span class="threshold-key">📐 Risk Score</span><span class="threshold-val" style="color:#a78bfa;">(HSV/5)×0.35 + (vent/15)×0.25 + (1-SST_ok)×0.25 + (MSL_risk)×0.15</span></div>
    </div>
    """, unsafe_allow_html=True)

    wh = W()
    show_kpis(wh)

    with st.spinner():
        r_ws2 = q(f"""
            SELECT AVG(wind_speed) AS avg_ws, MAX(wind_speed) AS max_ws,
                   SUM(CASE WHEN wind_speed>=10 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_fort
            FROM hsv {wh}
        """)
    if not r_ws2.empty:
        section("💨", "Indicateurs vent — impact dessalement")
        c1,c2,c3 = st.columns(3)
        c1.metric("💨 Vent Moyen",           f"{r_ws2['avg_ws'].iloc[0]:.2f} m/s")
        c2.metric("🌪️ Vent Maximum",         f"{r_ws2['max_ws'].iloc[0]:.1f} m/s")
        c3.metric("⚡ % Vent Fort ≥10 m/s",  f"{r_ws2['pct_fort'].iloc[0]:.1f}%")

    tab1, tab2, tab3, tab4 = st.tabs(["🌡️ SST","📊 Pression MSL","🌬️ Vent & Direction","⚙️ Conditions Opérationnelles"])

    with tab1:
        section("🌡️","Température de surface de la mer (SST)")
        col1, col2 = st.columns(2)
        with col1:
            with st.spinner():
                df_sst_m = q(f"""
                    SELECT MONTH, AVG(sst) AS avg_sst, MIN(sst) AS min_sst,
                           MAX(sst) AS max_sst, STDDEV(sst) AS std_sst
                    FROM hsv {wh} WHERE sst IS NOT NULL GROUP BY MONTH ORDER BY MONTH
                """)
            if not df_sst_m.empty:
                df_sst_m["MOIS_L"] = df_sst_m["MONTH"].map(MOIS_SHORT)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_sst_m["MOIS_L"], y=df_sst_m["avg_sst"]+df_sst_m["std_sst"], fill=None, mode="lines", line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=df_sst_m["MOIS_L"], y=df_sst_m["avg_sst"]-df_sst_m["std_sst"], fill="tonexty", mode="lines", line=dict(width=0), fillcolor="rgba(6,182,212,0.1)", showlegend=False))
                fig.add_trace(go.Scatter(x=df_sst_m["MOIS_L"], y=df_sst_m["avg_sst"], name="SST moy", mode="lines+markers", line=dict(color="#06b6d4", width=2)))
                for val, color, label in [(16,"#f59e0b","Seuil min 16°C"),(26,"#ef4444","Seuil max 26°C")]:
                    fig.add_hline(y=val, line_dash="dash", line_color=color, annotation_text=label, annotation_font_color=color)
                apply_theme(fig); fig.update_layout(title="Cycle saisonnier SST", yaxis_title="SST (°C)", height=340)
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            with st.spinner():
                df_sst_oor = q(f"""
                    SELECT MONTH,
                           SUM(CASE WHEN sst<16 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_froid,
                           SUM(CASE WHEN sst>26 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_chaud
                    FROM hsv {wh} WHERE sst IS NOT NULL GROUP BY MONTH ORDER BY MONTH
                """)
            if not df_sst_oor.empty:
                df_sst_oor["MOIS_L"] = df_sst_oor["MONTH"].map(MOIS_SHORT)
                fig = go.Figure()
                fig.add_trace(go.Bar(x=df_sst_oor["MOIS_L"], y=df_sst_oor["pct_froid"], name="SST < 16°C", marker_color="#818cf8"))
                fig.add_trace(go.Bar(x=df_sst_oor["MOIS_L"], y=df_sst_oor["pct_chaud"], name="SST > 26°C", marker_color="#ef4444"))
                apply_theme(fig); fig.update_layout(barmode="stack", title="% hors plage optimale SST", yaxis_title="%", height=340)
                st.plotly_chart(fig, use_container_width=True)

        with st.spinner():
            df_sst_yr = q(f"SELECT YEAR, AVG(sst) AS avg_sst FROM hsv {wh} WHERE sst IS NOT NULL GROUP BY YEAR ORDER BY YEAR")
        if not df_sst_yr.empty:
            z = np.polyfit(df_sst_yr["YEAR"], df_sst_yr["avg_sst"], 1)
            p_fn = np.poly1d(z)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_sst_yr["YEAR"], y=df_sst_yr["avg_sst"], mode="lines+markers", line=dict(color="#0ea5e9", width=2)))
            fig.add_trace(go.Scatter(x=df_sst_yr["YEAR"], y=p_fn(df_sst_yr["YEAR"]), mode="lines", name="Tendance", line=dict(color="#f59e0b", dash="dash")))
            apply_theme(fig); fig.update_layout(title="Tendance annuelle SST", xaxis_title="Année", yaxis_title="SST (°C)", height=320)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        section("📊","Pression atmosphérique (MSL)")
        col1, col2 = st.columns(2)
        with col1:
            with st.spinner():
                df_msl_m = q(f"SELECT MONTH, AVG(msl) AS avg_msl, MIN(msl) AS min_msl FROM hsv {wh} WHERE msl IS NOT NULL GROUP BY MONTH ORDER BY MONTH")
            if not df_msl_m.empty:
                df_msl_m["MOIS_L"] = df_msl_m["MONTH"].map(MOIS_SHORT)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_msl_m["MOIS_L"], y=df_msl_m["avg_msl"], name="MSL moy", mode="lines+markers", line=dict(color="#0ea5e9", width=2)))
                fig.add_trace(go.Scatter(x=df_msl_m["MOIS_L"], y=df_msl_m["min_msl"], name="MSL min", mode="lines", line=dict(color="#ef4444", dash="dot", width=1.5)))
                fig.add_hline(y=1005, line_dash="dash", line_color="#ef4444", annotation_text="Seuil tempête 1005 hPa", annotation_font_color="#ef4444")
                apply_theme(fig); fig.update_layout(title="Pression MSL mensuelle", yaxis_title="hPa", height=340)
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            with st.spinner():
                df_msl_ext = q(f"SELECT MONTH, SUM(CASE WHEN msl<1005 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_tempete FROM hsv {wh} WHERE msl IS NOT NULL GROUP BY MONTH ORDER BY MONTH")
            if not df_msl_ext.empty:
                df_msl_ext["MOIS_L"] = df_msl_ext["MONTH"].map(MOIS_SHORT)
                fig = go.Figure(go.Bar(x=df_msl_ext["MOIS_L"], y=df_msl_ext["pct_tempete"],
                    marker_color=df_msl_ext["pct_tempete"].apply(lambda v: "#ef4444" if v>3 else "#f59e0b" if v>1 else "#10b981"),
                    text=df_msl_ext["pct_tempete"].map(lambda v: f"{v:.1f}%"), textposition="outside"))
                apply_theme(fig); fig.update_layout(title="% MSL < 1005 hPa par mois", yaxis_title="%", height=340)
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        section("🌬️", "Vent — impact sur la prise d'eau et la turbidité")
        col1, col2 = st.columns(2)
        with col1:
            with st.spinner():
                df_ws_desal = q(f"""
                    SELECT MONTH,
                           AVG(wind_speed) AS avg_ws, MAX(wind_speed) AS max_ws,
                           AVG(u10) AS avg_u10, AVG(v10) AS avg_v10,
                           SUM(CASE WHEN wind_speed>=10 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_fort
                    FROM hsv {wh} GROUP BY MONTH ORDER BY MONTH
                """)
            if not df_ws_desal.empty:
                df_ws_desal["MOIS_L"] = df_ws_desal["MONTH"].map(MOIS_SHORT)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_ws_desal["MOIS_L"], y=df_ws_desal["avg_ws"],
                    name="Vent moy", mode="lines+markers", line=dict(color="#06b6d4", width=2)))
                fig.add_trace(go.Bar(x=df_ws_desal["MOIS_L"], y=df_ws_desal["pct_fort"],
                    name="% vent ≥10 m/s", marker_color="rgba(239,68,68,0.5)", yaxis="y2"))
                fig.add_hline(y=10, line_dash="dash", line_color="#f59e0b", annotation_text="Alerte turbidité 10 m/s")
                apply_theme(fig)
                fig.update_layout(title="Vent mensuel — risque turbidité",
                    yaxis=dict(title="Vent moy (m/s)", **PLOTLY_THEME["yaxis"]),
                    yaxis2=dict(title="% vent fort", overlaying="y", side="right",
                                gridcolor="rgba(0,0,0,0)", tickcolor="#4a7a96", tickfont=dict(color="#94b8cc")),
                    height=360)
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            section("🧭", "Direction MWD — vagues vers la côte")
            with st.spinner():
                df_mwd_desal = q(f"""
                    SELECT FLOOR(mwd/22.5)*22.5 AS dir_bin, COUNT(*) AS n, AVG(MESURE) AS avg_hsv
                    FROM hsv {wh} WHERE mwd IS NOT NULL GROUP BY dir_bin ORDER BY dir_bin
                """)
            if not df_mwd_desal.empty:
                fig2 = go.Figure(go.Barpolar(
                    r=df_mwd_desal["n"], theta=df_mwd_desal["dir_bin"], width=22,
                    marker_color=df_mwd_desal["avg_hsv"],
                    marker_colorscale=[[0,"#10b981"],[0.5,"#f59e0b"],[1,"#ef4444"]],
                    marker_showscale=True,
                    marker_colorbar=dict(title="HSV moy (m)", tickfont=dict(color="#94b8cc"))
                ))
                apply_theme(fig2)
                fig2.update_layout(
                    title="Rose des vagues — directions vers stations",
                    polar=dict(
                        bgcolor="rgba(12,24,41,0.8)",
                        angularaxis=dict(tickmode="array", tickvals=[0,45,90,135,180,225,270,315],
                                         ticktext=["N","NE","E","SE","S","SO","O","NO"],
                                         direction="clockwise", rotation=90,
                                         gridcolor="rgba(30,90,150,0.3)", tickfont=dict(color="#94b8cc")),
                        radialaxis=dict(gridcolor="rgba(30,90,150,0.2)", tickfont=dict(color="#94b8cc"))
                    ), height=400
                )
                st.plotly_chart(fig2, use_container_width=True)

        section("🔗", "Corrélation vent × HSV — impact sur la prise d'eau")
        with st.spinner():
            df_ws_vs_hsv = q(f"""
                SELECT MONTH,
                       AVG(MESURE)     AS avg_hsv,
                       AVG(wind_speed) AS avg_ws,
                       SUM(CASE WHEN MESURE>=3.0 AND wind_speed>=10 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_double_alerte
                FROM hsv {wh} GROUP BY MONTH ORDER BY MONTH
            """)
        if not df_ws_vs_hsv.empty:
            df_ws_vs_hsv["MOIS_L"] = df_ws_vs_hsv["MONTH"].map(MOIS_SHORT)
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=df_ws_vs_hsv["MOIS_L"], y=df_ws_vs_hsv["avg_hsv"],
                name="HSV moy (m)", mode="lines+markers", line=dict(color="#0ea5e9", width=2)))
            fig3.add_trace(go.Scatter(x=df_ws_vs_hsv["MOIS_L"], y=df_ws_vs_hsv["avg_ws"],
                name="Vent moy (m/s)", mode="lines+markers", line=dict(color="#06b6d4", width=2), yaxis="y2"))
            apply_theme(fig3)
            fig3.update_layout(
                title="Corrélation HSV × Vent mensuel — risque prise d'eau",
                yaxis=dict(title="HSV (m)", **PLOTLY_THEME["yaxis"]),
                yaxis2=dict(title="Vent (m/s)", overlaying="y", side="right",
                            gridcolor="rgba(0,0,0,0)", tickcolor="#4a7a96", tickfont=dict(color="#94b8cc")),
                height=340
            )
            st.plotly_chart(fig3, use_container_width=True)

        section("📡", "Vent et HSV par zone de distance — sélection site dessalement")
        with st.spinner():
            df_dist_desal = q(f"""
                SELECT DISTANCE,
                       AVG(MESURE)     AS avg_hsv, AVG(wind_speed) AS avg_ws,
                       SUM(CASE WHEN sst BETWEEN 16 AND 26
                                AND msl > 1005
                                AND wind_speed < 10
                                AND MESURE < 3.0 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_ok
                FROM hsv {wh} WHERE sst IS NOT NULL AND msl IS NOT NULL
                GROUP BY DISTANCE ORDER BY DISTANCE
            """)
        if not df_dist_desal.empty:
            df_dist_desal["DIST_LABEL"] = df_dist_desal["DISTANCE"].map(DISTANCE_LABELS)
            dist_c = [list(DISTANCE_COLORS.values())[i % 4] for i in range(len(df_dist_desal))]
            fig4 = go.Figure()
            fig4.add_trace(go.Bar(x=df_dist_desal["DIST_LABEL"], y=df_dist_desal["avg_ws"],
                name="Vent moy (m/s)", marker_color=dist_c))
            fig4.add_trace(go.Scatter(x=df_dist_desal["DIST_LABEL"], y=df_dist_desal["pct_ok"],
                name="% cond. optimales SWRO", mode="lines+markers", line=dict(color="#10b981", width=2), yaxis="y2"))
            apply_theme(fig4)
            fig4.update_layout(title="Vent et % conditions optimales SWRO par zone",
                yaxis=dict(title="Vent moy (m/s)", **PLOTLY_THEME["yaxis"]),
                yaxis2=dict(title="% optimal", overlaying="y", side="right",
                            gridcolor="rgba(0,0,0,0)", tickcolor="#4a7a96", tickfont=dict(color="#94b8cc")),
                height=320)
            st.plotly_chart(fig4, use_container_width=True)

    with tab4:
        section("⚙️","Fenêtres opérationnelles optimales — SST + MSL + Vent + HSV")
        with st.spinner():
            df_ops = q(f"""
                SELECT MONTH,
                       SUM(CASE WHEN sst BETWEEN 16 AND 26
                                AND msl > 1005
                                AND wind_speed < 10
                                AND MESURE < 3.0 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_ok,
                       SUM(CASE WHEN sst BETWEEN 16 AND 26
                                AND msl > 1005
                                AND MESURE < 3.0 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_ok_sans_vent
                FROM hsv {wh} WHERE sst IS NOT NULL AND msl IS NOT NULL
                GROUP BY MONTH ORDER BY MONTH
            """)
        if not df_ops.empty:
            df_ops["MOIS_L"] = df_ops["MONTH"].map(MOIS_LABELS)
            col1, col2 = st.columns(2)
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=df_ops["MOIS_L"], y=df_ops["pct_ok"],
                    name="Avec contrainte vent",
                    marker_color=df_ops["pct_ok"].apply(lambda v: "#10b981" if v>80 else "#f59e0b" if v>60 else "#ef4444"),
                    text=df_ops["pct_ok"].map(lambda v: f"{v:.1f}%"), textposition="outside"))
                apply_theme(fig); fig.update_layout(title="% optimal SWRO (SST+MSL+Vent+HSV)",
                    yaxis_title="%", height=360)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(x=df_ops["MOIS_L"], y=df_ops["pct_ok_sans_vent"],
                    name="Sans contrainte vent", marker_color="#0ea5e9",
                    text=df_ops["pct_ok_sans_vent"].map(lambda v: f"{v:.1f}%"), textposition="outside"))
                fig2.add_trace(go.Bar(x=df_ops["MOIS_L"], y=df_ops["pct_ok"],
                    name="Avec contrainte vent", marker_color="#ef4444",
                    text=df_ops["pct_ok"].map(lambda v: f"{v:.1f}%"), textposition="outside"))
                apply_theme(fig2); fig2.update_layout(barmode="overlay", title="Impact du vent sur la disponibilité SWRO",
                    yaxis_title="%", height=360)
                st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# PAGE : AQUACULTURE
# ═══════════════════════════════════════════════════════════════════════
elif page == "🐟 Aquaculture":
    page_header("#10b981","🐟","Aquaculture Marine","Fenêtres favorables — HSV · SST · Vent · Direction · Distance")

    st.markdown("""
    <div class="info-card" style="border-left-color:#10b981; background:rgba(16,185,129,0.06);">
        <div class="title" style="color:#34d399;">Critères d'éligibilité aquaculture — tous paramètres</div>
        <div class="threshold-row"><span class="threshold-key">HSV sécurité cages</span><span class="threshold-val" style="color:#34d399;">HSV &lt; 1.2 m</span></div>
        <div class="threshold-row"><span class="threshold-key">SST croissance</span><span class="threshold-val" style="color:#34d399;">16 °C – 24 °C</span></div>
        <div class="threshold-row"><span class="threshold-key">MWP favorable</span><span class="threshold-val">&lt; 8 s</span></div>
        <div class="threshold-row"><span class="threshold-key">Vent favorable</span><span class="threshold-val" style="color:#34d399;">&lt; 8 m/s (stabilité structures)</span></div>
        <div class="threshold-row"><span class="threshold-key">📐 Risk Score</span><span class="threshold-val" style="color:#a78bfa;">(HSV/2)×0.4 + (vent/10)×0.3 + (SST_écart/8)×0.2 + (MWP/10)×0.1</span></div>
    </div>
    """, unsafe_allow_html=True)

    wh = W()
    show_kpis(wh)

    with st.spinner():
        r_aqua_vent = q(f"""
            SELECT AVG(wind_speed) AS avg_ws,
                   SUM(CASE WHEN wind_speed<8 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_fav_ws,
                   AVG(mwd) AS avg_mwd
            FROM hsv {wh}
        """)
    if not r_aqua_vent.empty:
        section("💨", "Indicateurs vent — aquaculture")
        c1,c2,c3 = st.columns(3)
        c1.metric("💨 Vent Moyen",         f"{r_aqua_vent['avg_ws'].iloc[0]:.2f} m/s")
        c2.metric("✅ % Vent Favorable <8 m/s", f"{r_aqua_vent['pct_fav_ws'].iloc[0]:.1f}%")
        c3.metric("🧭 MWD Moyen",          f"{r_aqua_vent['avg_mwd'].iloc[0]:.1f}°")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Fenêtres Favorables","🌬️ Vent & Direction","🏖️ Meilleurs Sites","📅 Saisonnalité"])

    with tab1:
        section("📊","Conditions favorables par mois — critère étendu avec vent")
        with st.spinner():
            df_aqua_m = q(f"""
                SELECT MONTH,
                       SUM(CASE WHEN AQUA_OK THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_ok,
                       SUM(CASE WHEN AQUA_OK AND wind_speed<8 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_ok_vent,
                       AVG(MESURE) AS avg_hsv, AVG(sst) AS avg_sst,
                       AVG(wind_speed) AS avg_ws, AVG(mwp) AS avg_mwp
                FROM hsv {wh} GROUP BY MONTH ORDER BY MONTH
            """)
        if not df_aqua_m.empty:
            df_aqua_m["MOIS_L"] = df_aqua_m["MONTH"].map(MOIS_SHORT)
            col1 = st.container()
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=df_aqua_m["MOIS_L"], y=df_aqua_m["pct_ok"],
                    name="Critères HSV+SST+MWP",
                    marker_color=df_aqua_m["pct_ok"].apply(lambda v: "#10b981" if v>70 else "#f59e0b" if v>40 else "#ef4444"),
                    text=df_aqua_m["pct_ok"].map(lambda v: f"{v:.0f}%"), textposition="outside"))
                fig.add_trace(go.Bar(x=df_aqua_m["MOIS_L"], y=df_aqua_m["pct_ok_vent"],
                    name="+ Vent <8 m/s", marker_color="rgba(6,182,212,0.6)"))
                apply_theme(fig); fig.update_layout(barmode="overlay",
                    title="% conditions favorables — avec et sans contrainte vent", yaxis_title="%", height=360)
                st.plotly_chart(fig, use_container_width=True)
            

        section("📈","Tendance annuelle % conditions favorables")
        with st.spinner():
            df_aqua_yr = q(f"""
                SELECT YEAR,
                       SUM(CASE WHEN AQUA_OK THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_ok,
                       SUM(CASE WHEN AQUA_OK AND wind_speed<8 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_ok_vent
                FROM hsv {wh} GROUP BY YEAR ORDER BY YEAR
            """)
        if not df_aqua_yr.empty:
            z  = np.polyfit(df_aqua_yr["YEAR"], df_aqua_yr["pct_ok_vent"], 1)
            p_ = np.poly1d(z)
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=df_aqua_yr["YEAR"], y=df_aqua_yr["pct_ok"],
                name="Critères HSV+SST+MWP", mode="lines+markers", line=dict(color="#10b981", width=2)))
            fig3.add_trace(go.Scatter(x=df_aqua_yr["YEAR"], y=df_aqua_yr["pct_ok_vent"],
                name="+ Vent <8 m/s", mode="lines+markers", line=dict(color="#06b6d4", width=2),
                fill="tozeroy", fillcolor="rgba(6,182,212,0.06)"))
            fig3.add_trace(go.Scatter(x=df_aqua_yr["YEAR"], y=p_(df_aqua_yr["YEAR"]),
                name="Tendance", mode="lines", line=dict(color="#f59e0b", dash="dash")))
            apply_theme(fig3); fig3.update_layout(title="Tendance — % conditions favorables aquaculture",
                xaxis_title="Année", yaxis_title="%", height=320)
            st.plotly_chart(fig3, use_container_width=True)

    with tab2:
        section("🌬️", "Vent mensuel — impact sur les structures aquacoles")
        col1, col2 = st.columns(2)
        with col1:
            with st.spinner():
                df_ws_aqua = q(f"""
                    SELECT MONTH, AVG(wind_speed) AS avg_ws, MAX(wind_speed) AS max_ws,
                           AVG(u10) AS avg_u10, AVG(v10) AS avg_v10,
                           SUM(CASE WHEN wind_speed<8 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_fav
                    FROM hsv {wh} GROUP BY MONTH ORDER BY MONTH
                """)
            if not df_ws_aqua.empty:
                df_ws_aqua["MOIS_L"] = df_ws_aqua["MONTH"].map(MOIS_SHORT)
                fig = go.Figure()
                fig.add_trace(go.Bar(x=df_ws_aqua["MOIS_L"], y=df_ws_aqua["avg_ws"],
                    name="Vent moy", marker_color=df_ws_aqua["avg_ws"].apply(
                        lambda v: "#ef4444" if v>8 else "#f59e0b" if v>5 else "#10b981"),
                    text=df_ws_aqua["avg_ws"].map(lambda v: f"{v:.1f}"), textposition="outside"))
                fig.add_hline(y=8, line_dash="dash", line_color="#f59e0b", annotation_text="Seuil 8 m/s")
                apply_theme(fig); fig.update_layout(title="Vent mensuel — sécurité cages", yaxis_title="m/s", height=340)
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            section("🧭", "Rose des vagues — direction favorable aquaculture")
            with st.spinner():
                df_mwd_aqua = q(f"""
                    SELECT FLOOR(mwd/22.5)*22.5 AS dir_bin, COUNT(*) AS n,
                           AVG(MESURE) AS avg_hsv,
                           SUM(CASE WHEN AQUA_OK THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_ok
                    FROM hsv {wh} WHERE mwd IS NOT NULL GROUP BY dir_bin ORDER BY dir_bin
                """)
            if not df_mwd_aqua.empty:
                fig2 = go.Figure(go.Barpolar(
                    r=df_mwd_aqua["pct_ok"], theta=df_mwd_aqua["dir_bin"], width=22,
                    marker_color=df_mwd_aqua["pct_ok"],
                    marker_colorscale=[[0,"#ef4444"],[0.5,"#f59e0b"],[1,"#10b981"]],
                    marker_showscale=True,
                    marker_colorbar=dict(title="% cond. fav.", tickfont=dict(color="#94b8cc"))
                ))
                apply_theme(fig2)
                fig2.update_layout(
                    title="Rose des directions — % conditions favorables aquaculture",
                    polar=dict(
                        bgcolor="rgba(12,24,41,0.8)",
                        angularaxis=dict(tickmode="array", tickvals=[0,45,90,135,180,225,270,315],
                                         ticktext=["N","NE","E","SE","S","SO","O","NO"],
                                         direction="clockwise", rotation=90,
                                         gridcolor="rgba(30,90,150,0.3)", tickfont=dict(color="#94b8cc")),
                        radialaxis=dict(gridcolor="rgba(30,90,150,0.2)", tickfont=dict(color="#94b8cc"))
                    ), height=400
                )
                st.plotly_chart(fig2, use_container_width=True)

        section("📡", "Conditions favorables par zone de distance — sélection site aquaculture")
        with st.spinner():
            df_dist_aqua = q(f"""
                SELECT DISTANCE,
                       SUM(CASE WHEN AQUA_OK THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_ok,
                       SUM(CASE WHEN AQUA_OK AND wind_speed<8 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_ok_vent,
                       AVG(MESURE)     AS avg_hsv,
                       AVG(wind_speed) AS avg_ws,
                       AVG(sst)        AS avg_sst
                FROM hsv {wh} GROUP BY DISTANCE ORDER BY DISTANCE
            """)
        if not df_dist_aqua.empty:
            df_dist_aqua["DIST_LABEL"] = df_dist_aqua["DISTANCE"].map(DISTANCE_LABELS)
            dist_c = [list(DISTANCE_COLORS.values())[i % 4] for i in range(len(df_dist_aqua))]

            cols_da = st.columns(len(df_dist_aqua))
            for i, (_, row) in enumerate(df_dist_aqua.iterrows()):
                c = list(DISTANCE_COLORS.values())[i % 4]
                score = row["pct_ok_vent"]
                score_color = "#10b981" if score > 60 else "#f59e0b" if score > 35 else "#ef4444"
                with cols_da[i]:
                    st.markdown(f"""
                    <div class="stat-card" style="border-top:2px solid {c};">
                        <div class="label">{row['DIST_LABEL']}</div>
                        <div class="value" style="color:{score_color};font-size:1.4rem;">{score:.1f}%</div>
                        <div class="sub">HSV moy: {row['avg_hsv']:.2f} m</div>
                        <div class="sub">Vent moy: {row['avg_ws']:.1f} m/s</div>
                        <div class="sub">SST moy: {row['avg_sst']:.1f}°C</div>
                    </div>
                    """, unsafe_allow_html=True)

            fig3 = go.Figure()
            fig3.add_trace(go.Bar(x=df_dist_aqua["DIST_LABEL"], y=df_dist_aqua["pct_ok"],
                name="HSV+SST+MWP", marker_color=dist_c))
            fig3.add_trace(go.Bar(x=df_dist_aqua["DIST_LABEL"], y=df_dist_aqua["pct_ok_vent"],
                name="+Vent <8m/s", marker_color="rgba(6,182,212,0.7)"))
            apply_theme(fig3); fig3.update_layout(barmode="overlay",
                title="% conditions favorables par zone de distance", yaxis_title="%", height=320)
            st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        section("🏖️","Meilleurs sites potentiels — Risk Score Aquaculture")

        with st.spinner():
            df_sites = q(f"""
                SELECT NOM_PLAGE, NOM_WILAYA, X, Y, DISTANCE,
                       SUM(CASE WHEN AQUA_OK AND wind_speed<8 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_ok,
                       AVG(MESURE)      AS avg_hsv,
                       AVG(sst)         AS avg_sst,
                       AVG(wind_speed)  AS avg_ws,
                       AVG(mwd)         AS avg_mwd,
                       AVG(mwp)         AS avg_mwp,
                       COUNT(*)         AS n
                FROM hsv {wh}
                GROUP BY NOM_PLAGE, NOM_WILAYA, X, Y, DISTANCE
            """)

        if not df_sites.empty:
            # ── Risk Score Aquaculture ──────────────────────────────────
            # Composantes : HSV (sécurité cages), vent, SST (écart zone optimale), MWP
            df_sites["avg_sst"] = df_sites["avg_sst"].fillna(20.0)
            df_sites["avg_mwp"] = df_sites["avg_mwp"].fillna(df_sites["avg_mwp"].median())
            # SST écart par rapport au centre de la plage optimale (20°C)
            df_sites["sst_ecart"] = (df_sites["avg_sst"] - 20.0).abs() / 8.0
            df_sites["sst_ecart"] = df_sites["sst_ecart"].clip(0, 1)

            df_sites["risk_score_aqua"] = (
                (df_sites["avg_hsv"] / 2.0)   * 0.4
              + (df_sites["avg_ws"]  / 10.0)  * 0.3
              + df_sites["sst_ecart"]          * 0.2
              + (df_sites["avg_mwp"] / 10.0)  * 0.1
            ).round(4)

            # Score de favorabilité = inverse du risk
            df_sites["favorabilite"] = (1 - df_sites["risk_score_aqua"].clip(0, 1)) * 100

            df_top = df_sites.sort_values("pct_ok", ascending=False).head(10)

            for _, row in df_top.iterrows():
                score = row["pct_ok"]
                color = "#10b981" if score > 60 else "#f59e0b" if score > 35 else "#ef4444"
                dist_lbl = DISTANCE_LABELS.get(int(row['DISTANCE']) if not np.isnan(row['DISTANCE']) else 1, "N/A")
                st.markdown(f"""
                <div style="background:var(--bg-card);border:1px solid var(--border-soft);
                     border-left:3px solid {color};border-radius:var(--r-md);
                     padding:10px 14px;margin-bottom:6px;
                     display:flex;align-items:center;justify-content:space-between;">
                    <div>
                        <span style="font-size:0.85rem;font-weight:600;color:var(--text-h);">{row['NOM_PLAGE']}</span>
                        <span style="font-size:0.75rem;color:var(--text-muted);margin-left:8px;">{row['NOM_WILAYA']} · {dist_lbl}</span>
                    </div>
                    <div style="text-align:right;">
                        <span style="font-family:var(--font-mono);font-size:0.95rem;font-weight:700;color:{color};">{score:.1f}% fav.</span>
                        <span style="font-size:0.72rem;color:var(--text-muted);margin-left:6px;">
                            HSV: {row['avg_hsv']:.2f}m · Vent: {row['avg_ws']:.1f}m/s
                            {f"· SST: {row['avg_sst']:.1f}°C" if not np.isnan(row['avg_sst']) else ""}
                            · MWP: {row['avg_mwp']:.1f}s · Risk: {row['risk_score_aqua']:.3f}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("""
            <div style="font-size:0.82rem;color:#94b8cc;font-style:italic;margin-bottom:1rem;">
              Risk Score Aquaculture : <code>(HSV/2)×0.4 + (Vent/10)×0.3 + (|SST-20|/8)×0.2 + (MWP/10)×0.1</code>
            </div>
            """, unsafe_allow_html=True)

            section("🗺️","Carte — sites aquaculture (Risk Score Aquaculture)")
            if not df_sites[["X","Y"]].isnull().any().any():
                fig = px.scatter_mapbox(
                    df_sites, lat="Y", lon="X",
                    color="risk_score_aqua", size="pct_ok",
                    hover_name="NOM_PLAGE",
                    hover_data={"NOM_WILAYA":True,"avg_hsv":":.2f","avg_ws":":.1f","avg_sst":":.1f","avg_mwp":":.1f","pct_ok":":.1f","risk_score_aqua":":.3f","X":False,"Y":False},
                    color_continuous_scale=["#10b981","#f59e0b","#ef4444"],
                    range_color=[0, 0.8], size_max=18, zoom=5,
                    mapbox_style="carto-darkmatter",
                    labels={"risk_score_aqua":"Risk Aqua","pct_ok":"% fav."}
                )
                apply_theme(fig); fig.update_layout(height=450, margin=dict(l=0,r=0,t=40,b=0),
                    title="Sites aquaculture — Risk Score (vert=favorable, rouge=risqué)")
                st.plotly_chart(fig, use_container_width=True)

    with tab4:
        section("📅","SST, Vent et HSV — critères de croissance par saison")
        if has_col("sst"):
            with st.spinner():
                df_sst_aqua = q(f"""
                    SELECT MONTH, SEASON,
                           AVG(sst) AS avg_sst, AVG(MESURE) AS avg_hsv,
                           AVG(wind_speed) AS avg_ws,
                           SUM(CASE WHEN sst BETWEEN 16 AND 24 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_sst_ok,
                           SUM(CASE WHEN MESURE < 1.2 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_hsv_ok,
                           SUM(CASE WHEN wind_speed < 8 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_ws_ok
                    FROM hsv {wh} WHERE sst IS NOT NULL
                    GROUP BY MONTH, SEASON ORDER BY MONTH
                """)
            if not df_sst_aqua.empty:
                df_sst_aqua["MOIS_L"] = df_sst_aqua["MONTH"].map(MOIS_SHORT)
                col1, col2 = st.columns(2)
                with col1:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=df_sst_aqua["MOIS_L"], y=df_sst_aqua["pct_sst_ok"], name="SST 16–24°C", marker_color="#06b6d4"))
                    fig.add_trace(go.Bar(x=df_sst_aqua["MOIS_L"], y=df_sst_aqua["pct_hsv_ok"], name="HSV < 1.2 m",  marker_color="#10b981"))
                    fig.add_trace(go.Bar(x=df_sst_aqua["MOIS_L"], y=df_sst_aqua["pct_ws_ok"],  name="Vent < 8 m/s", marker_color="#8b5cf6"))
                    apply_theme(fig); fig.update_layout(barmode="group",
                        title="% critères satisfaits par mois", yaxis_title="%", height=340)
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    fig2 = px.scatter(df_sst_aqua, x="avg_sst", y="avg_hsv",
                        color="SEASON", size="avg_ws", text="MOIS_L",
                        color_discrete_map=SEASON_COLORS,
                        labels={"avg_sst":"SST (°C)","avg_hsv":"HSV (m)","avg_ws":"Vent moy (m/s)","SEASON":"Saison"},
                        size_max=20)
                    fig2.add_hrect(y0=0, y1=1.2, fillcolor="rgba(16,185,129,0.08)", line_width=0)
                    fig2.add_vrect(x0=16, x1=24, fillcolor="rgba(16,185,129,0.08)", line_width=0)
                    apply_theme(fig2); fig2.update_layout(
                        title="Zone favorable (SST × HSV) — taille bulle = vent", height=340)
                    st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# PAGE : SYNTHÈSE & EXPORT
# ═══════════════════════════════════════════════════════════════════════
elif page == "📋 Synthèse & Export":
    page_header("#8b5cf6","📋","Synthèse & Export","Tableaux récapitulatifs et téléchargement des données")
    wh = W()

    tab1, tab2, tab3 = st.tabs(["📊 Synthèse par Plage","📅 Synthèse Mensuelle","💾 Export"])

    with tab1:
        section("📊","Statistiques complètes par plage — HSV, vent, direction")
        with st.spinner("Calcul en cours..."):
            cols_extra = ""
            if has_col("sst"):    cols_extra += ", AVG(sst) AS avg_sst, MIN(sst) AS min_sst, MAX(sst) AS max_sst"
            if has_col("mwp"):    cols_extra += ", AVG(mwp) AS avg_mwp"
            if has_col("msl"):    cols_extra += ", AVG(msl) AS avg_msl, MIN(msl) AS min_msl"
            if has_col("wind_speed"): cols_extra += ", AVG(wind_speed) AS avg_ws, MAX(wind_speed) AS max_ws, SUM(CASE WHEN wind_speed>=10 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_vent_fort"
            if has_col("mwd"):    cols_extra += ", AVG(mwd) AS avg_mwd"

            df_synth = q(f"""
                SELECT NOM_PLAGE, NOM_WILAYA,
                       COUNT(*)        AS n,
                       ROUND(AVG(MESURE),3)    AS avg_hsv,
                       ROUND(MAX(MESURE),2)    AS max_hsv,
                       ROUND(STDDEV(MESURE),3) AS std_hsv,
                       ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY MESURE),2) AS p95,
                       ROUND(SUM(CASE WHEN MESURE>=1 THEN 1 ELSE 0 END)*100.0/COUNT(*),1) AS pct_vig,
                       ROUND(SUM(CASE WHEN MESURE>=2 THEN 1 ELSE 0 END)*100.0/COUNT(*),1) AS pct_danger,
                       ROUND(SUM(CASE WHEN AQUA_OK THEN 1 ELSE 0 END)*100.0/COUNT(*),1)   AS pct_aqua,
                       ROUND(SUM(CASE WHEN DESSAL_OK THEN 1 ELSE 0 END)*100.0/COUNT(*),1) AS pct_dessal
                       {cols_extra}
                FROM hsv {wh} GROUP BY NOM_PLAGE, NOM_WILAYA ORDER BY avg_hsv DESC
            """)
        if not df_synth.empty:
            st.dataframe(df_synth, use_container_width=True, hide_index=True)
            csv = df_synth.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Télécharger CSV", csv, "synthese_plages.csv", "text/csv")

    with tab2:
        section("📅","Statistiques mensuelles globales — toutes variables")
        with st.spinner():
            cols_m = ""
            if has_col("sst"):        cols_m += ", ROUND(AVG(sst),2) AS avg_sst"
            if has_col("mwp"):        cols_m += ", ROUND(AVG(mwp),2) AS avg_mwp"
            if has_col("msl"):        cols_m += ", ROUND(AVG(msl),1) AS avg_msl"
            if has_col("wind_speed"): cols_m += ", ROUND(AVG(wind_speed),2) AS avg_ws, ROUND(MAX(wind_speed),1) AS max_ws, ROUND(SUM(CASE WHEN wind_speed>=10 THEN 1 ELSE 0 END)*100.0/COUNT(*),1) AS pct_vent_fort"
            if has_col("mwd"):        cols_m += ", ROUND(AVG(mwd),1) AS avg_mwd"

            df_mo = q(f"""
                SELECT MONTH, COUNT(*) AS n,
                       ROUND(AVG(MESURE),3)    AS avg_hsv,
                       ROUND(MAX(MESURE),2)    AS max_hsv,
                       ROUND(STDDEV(MESURE),3) AS std_hsv,
                       ROUND(SUM(CASE WHEN MESURE>=1 THEN 1 ELSE 0 END)*100.0/COUNT(*),1) AS pct_vig,
                       ROUND(SUM(CASE WHEN MESURE>=2 THEN 1 ELSE 0 END)*100.0/COUNT(*),1) AS pct_danger,
                       ROUND(SUM(CASE WHEN AQUA_OK THEN 1 ELSE 0 END)*100.0/COUNT(*),1)   AS pct_aqua,
                       ROUND(SUM(CASE WHEN DESSAL_OK THEN 1 ELSE 0 END)*100.0/COUNT(*),1) AS pct_dessal
                       {cols_m}
                FROM hsv {wh} GROUP BY MONTH ORDER BY MONTH
            """)
        if not df_mo.empty:
            df_mo["MOIS"] = df_mo["MONTH"].map(MOIS_LABELS)
            df_mo = df_mo.drop(columns=["MONTH"])
            cols_ord = ["MOIS"] + [c for c in df_mo.columns if c != "MOIS"]
            st.dataframe(df_mo[cols_ord], use_container_width=True, hide_index=True)
            csv2 = df_mo[cols_ord].to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Télécharger CSV mensuel", csv2, "synthese_mensuelle.csv", "text/csv")

    with tab3:
        section("💾","Export personnalisé")
        col_choices = st.multiselect("Colonnes à exporter",
            ["NOM_PLAGE","NOM_WILAYA","DATETIME","MESURE","ALERTE","NIVEAU",
             "wind_speed","u10","v10","mwp","mwd","sst","msl","DISTANCE","SEASON","YEAR","MONTH"],
            default=["NOM_PLAGE","NOM_WILAYA","DATETIME","MESURE","ALERTE","NIVEAU","wind_speed","mwp","mwd"])
        max_rows = st.slider("Nombre maximum de lignes", 1000, 100000, 10000, 1000)
        if col_choices:
            with st.spinner("Extraction..."):
                cols_sql = ", ".join(col_choices)
                df_exp = q(f"SELECT {cols_sql} FROM hsv {wh} LIMIT {max_rows}")
            st.dataframe(df_exp, use_container_width=True, hide_index=True)
            csv3 = df_exp.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Télécharger CSV", csv3, "export_hsv.csv", "text/csv")
            pq_bytes = df_exp.to_parquet(index=False)
            st.download_button("⬇️ Télécharger Parquet", pq_bytes, "export_hsv.parquet", "application/octet-stream")

# ═══════════════════════════════════════════════════════════════════════
# PAGE : CARTE DES DANGERS
# ═══════════════════════════════════════════════════════════════════════
elif page == "🗺️ Carte des Dangers":
    page_header("#ef4444","🗺️","Carte des Dangers","Cartographie interactive des risques — côtes algériennes")

    wh = W()

    section("🗺️","Carte interactive des plages")
    map_metric = st.radio(
        "Indicateur cartographié",
        ["HSV Moyenne", "% Danger (≥2m)", "% Aquaculture OK", "% Dessalement OK", "HSV Maximum", "HSV par Distance"],
        horizontal=True
    )

    # ── Modes standard ──────────────────────────────────────────────────
    if map_metric != "HSV par Distance":

        metric_map = {
            "HSV Moyenne":       ("avg_hsv",    "AVG(MESURE)",   "HSV Moy (m)",    "Blues"),
            "% Danger (≥2m)":    ("pct_danger", "SUM(CASE WHEN MESURE>=2 THEN 1 ELSE 0 END)*100.0/COUNT(*)", "% Danger", ["#10b981","#f59e0b","#ef4444"]),
            "% Aquaculture OK":  ("pct_aqua",   "SUM(CASE WHEN AQUA_OK THEN 1 ELSE 0 END)*100.0/COUNT(*)",  "% Aqua OK", ["#ef4444","#f59e0b","#10b981"]),
            "% Dessalement OK":  ("pct_dessal", "SUM(CASE WHEN DESSAL_OK THEN 1 ELSE 0 END)*100.0/COUNT(*)", "% Dessal OK", ["#ef4444","#f59e0b","#10b981"]),
            "HSV Maximum":       ("max_hsv",    "MAX(MESURE)",   "HSV Max (m)",    "Reds"),
        }

        col_name, sql_expr, label, cscale = metric_map[map_metric]

        with st.spinner("Chargement des données cartographiques..."):
            extra_sst = ", ROUND(AVG(sst),1) AS avg_sst" if has_col("sst") else ""
            df_map = q(f"""
                SELECT NOM_PLAGE, NOM_WILAYA,
                       FIRST(X) AS lon, FIRST(Y) AS lat,
                       ROUND({sql_expr}, 3) AS val,
                       AVG(MESURE)  AS avg_hsv,
                       MAX(MESURE)  AS max_hsv
                       {extra_sst}
                FROM hsv {wh}
                GROUP BY NOM_PLAGE, NOM_WILAYA
                HAVING lon IS NOT NULL AND lat IS NOT NULL
            """)

        if not df_map.empty:
            hover_data = {"NOM_WILAYA": True, "avg_hsv": ":.2f", "max_hsv": ":.2f",
                          "lon": False, "lat": False}
            if "avg_sst" in df_map.columns:
                hover_data["avg_sst"] = ":.1f"

            fig = px.scatter_mapbox(
                df_map, lat="lat", lon="lon",
                color="val", size="val",
                hover_name="NOM_PLAGE",
                hover_data=hover_data,
                color_continuous_scale=cscale,
                size_max=22, zoom=5,
                mapbox_style="carto-darkmatter",
                labels={"val": label}
            )
            apply_theme(fig)
            fig.update_layout(
                height=520,
                margin=dict(l=0,r=0,t=40,b=0),
                title=f"Côtes algériennes — {map_metric}"
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── Classement par wilaya ──────────────────────────────────
            section("📊","Classement par wilaya")
            with st.spinner():
                df_wil = q(f"""
                    SELECT NOM_WILAYA,
                           COUNT(DISTINCT NOM_PLAGE) AS nb_plages,
                           ROUND(AVG(MESURE),3)  AS avg_hsv,
                           ROUND(MAX(MESURE),2)  AS max_hsv,
                           ROUND(SUM(CASE WHEN MESURE>=2 THEN 1 ELSE 0 END)*100.0/COUNT(*),1) AS pct_danger,
                           ROUND(SUM(CASE WHEN AQUA_OK THEN 1 ELSE 0 END)*100.0/COUNT(*),1)  AS pct_aqua,
                           ROUND(SUM(CASE WHEN DESSAL_OK THEN 1 ELSE 0 END)*100.0/COUNT(*),1) AS pct_dessal
                    FROM hsv {wh}
                    GROUP BY NOM_WILAYA ORDER BY avg_hsv DESC
                """)
            if not df_wil.empty:
                df_wil.columns = ["Wilaya","Nb plages","HSV Moy (m)","HSV Max (m)","% Danger","% Aqua OK","% Dessal OK"]
                st.dataframe(df_wil, use_container_width=True, hide_index=True)

            # ── Classement par plage ───────────────────────────────────
            section("🏖️","Classement par plage")
            col_sort, col_n = st.columns([3, 1])
            with col_sort:
                sort_by = st.selectbox(
                    "Trier par",
                    ["HSV Moyenne ↓", "HSV Maximum ↓", "% Danger ↓", "% Aquaculture OK ↓", "% Dessalement OK ↓"],
                    key="sort_plage_std"
                )
            with col_n:
                top_n = st.number_input("Top N plages", min_value=5, max_value=56, value=20, step=5, key="topn_std")

            sort_col_map = {
                "HSV Moyenne ↓":        "avg_hsv",
                "HSV Maximum ↓":        "max_hsv",
                "% Danger ↓":           "pct_danger",
                "% Aquaculture OK ↓":   "pct_aqua",
                "% Dessalement OK ↓":   "pct_dessal",
            }
            order_col = sort_col_map[sort_by]

            with st.spinner():
                df_plage = q(f"""
                    SELECT NOM_PLAGE, NOM_WILAYA,
                           ROUND(AVG(MESURE),3)  AS avg_hsv,
                           ROUND(MAX(MESURE),2)  AS max_hsv,
                           ROUND(SUM(CASE WHEN MESURE>=2 THEN 1 ELSE 0 END)*100.0/COUNT(*),1) AS pct_danger,
                           ROUND(SUM(CASE WHEN AQUA_OK THEN 1 ELSE 0 END)*100.0/COUNT(*),1)  AS pct_aqua,
                           ROUND(SUM(CASE WHEN DESSAL_OK THEN 1 ELSE 0 END)*100.0/COUNT(*),1) AS pct_dessal,
                           COUNT(*) AS nb_obs
                    FROM hsv {wh}
                    GROUP BY NOM_PLAGE, NOM_WILAYA
                    ORDER BY {order_col} DESC
                    LIMIT {top_n}
                """)
            if not df_plage.empty:
                df_plage.columns = ["Plage","Wilaya","HSV Moy (m)","HSV Max (m)","% Danger","% Aqua OK","% Dessal OK","Nb obs"]

                # Graphique horizontal
                fig_p = go.Figure()
                bar_colors = df_plage["% Danger"].apply(
                    lambda v: "#ef4444" if v >= 30 else ("#f59e0b" if v >= 15 else "#10b981")
                )
                fig_p.add_trace(go.Bar(
                    x=df_plage["HSV Moy (m)"],
                    y=df_plage["Plage"],
                    orientation="h",
                    marker_color=bar_colors,
                    text=df_plage["HSV Moy (m)"].map(lambda v: f"{v:.3f} m"),
                    textposition="outside",
                    customdata=df_plage[["Wilaya","% Danger","HSV Max (m)","% Aqua OK","% Dessal OK"]].values,
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "Wilaya : %{customdata[0]}<br>"
                        "HSV Moy : %{x:.3f} m<br>"
                        "HSV Max : %{customdata[2]:.2f} m<br>"
                        "% Danger : %{customdata[1]:.1f}%<br>"
                        "% Aqua OK : %{customdata[3]:.1f}%<br>"
                        "% Dessal OK : %{customdata[4]:.1f}%<extra></extra>"
                    )
                ))
                apply_theme(fig_p)
                fig_p.update_layout(
                    title=f"Top {top_n} plages — {sort_by.replace(' ↓','')}",
                    xaxis_title="HSV Moyenne (m)",
                    yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
                    height=max(340, top_n * 22),
                    margin=dict(l=0, r=80, t=40, b=0),
                    showlegend=False
                )
                st.plotly_chart(fig_p, use_container_width=True)

                # Légende couleurs danger
                st.markdown("""
                <div style="display:flex;gap:1.5rem;margin-top:-0.5rem;margin-bottom:1rem;font-size:0.82rem;">
                    <span style="color:#ef4444;">● % Danger ≥ 30% — Risque élevé</span>
                    <span style="color:#f59e0b;">● % Danger ≥ 15% — Risque modéré</span>
                    <span style="color:#10b981;">● % Danger &lt; 15% — Risque faible</span>
                </div>
                """, unsafe_allow_html=True)

                # Tableau détaillé
                st.dataframe(df_plage, use_container_width=True, hide_index=True)

        else:
            st.warning("Données cartographiques non disponibles.")

    # ── Mode : HSV par Distance ─────────────────────────────────────────
    else:
        st.markdown("""
        <div class="info-card" style="border-left-color:#0ea5e9;">
            <div class="title" style="color:#38bdf8;">Signification des zones de distance</div>
            <div class="threshold-row"><span class="threshold-key" style="color:#ef4444;">● Dist. 1 — ~1 km</span><span class="threshold-val">Frontière terre-mer · Danger direct baigneurs</span></div>
            <div class="threshold-row"><span class="threshold-key" style="color:#f59e0b;">● Dist. 2 — ~5 km</span><span class="threshold-val">Zone de baignade étendue</span></div>
            <div class="threshold-row"><span class="threshold-key" style="color:#0ea5e9;">● Dist. 3 — ~10 km</span><span class="threshold-val">Zone intermédiaire</span></div>
            <div class="threshold-row"><span class="threshold-key" style="color:#8b5cf6;">● Dist. 4 — ~20 km</span><span class="threshold-val">Conditions du large</span></div>
        </div>
        """, unsafe_allow_html=True)

        dist_select = st.multiselect(
            "Distances à afficher",
            options=[1,2,3,4],
            default=[1,2,3,4],
            format_func=lambda x: DISTANCE_LABELS[x]
        )

        if dist_select:
            dist_in  = ",".join(str(d) for d in dist_select)
            extra_wh = where_clause(f"DISTANCE IN ({dist_in})")

            with st.spinner("Chargement clusters..."):
                df_dm = q(f"""
                    SELECT DISTANCE, X, Y, NOM_WILAYA, NOM_PLAGE,
                           AVG(MESURE)     AS avg_hsv,
                           MAX(MESURE)     AS max_hsv,
                           AVG(wind_speed) AS avg_ws,
                           MAX(wind_speed) AS max_ws,
                           AVG(mwp)        AS avg_mwp,
                           AVG(mwd)        AS avg_mwd,
                           SUM(CASE WHEN MESURE>=2    THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_danger,
                           SUM(CASE WHEN wind_speed>=10 THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_vent_fort,
                           SUM(CASE WHEN AQUA_OK THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_ok,
                           SUM(CASE WHEN DESSAL_OK THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_dessal,
                           COUNT(*)        AS nb
                    FROM hsv {extra_wh}
                    GROUP BY DISTANCE, X, Y, NOM_WILAYA, NOM_PLAGE
                """)

            if not df_dm.empty:
                df_dm["DIST_LABEL"] = df_dm["DISTANCE"].map(DISTANCE_LABELS)

                # ── Carte ──────────────────────────────────────────────
                section("📡","Clusters HSV et vent par distance à la côte")
                fig = px.scatter_mapbox(
                    df_dm, lat="Y", lon="X",
                    color="DIST_LABEL",
                    size="avg_hsv",
                    size_max=22,
                    hover_name="NOM_PLAGE",
                    hover_data={
                        "NOM_WILAYA": True, "DIST_LABEL": True,
                        "avg_hsv": ":.3f", "max_hsv": ":.2f",
                        "avg_ws": ":.1f", "avg_mwp": ":.1f",
                        "avg_mwd": ":.0f", "pct_danger": ":.1f",
                        "nb": True, "Y": False, "X": False
                    },
                    mapbox_style="carto-darkmatter",
                    zoom=5, center={"lat": 36.5, "lon": 3.0},
                    color_discrete_map=DISTANCE_COLORS,
                    title="Côtes algériennes — HSV par distance à la côte"
                )
                fig.update_layout(
                    paper_bgcolor="rgba(4,18,32,0)",
                    font_color="#7fb5d5",
                    height=520,
                    margin=dict(l=0,r=0,t=40,b=0)
                )
                st.plotly_chart(fig, use_container_width=True)

                # ── KPIs par distance ──────────────────────────────────
                section("📊","HSV, vent et danger par zone")
                df_kpi = (
                    df_dm.groupby("DIST_LABEL")
                    .agg(
                        avg_hsv       =("avg_hsv",       "mean"),
                        max_hsv       =("max_hsv",       "max"),
                        avg_ws        =("avg_ws",        "mean"),
                        pct_danger    =("pct_danger",    "mean"),
                        pct_vent_fort =("pct_vent_fort", "mean"),
                    )
                    .reset_index()
                )

                cols_d = st.columns(len(df_kpi))
                for i, (_, row) in enumerate(df_kpi.iterrows()):
                    c = list(DISTANCE_COLORS.values())[i % 4]
                    with cols_d[i]:
                        st.markdown(f"""
                        <div class="stat-card" style="border-top:2px solid {c};">
                            <div class="label">{row['DIST_LABEL']}</div>
                            <div class="value" style="color:{c};font-size:1.3rem;">{row['avg_hsv']:.2f} m</div>
                            <div class="sub">% Danger : {row['pct_danger']:.1f}%</div>
                            <div class="sub">Vent moy : {row['avg_ws']:.1f} m/s · % fort : {row['pct_vent_fort']:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)

                # ── Graphiques ─────────────────────────────────────────
                col_a, col_b = st.columns(2)
                colors = [list(DISTANCE_COLORS.values())[i % 4] for i in range(len(df_kpi))]

                with col_a:
                    fig2 = go.Figure()
                    fig2.add_trace(go.Bar(
                        x=df_kpi["DIST_LABEL"], y=df_kpi["avg_hsv"],
                        name="HSV moy (m)", marker_color=colors,
                        text=df_kpi["avg_hsv"].map(lambda v: f"{v:.2f}"),
                        textposition="outside"
                    ))
                    fig2.add_trace(go.Scatter(
                        x=df_kpi["DIST_LABEL"], y=df_kpi["pct_danger"],
                        name="% Danger", mode="lines+markers",
                        line=dict(color="#ef4444", width=2), yaxis="y2"
                    ))
                    apply_theme(fig2)
                    fig2.update_layout(
                        title="HSV et % danger par zone",
                        yaxis=dict(title="HSV (m)", **PLOTLY_THEME["yaxis"]),
                        yaxis2=dict(
                            title="% Danger", overlaying="y", side="right",
                            gridcolor="rgba(0,0,0,0)",
                            tickcolor="#4a7a96", tickfont=dict(color="#94b8cc")
                        ),
                        height=320
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                with col_b:
                    fig3 = go.Figure()
                    fig3.add_trace(go.Bar(
                        x=df_kpi["DIST_LABEL"], y=df_kpi["avg_ws"],
                        name="Vent moy (m/s)", marker_color=colors,
                        text=df_kpi["avg_ws"].map(lambda v: f"{v:.1f}"),
                        textposition="outside"
                    ))
                    fig3.add_trace(go.Scatter(
                        x=df_kpi["DIST_LABEL"], y=df_kpi["pct_vent_fort"],
                        name="% Vent fort", mode="lines+markers",
                        line=dict(color="#f59e0b", width=2), yaxis="y2"
                    ))
                    apply_theme(fig3)
                    fig3.update_layout(
                        title="Vent et % fort par zone",
                        yaxis=dict(title="Vent moy (m/s)", **PLOTLY_THEME["yaxis"]),
                        yaxis2=dict(
                            title="% Vent fort", overlaying="y", side="right",
                            gridcolor="rgba(0,0,0,0)",
                            tickcolor="#4a7a96", tickfont=dict(color="#94b8cc")
                        ),
                        height=320
                    )
                    st.plotly_chart(fig3, use_container_width=True)

                # ── Classement par wilaya (mode distance) ──────────────
                section("📊","Classement par wilaya — zones sélectionnées")
                with st.spinner():
                    df_wil_d = q(f"""
                        SELECT NOM_WILAYA,
                               COUNT(DISTINCT NOM_PLAGE)  AS nb_plages,
                               ROUND(AVG(MESURE),3)       AS avg_hsv,
                               ROUND(MAX(MESURE),2)       AS max_hsv,
                               ROUND(SUM(CASE WHEN MESURE>=2 THEN 1 ELSE 0 END)*100.0/COUNT(*),1) AS pct_danger,
                               ROUND(AVG(wind_speed),1)   AS avg_ws,
                               ROUND(SUM(CASE WHEN AQUA_OK THEN 1 ELSE 0 END)*100.0/COUNT(*),1)  AS pct_aqua,
                               ROUND(SUM(CASE WHEN DESSAL_OK THEN 1 ELSE 0 END)*100.0/COUNT(*),1) AS pct_dessal
                        FROM hsv {extra_wh}
                        GROUP BY NOM_WILAYA ORDER BY avg_hsv DESC
                    """)
                if not df_wil_d.empty:
                    df_wil_d.columns = ["Wilaya","Nb plages","HSV Moy (m)","HSV Max (m)","% Danger","Vent Moy (m/s)","% Aqua OK","% Dessal OK"]
                    st.dataframe(df_wil_d, use_container_width=True, hide_index=True)

                # ── Classement par plage (mode distance) ───────────────
                section("🏖️","Classement par plage — zones sélectionnées")
                col_sort_d, col_n_d, col_dist_d = st.columns([2, 1, 2])
                with col_sort_d:
                    sort_by_d = st.selectbox(
                        "Trier par",
                        ["HSV Moyenne ↓", "HSV Maximum ↓", "% Danger ↓", "Vent Moyen ↓", "% Aquaculture OK ↓", "% Dessalement OK ↓"],
                        key="sort_plage_dist"
                    )
                with col_n_d:
                    top_n_d = st.number_input(
                        "Top N plages", min_value=5, max_value=56,
                        value=20, step=5, key="topn_dist"
                    )
                with col_dist_d:
                    dist_group = st.radio(
                        "Affichage",
                        ["Toutes distances confondues", "Par distance"],
                        horizontal=True,
                        key="dist_group_radio"
                    )

                sort_col_dist = {
                    "HSV Moyenne ↓":        "avg_hsv",
                    "HSV Maximum ↓":        "max_hsv",
                    "% Danger ↓":           "pct_danger",
                    "Vent Moyen ↓":         "avg_ws",
                    "% Aquaculture OK ↓":   "pct_ok",
                    "% Dessalement OK ↓":   "pct_dessal",
                }[sort_by_d]

                if dist_group == "Toutes distances confondues":
                    # Agrégation toutes distances
                    df_plage_d = (
                        df_dm.groupby(["NOM_PLAGE","NOM_WILAYA"])
                        .agg(
                            avg_hsv      =("avg_hsv",      "mean"),
                            max_hsv      =("max_hsv",      "max"),
                            avg_ws       =("avg_ws",       "mean"),
                            pct_danger   =("pct_danger",   "mean"),
                            pct_vent_fort=("pct_vent_fort","mean"),
                            pct_ok       =("pct_ok",       "mean"),
                            pct_dessal   =("pct_dessal",   "mean"),
                            nb           =("nb",           "sum"),
                        )
                        .reset_index()
                        .sort_values(sort_col_dist, ascending=False)
                        .head(top_n_d)
                    )

                    bar_colors_d = df_plage_d["pct_danger"].apply(
                        lambda v: "#ef4444" if v >= 30 else ("#f59e0b" if v >= 15 else "#10b981")
                    )
                    fig_pd = go.Figure()
                    fig_pd.add_trace(go.Bar(
                        x=df_plage_d["avg_hsv"],
                        y=df_plage_d["NOM_PLAGE"],
                        orientation="h",
                        marker_color=bar_colors_d,
                        text=df_plage_d["avg_hsv"].map(lambda v: f"{v:.3f} m"),
                        textposition="outside",
                        customdata=df_plage_d[["NOM_WILAYA","pct_danger","max_hsv","avg_ws","pct_vent_fort","pct_ok","pct_dessal"]].values,
                        hovertemplate=(
                            "<b>%{y}</b><br>"
                            "Wilaya : %{customdata[0]}<br>"
                            "HSV Moy : %{x:.3f} m<br>"
                            "HSV Max : %{customdata[2]:.2f} m<br>"
                            "% Danger : %{customdata[1]:.1f}%<br>"
                            "Vent Moy : %{customdata[3]:.1f} m/s<br>"
                            "% Vent fort : %{customdata[4]:.1f}%<br>"
                            "% Aqua OK : %{customdata[5]:.1f}%<br>"
                            "% Dessal OK : %{customdata[6]:.1f}%<extra></extra>"
                        )
                    ))
                    apply_theme(fig_pd)
                    fig_pd.update_layout(
                        title=f"Top {top_n_d} plages — {sort_by_d.replace(' ↓','')} (toutes distances)",
                        xaxis_title="HSV Moyenne (m)",
                        yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
                        height=max(340, top_n_d * 22),
                        margin=dict(l=0, r=80, t=40, b=0),
                        showlegend=False
                    )
                    st.plotly_chart(fig_pd, use_container_width=True)

                    st.markdown("""
                    <div style="display:flex;gap:1.5rem;margin-top:-0.5rem;margin-bottom:1rem;font-size:0.82rem;">
                        <span style="color:#ef4444;">● % Danger ≥ 30% — Risque élevé</span>
                        <span style="color:#f59e0b;">● % Danger ≥ 15% — Risque modéré</span>
                        <span style="color:#10b981;">● % Danger &lt; 15% — Risque faible</span>
                    </div>
                    """, unsafe_allow_html=True)

                    df_show_d = df_plage_d[["NOM_PLAGE","NOM_WILAYA","avg_hsv","max_hsv","pct_danger","avg_ws","pct_vent_fort","pct_ok","pct_dessal","nb"]].copy()
                    df_show_d.columns = ["Plage","Wilaya","HSV Moy (m)","HSV Max (m)","% Danger","Vent Moy (m/s)","% Vent fort","% Aqua OK","% Dessal OK","Nb obs"]
                    st.dataframe(df_show_d, use_container_width=True, hide_index=True)

                else:
                    # Vue par distance : un graphique par zone sélectionnée
                    for dist_val in dist_select:
                        dist_lbl = DISTANCE_LABELS[dist_val]
                        c_dist   = list(DISTANCE_COLORS.values())[(dist_val - 1) % 4]

                        df_sub = (
                            df_dm[df_dm["DISTANCE"] == dist_val]
                            .sort_values(sort_col_dist, ascending=False)
                            .head(top_n_d)
                        )
                        if df_sub.empty:
                            continue

                        st.markdown(f"""
                        <div style="margin:1rem 0 0.4rem;font-size:0.95rem;font-weight:600;color:{c_dist};">
                            {dist_lbl}
                        </div>
                        """, unsafe_allow_html=True)

                        bar_colors_sub = df_sub["pct_danger"].apply(
                            lambda v: "#ef4444" if v >= 30 else ("#f59e0b" if v >= 15 else "#10b981")
                        )
                        fig_sub = go.Figure()
                        fig_sub.add_trace(go.Bar(
                            x=df_sub["avg_hsv"],
                            y=df_sub["NOM_PLAGE"],
                            orientation="h",
                            marker_color=bar_colors_sub,
                            text=df_sub["avg_hsv"].map(lambda v: f"{v:.3f} m"),
                            textposition="outside",
                            customdata=df_sub[["NOM_WILAYA","pct_danger","max_hsv","avg_ws"]].values,
                            hovertemplate=(
                                "<b>%{y}</b><br>"
                                "Wilaya : %{customdata[0]}<br>"
                                "HSV Moy : %{x:.3f} m<br>"
                                "HSV Max : %{customdata[2]:.2f} m<br>"
                                "% Danger : %{customdata[1]:.1f}%<br>"
                                "Vent Moy : %{customdata[3]:.1f} m/s<extra></extra>"
                            )
                        ))
                        apply_theme(fig_sub)
                        fig_sub.update_layout(
                            title=f"Top {top_n_d} plages · {dist_lbl} — {sort_by_d.replace(' ↓','')}",
                            xaxis_title="HSV Moyenne (m)",
                            yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
                            height=max(300, len(df_sub) * 22),
                            margin=dict(l=0, r=80, t=40, b=0),
                            showlegend=False
                        )
                        st.plotly_chart(fig_sub, use_container_width=True)

                    st.markdown("""
                    <div style="display:flex;gap:1.5rem;margin-top:0.2rem;margin-bottom:1rem;font-size:0.82rem;">
                        <span style="color:#ef4444;">● % Danger ≥ 30% — Risque élevé</span>
                        <span style="color:#f59e0b;">● % Danger ≥ 15% — Risque modéré</span>
                        <span style="color:#10b981;">● % Danger &lt; 15% — Risque faible</span>
                    </div>
                    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style="text-align:center;padding:1rem 0;font-size:0.75rem;color:var(--text-muted);">
    Système HSV · Côtes Algériennes · 1985–2023 &nbsp;·&nbsp;
    Données ERA5 (ECMWF) &nbsp;·&nbsp;
    LSTM + Transfer Learning &nbsp;·&nbsp;
    Powered by DuckDB + Streamlit
</div>
""", unsafe_allow_html=True)