import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import io, os, time
from datetime import datetime
import base64

st.set_page_config(
    page_title="FireSense AI · Wildfire Intelligence",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════
# NASA COMMAND CENTER CSS
# ═══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Space+Grotesk:wght@300;400;500;600;700&family=Share+Tech+Mono&display=swap');

:root {
  --nasa-blue:#0b3d91; --nasa-red:#fc3d21;
  --sky:#4db8ff;  --sky2:#00d4ff;
  --bg-0:#03050d; --bg-1:#070c18; --bg-2:#0a1020; --bg-3:#0d1428; --bg-4:#111930;
  --r-low:#00e5a0; --r-low2:#00ffb3;
  --r-med:#ffb020; --r-med2:#ffd060;
  --r-high:#ff2d20; --r-high2:#ff6050;
  --t-1:#e8edf8; --t-2:#7a90b8; --t-3:#3a4a6a;
  --b-1:rgba(77,184,255,0.14); --b-2:rgba(77,184,255,0.28); --b-h:rgba(77,184,255,0.50);
}

*,*::before,*::after{box-sizing:border-box;}
html,body,.stApp{background:var(--bg-0)!important;color:var(--t-1)!important;font-family:'Space Grotesk',sans-serif!important;}
.block-container{padding:1.2rem 2rem 2rem!important;max-width:1600px!important;}
.stApp::before{content:'';position:fixed;inset:0;z-index:0;
  background:radial-gradient(ellipse at 20% 20%,rgba(11,61,145,0.10) 0%,transparent 60%),
             radial-gradient(ellipse at 80% 80%,rgba(252,61,33,0.06) 0%,transparent 55%),
             radial-gradient(ellipse at 50% 50%,rgba(0,212,255,0.04) 0%,transparent 70%);
  pointer-events:none;}

/* SIDEBAR */
section[data-testid="stSidebar"]{background:linear-gradient(170deg,#060a15 0%,#08101e 50%,#060c16 100%)!important;border-right:1px solid var(--b-1)!important;}
section[data-testid="stSidebar"]>div{padding-top:0!important;}
.sb-logo{background:linear-gradient(135deg,rgba(11,61,145,0.3),rgba(0,212,255,0.1));border-bottom:1px solid var(--b-1);padding:20px 16px 16px;text-align:center;margin-bottom:8px;}
.sb-logo-title{font-family:'Orbitron',monospace;font-size:1.05rem;font-weight:900;letter-spacing:3px;background:linear-gradient(90deg,var(--sky),var(--sky2));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.sb-logo-sub{font-family:'Share Tech Mono',monospace;font-size:0.58rem;color:var(--t-3);letter-spacing:3px;text-transform:uppercase;margin-top:3px;}
.sb-stats{background:rgba(77,184,255,0.04);border:1px solid var(--b-1);border-radius:10px;padding:14px 12px;margin:8px;font-family:'Share Tech Mono',monospace;}
.sb-stat-row{display:flex;justify-content:space-between;align-items:center;padding:4px 0;font-size:0.68rem;color:var(--t-3);border-bottom:1px solid rgba(77,184,255,0.06);}
.sb-stat-row:last-child{border-bottom:none;}
.sb-stat-val{color:var(--sky);font-weight:600;}
.sb-alert-badge{background:rgba(255,45,32,0.12);border:1px solid rgba(255,45,32,0.35);border-radius:10px;padding:12px;margin:8px;text-align:center;}
.sb-alert-num{font-family:'Orbitron',monospace;font-size:2rem;font-weight:900;line-height:1;}
.sb-alert-lbl{font-family:'Share Tech Mono',monospace;font-size:0.58rem;letter-spacing:2px;margin-top:4px;}

/* SPLASH */
.splash{display:flex;flex-direction:column;align-items:center;justify-content:center;padding:60px 20px;text-align:center;}
.splash-ring{width:120px;height:120px;border-radius:50%;border:2px solid transparent;border-top-color:var(--sky);border-right-color:var(--sky2);animation:spin 1.2s linear infinite;margin:0 auto 24px;box-shadow:0 0 30px rgba(0,212,255,0.3),inset 0 0 30px rgba(0,212,255,0.05);}
@keyframes spin{to{transform:rotate(360deg);}}
.splash-text{font-family:'Orbitron',monospace;font-size:0.85rem;color:var(--sky2);letter-spacing:3px;animation:blink 1.2s ease-in-out infinite alternate;}
@keyframes blink{from{opacity:0.4;}to{opacity:1;}}

/* PAGE HEADER */
.page-header{border-bottom:1px solid var(--b-1);padding-bottom:14px;margin-bottom:24px;}
.page-title{font-family:'Orbitron',monospace;font-size:clamp(1.2rem,2.5vw,1.9rem);font-weight:900;letter-spacing:2px;background:linear-gradient(90deg,#ffffff 0%,var(--sky) 60%,var(--sky2) 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.page-sub{font-family:'Share Tech Mono',monospace;font-size:0.68rem;color:var(--t-3);letter-spacing:3px;text-transform:uppercase;margin-top:4px;}
.sec-label{font-family:'Share Tech Mono',monospace;font-size:0.65rem;letter-spacing:3px;text-transform:uppercase;color:var(--sky);border-left:3px solid var(--sky);padding-left:10px;margin-bottom:14px;opacity:0.85;}

/* KPI */
.kpi-grid{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin-bottom:20px;}
@media(max-width:1100px){.kpi-grid{grid-template-columns:repeat(3,1fr);}}
@media(max-width:700px){.kpi-grid{grid-template-columns:repeat(2,1fr);}}
.kpi{background:var(--bg-2);border:1px solid var(--b-1);border-radius:12px;padding:16px 12px;text-align:center;position:relative;overflow:hidden;transition:transform 0.22s,box-shadow 0.22s,border-color 0.22s;animation:fadeUp 0.5s ease both;}
.kpi::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--nasa-blue),var(--sky),var(--sky2));}
.kpi::after{content:'';position:absolute;inset:0;background:radial-gradient(ellipse at 50% 0%,rgba(77,184,255,0.06) 0%,transparent 70%);pointer-events:none;}
.kpi:hover{transform:translateY(-5px);box-shadow:0 12px 35px rgba(0,212,255,0.12);border-color:var(--b-2);}
.kpi-icon{font-size:1.3rem;margin-bottom:6px;line-height:1;}
.kpi-val{font-family:'Orbitron',monospace;font-size:1.55rem;font-weight:700;color:#fff;line-height:1.1;text-shadow:0 0 20px rgba(77,184,255,0.3);}
.kpi-lbl{font-family:'Share Tech Mono',monospace;font-size:0.56rem;color:var(--t-3);letter-spacing:2px;text-transform:uppercase;margin-top:5px;}
@keyframes fadeUp{from{opacity:0;transform:translateY(16px);}to{opacity:1;transform:translateY(0);}}

/* PANEL */
.panel{background:var(--bg-2);border:1px solid var(--b-1);border-radius:14px;padding:20px;margin-bottom:16px;position:relative;overflow:hidden;animation:fadeUp 0.4s ease both;}
.panel::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--sky),var(--sky2),rgba(77,184,255,0));}

/* RISK */
.risk-high{font-family:'Orbitron',monospace!important;font-size:1.35rem!important;font-weight:900!important;color:var(--r-high)!important;letter-spacing:2px!important;text-shadow:0 0 10px rgba(255,45,32,0.7),0 0 25px rgba(255,45,32,0.3)!important;animation:highGlow 1.8s ease-in-out infinite alternate!important;}
@keyframes highGlow{from{text-shadow:0 0 8px rgba(255,45,32,0.5);}to{text-shadow:0 0 20px rgba(255,45,32,1.0),0 0 40px rgba(255,80,50,0.4);}}
.risk-med{font-family:'Orbitron',monospace!important;font-size:1.35rem!important;font-weight:900!important;color:var(--r-med)!important;letter-spacing:2px!important;text-shadow:0 0 10px rgba(255,176,32,0.6)!important;}
.risk-low{font-family:'Orbitron',monospace!important;font-size:1.35rem!important;font-weight:900!important;color:var(--r-low)!important;letter-spacing:2px!important;text-shadow:0 0 10px rgba(0,229,160,0.55)!important;}

/* ALERTS */
.alert-critical{background:rgba(255,45,32,0.07);border:1px solid rgba(255,45,32,0.35);border-left:4px solid var(--r-high);border-radius:10px;padding:13px 18px;margin:7px 0;color:#ffb5b0;font-size:0.9rem;font-weight:500;display:flex;align-items:center;gap:10px;animation:critGlow 2.5s ease-in-out infinite;}
@keyframes critGlow{0%,100%{box-shadow:-4px 0 0 var(--r-high);}50%{box-shadow:-4px 0 0 var(--r-high2),0 0 20px rgba(255,45,32,0.12);}}
.alert-warning{background:rgba(255,176,32,0.06);border:1px solid rgba(255,176,32,0.30);border-left:4px solid var(--r-med);border-radius:10px;padding:13px 18px;margin:7px 0;color:#ffd98a;font-size:0.9rem;font-weight:500;display:flex;align-items:center;gap:10px;}
.alert-safe{background:rgba(0,229,160,0.05);border:1px solid rgba(0,229,160,0.25);border-left:4px solid var(--r-low);border-radius:10px;padding:13px 18px;margin:7px 0;color:#90f0d0;font-size:0.9rem;font-weight:500;display:flex;align-items:center;gap:10px;}

/* MODEL CARDS */
.model-card{background:var(--bg-3);border:1px solid var(--b-1);border-radius:14px;padding:22px 16px;text-align:center;position:relative;overflow:hidden;transition:transform 0.22s,box-shadow 0.22s;}
.model-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--sky),var(--sky2));}
.model-card:hover{transform:translateY(-5px);box-shadow:0 10px 30px rgba(0,212,255,0.14);border-color:var(--b-2);}
.model-card.best-model{border-color:rgba(0,229,160,0.4);box-shadow:0 0 0 1px rgba(0,229,160,0.15);}
.model-card.best-model::before{background:linear-gradient(90deg,var(--r-low),var(--sky2));}
.model-name{font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:var(--t-3);letter-spacing:2.5px;text-transform:uppercase;margin-bottom:8px;}
.model-acc{font-family:'Orbitron',monospace;font-size:1.8rem;font-weight:700;line-height:1.1;}

/* WIDGETS */
div[data-testid="stMetricValue"]{font-family:'Orbitron',monospace!important;color:#ffffff!important;font-weight:700!important;}
div[data-testid="stMetricLabel"]{font-family:'Share Tech Mono',monospace!important;color:var(--t-3)!important;font-size:0.62rem!important;letter-spacing:1.5px!important;text-transform:uppercase!important;}
.stButton>button{background:linear-gradient(135deg,#0b3d91 0%,#1565c0 50%,#0288d1 100%)!important;color:#fff!important;border:1px solid rgba(77,184,255,0.3)!important;border-radius:8px!important;font-family:'Space Grotesk',sans-serif!important;font-size:0.88rem!important;font-weight:600!important;letter-spacing:1px!important;padding:10px 24px!important;box-shadow:0 4px 16px rgba(11,61,145,0.4)!important;transition:all 0.22s ease!important;}
.stButton>button:hover{background:linear-gradient(135deg,#1565c0 0%,#0288d1 50%,var(--sky2) 100%)!important;box-shadow:0 8px 28px rgba(0,136,209,0.5)!important;transform:translateY(-2px)!important;border-color:rgba(77,184,255,0.5)!important;}
div[data-testid="stSlider"] div[role="slider"]{background:var(--sky2)!important;border:2px solid #fff!important;box-shadow:0 0 10px rgba(0,212,255,0.7)!important;}
div[data-testid="stNumberInput"] input,div[data-testid="stTextInput"] input{background:var(--bg-3)!important;border:1px solid var(--b-1)!important;border-radius:8px!important;color:var(--t-1)!important;font-family:'Space Grotesk',sans-serif!important;transition:border-color 0.2s,box-shadow 0.2s!important;}
div[data-testid="stNumberInput"] input:focus,div[data-testid="stTextInput"] input:focus{border-color:var(--sky)!important;box-shadow:0 0 0 3px rgba(77,184,255,0.15)!important;}
div[data-testid="stSelectbox"]>div>div{background:var(--bg-3)!important;border:1px solid var(--b-1)!important;border-radius:8px!important;color:var(--t-1)!important;}
div[data-testid="stMultiSelect"]>div>div{background:var(--bg-3)!important;border:1px solid var(--b-1)!important;border-radius:8px!important;}
span[data-baseweb="tag"]{background:rgba(77,184,255,0.12)!important;border:1px solid rgba(77,184,255,0.35)!important;border-radius:6px!important;color:var(--sky2)!important;font-family:'Share Tech Mono',monospace!important;font-size:0.72rem!important;}
button[data-baseweb="tab"]{font-family:'Space Grotesk',sans-serif!important;font-size:0.82rem!important;font-weight:500!important;color:var(--t-2)!important;background:transparent!important;transition:color 0.2s!important;padding:8px 16px!important;}
button[data-baseweb="tab"]:hover{color:var(--sky2)!important;}
button[data-baseweb="tab"][aria-selected="true"]{color:var(--sky)!important;font-weight:700!important;border-bottom:2px solid var(--sky)!important;}
div[data-testid="stTabContent"]{border:1px solid var(--b-1)!important;border-top:none!important;border-radius:0 0 12px 12px!important;padding:16px!important;background:rgba(77,184,255,0.02)!important;}
div[data-testid="stDataFrame"]{border:1px solid var(--b-1)!important;border-radius:10px!important;overflow:hidden!important;}
section[data-testid="stSidebar"] div[data-testid="stRadio"]>label{font-family:'Space Grotesk',sans-serif!important;font-size:0.88rem!important;font-weight:500!important;color:var(--t-2)!important;transition:color 0.2s!important;}
section[data-testid="stSidebar"] div[data-testid="stRadio"]>label:hover{color:var(--sky)!important;}
hr{border-color:var(--b-1)!important;margin:16px 0!important;}
div[data-testid="stSpinner"]>div{border-top-color:var(--sky)!important;}
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:var(--bg-0);}
::-webkit-scrollbar-thumb{background:rgba(77,184,255,0.3);border-radius:3px;}
::-webkit-scrollbar-thumb:hover{background:var(--sky);}
div[data-testid="stCheckbox"] label{color:var(--t-1)!important;font-family:'Space Grotesk',sans-serif!important;font-size:0.88rem!important;}
#MainMenu,footer,header{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ── CHART THEME ──────────────────────────────────────────────
CHART = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
             font=dict(color='#7a90b8', family='Space Grotesk'),
             margin=dict(t=36, b=20, l=10, r=10))
GRID  = dict(gridcolor='rgba(77,184,255,0.08)', zerolinecolor='rgba(77,184,255,0.15)')
RISK_COLORS = {'High':'#ff2d20','Medium':'#ffb020','Low':'#00e5a0'}
CS_FIRE = 'YlOrRd'
CS_BLUE = [[0,'#0b3d91'],[0.5,'#0288d1'],[1,'#00d4ff']]

def apply_chart(fig, height=None):
    fig.update_layout(**CHART)
    if height: fig.update_layout(height=height)
    fig.update_xaxes(**GRID)
    fig.update_yaxes(**GRID)
    fig.update_layout(legend=dict(bgcolor='rgba(10,16,32,0.7)',
                                   bordercolor='rgba(77,184,255,0.2)',
                                   borderwidth=1, font=dict(color='#7a90b8', size=11)))
    return fig

# ═══════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════
@st.cache_data
def load_and_process(file_bytes=None, filepath=None):
    df = pd.read_csv(filepath) if filepath else pd.read_csv(io.BytesIO(file_bytes))
    df['acq_date']       = pd.to_datetime(df['acq_date'])
    df['risk_label']     = df['confidence'].apply(lambda x: 'High' if x>=70 else ('Medium' if x>=40 else 'Low'))
    df['risk_num']       = df['confidence'].apply(lambda x: 2 if x>=70 else (1 if x>=40 else 0))
    df['delta_b']        = df['brightness'] - df['bright_t31']
    df['month']          = df['acq_date'].dt.month
    df['week']           = df['acq_date'].dt.isocalendar().week.astype(int)
    df['fire_intensity'] = pd.cut(df['frp'], bins=[-1,10,50,100,500,99999],
                                   labels=['Very Low','Low','Medium','High','Extreme'])
    return df

df = None
for name in ['fire_archive.csv','1772814556091_fire_archive.csv']:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    if os.path.exists(path):
        df = load_and_process(filepath=path); break

if df is None:
    st.markdown("""
    <div class="splash">
      <div style="font-family:'Orbitron',monospace;font-size:2rem;font-weight:900;
        background:linear-gradient(90deg,#4db8ff,#00d4ff);-webkit-background-clip:text;
        -webkit-text-fill-color:transparent;background-clip:text;letter-spacing:4px;margin-bottom:8px">
        🔥 FIRESENSE AI</div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;color:#3a4a6a;
        letter-spacing:4px;margin-bottom:40px">WILDFIRE INTELLIGENCE SYSTEM</div>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<p class="sec-label">📂 DATA SOURCE</p>', unsafe_allow_html=True)
    st.info("Apni **fire_archive.csv** file upload karo — dashboard shuru hoga")
    up = st.file_uploader("CSV File Choose Karo", type=["csv"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    if up:
        df = load_and_process(file_bytes=up.read()); st.rerun()
    else:
        st.stop()

# ── SPLASH ───────────────────────────────────────────────────
if 'loaded' not in st.session_state:
    splash = st.empty()
    splash.markdown("""
    <div class="splash">
      <div class="splash-ring"></div>
      <div style="font-family:'Orbitron',monospace;font-size:1.8rem;font-weight:900;
        background:linear-gradient(90deg,#4db8ff,#00d4ff);-webkit-background-clip:text;
        -webkit-text-fill-color:transparent;background-clip:text;letter-spacing:4px;margin-bottom:8px">
        FIRESENSE AI</div>
      <div class="splash-text">INITIALIZING WILDFIRE INTELLIGENCE SYSTEM...</div>
      <div style="margin-top:16px;font-family:'Share Tech Mono',monospace;font-size:0.62rem;
        color:#3a4a6a;letter-spacing:3px">NASA MODIS · AUSTRALIA 2019 · ML MODELS LOADING</div>
    </div>""", unsafe_allow_html=True)
    time.sleep(1.5)
    splash.empty()
    st.session_state['loaded'] = True

# ── TRAIN MODELS ─────────────────────────────────────────────
@st.cache_resource
def train_models(_df):
    feats = ['brightness','frp','confidence','bright_t31','scan','track','delta_b']
    X, y  = _df[feats], _df['risk_num']
    sc    = StandardScaler(); Xs = sc.fit_transform(X)
    Xtr,Xte,ytr,yte = train_test_split(Xs, y, test_size=0.2, random_state=42)
    mods = {'Logistic Regression': LogisticRegression(max_iter=1000),
            'Random Forest':       RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
            'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42)}
    res = {}
    for nm, m in mods.items():
        m.fit(Xtr, ytr); p = m.predict(Xte)
        res[nm] = {'model':m, 'acc':accuracy_score(yte,p), 'cm':confusion_matrix(yte,p),
                   'report':classification_report(yte,p,target_names=['Low','Medium','High'],output_dict=True)}
    return res, sc, feats

with st.spinner("🛰 Satellite data processing & AI model training..."):
    model_res, scaler, FEATS = train_models(df)

best_name = max(model_res, key=lambda k: model_res[k]['acc'])

# ── ALERTS ───────────────────────────────────────────────────
def get_alerts(df):
    alerts, r = [], df.tail(500)
    hi = len(r[r['risk_label']=='High'])
    if hi > 100: alerts.append(('CRITICAL', f"🚨 {hi} high-risk events in recent data!"))
    ex = len(r[r['frp']>500])
    if ex > 0:   alerts.append(('CRITICAL', f"🔴 {ex} EXTREME fires detected (FRP > 500 MW)!"))
    if r['frp'].mean() > 60: alerts.append(('WARNING', f"⚠️ Avg FRP critically elevated: {r['frp'].mean():.1f} MW"))
    nf = len(r[r['daynight']=='N'])
    if nf > 50:  alerts.append(('WARNING', f"🌙 {nf} nighttime fires detected!"))
    if not alerts: alerts.append(('SAFE','✅ All systems nominal. No critical alerts.'))
    return alerts

alerts     = get_alerts(df)
critical_n = sum(1 for a in alerts if a[0]=='CRITICAL')

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
      <div style="font-size:2rem;margin-bottom:6px">🔥</div>
      <div class="sb-logo-title">FIRESENSE AI</div>
      <div class="sb-logo-sub">Wildfire Intelligence System</div>
    </div>""", unsafe_allow_html=True)

    page = st.radio("", [
        "🏠  Command Center",
        "🗺   Fire Hotspot Map",
        "📈  Trend Analysis",
        "🌐  3D Visualization",
        "🚨  Alert System",
        "🧠  ML Model Lab",
        "🎯  Predict Risk",
        "📄  Download Report"
    ], label_visibility="collapsed")

    st.markdown("---")
    ac = '#ff2d20' if critical_n > 0 else '#00e5a0'
    st.markdown(f"""
    <div class="sb-alert-badge" style="border-color:rgba({'255,45,32' if critical_n>0 else '0,229,160'},0.35)">
      <div class="sb-alert-num" style="color:{ac}">{len(alerts)}</div>
      <div class="sb-alert-lbl" style="color:{ac}">{'🔴 CRITICAL ALERTS' if critical_n>0 else '🟢 SYSTEM NOMINAL'}</div>
    </div>
    <div class="sb-stats">
      <div class="sb-stat-row"><span>RECORDS</span><span class="sb-stat-val">{len(df):,}</span></div>
      <div class="sb-stat-row"><span>BEST MODEL</span><span class="sb-stat-val" style="font-size:0.6rem">{best_name.split()[0].upper()}</span></div>
      <div class="sb-stat-row"><span>ACCURACY</span><span class="sb-stat-val">{model_res[best_name]['acc']*100:.1f}%</span></div>
      <div class="sb-stat-row"><span>PERIOD</span><span class="sb-stat-val">AUG–SEP 2019</span></div>
      <div class="sb-stat-row"><span>SOURCE</span><span class="sb-stat-val">NASA MODIS</span></div>
    </div>""", unsafe_allow_html=True)

# ── HELPERS ──────────────────────────────────────────────────
def page_header(title, sub):
    st.markdown(f'<div class="page-header"><div class="page-title">{title}</div><div class="page-sub">{sub}</div></div>', unsafe_allow_html=True)

def sec(label):
    st.markdown(f'<p class="sec-label">{label}</p>', unsafe_allow_html=True)

def alert_box(atype, msg):
    cls  = {'CRITICAL':'alert-critical','WARNING':'alert-warning','SAFE':'alert-safe'}[atype]
    icon = {'CRITICAL':'🚨','WARNING':'⚠️','SAFE':'✅'}[atype]
    st.markdown(f'<div class="{cls}"><span style="font-size:1.1rem">{icon}</span><span><strong>[{atype}]</strong> {msg}</span></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# PAGE 1 — COMMAND CENTER
# ═══════════════════════════════════════════════════════════
if page == "🏠  Command Center":
    page_header("🔥 FireSense AI — Command Center",
                "Australia Wildfire Intelligence · NASA MODIS Satellite · 2019")

    kpis = [
        ("🔥", f"{len(df):,}",                             "TOTAL EVENTS"),
        ("📡", f"{df['confidence'].mean():.0f}%",          "AVG CONFIDENCE"),
        ("⚡", f"{df['frp'].mean():.1f}",                  "AVG FRP (MW)"),
        ("🚨", f"{len(df[df['risk_label']=='High']):,}",   "HIGH RISK"),
        ("🌙", f"{len(df[df['daynight']=='N']):,}",        "NIGHT FIRES"),
        ("🏆", f"{model_res[best_name]['acc']*100:.1f}%",  "ML ACCURACY"),
    ]
    html = '<div class="kpi-grid">'
    for i,(icon,val,lbl) in enumerate(kpis):
        html += f'<div class="kpi" style="animation-delay:{i*0.08}s"><div class="kpi-icon">{icon}</div><div class="kpi-val">{val}</div><div class="kpi-lbl">{lbl}</div></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

    sec("🚨 LIVE ALERT STATUS")
    for atype, amsg in alerts:
        alert_box(atype, amsg)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        sec("📊 RISK LEVEL DISTRIBUTION")
        rc  = df['risk_label'].value_counts()
        fig = px.pie(values=rc.values, names=rc.index, hole=0.58,
                     color=rc.index, color_discrete_map=RISK_COLORS)
        fig.update_traces(textfont_color='white', textfont_size=12,
                          marker=dict(line=dict(color='#06060e', width=2)))
        apply_chart(fig, 320)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sec("🌡 BRIGHTNESS vs FRP DENSITY")
        fig = px.density_heatmap(df.sample(min(8000,len(df))), x='brightness', y='frp',
                                  nbinsx=35, nbinsy=35, color_continuous_scale=CS_FIRE)
        apply_chart(fig, 320)
        fig.update_layout(yaxis_range=[0,300])
        st.plotly_chart(fig, use_container_width=True)

    sec("📈 DAILY FIRE ACTIVITY TIMELINE")
    daily = df.groupby('acq_date').agg(
        count=('frp','count'), avg_frp=('frp','mean'),
        high_risk=('risk_num', lambda x:(x==2).sum())).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=daily['acq_date'], y=daily['count'],     name='Total Fires',
                         marker_color='rgba(77,144,255,0.45)', marker_line_width=0))
    fig.add_trace(go.Bar(x=daily['acq_date'], y=daily['high_risk'], name='High Risk',
                         marker_color='rgba(255,45,32,0.75)',  marker_line_width=0))
    fig.add_trace(go.Scatter(x=daily['acq_date'], y=daily['avg_frp'], name='Avg FRP',
                              yaxis='y2', line=dict(color='#ffb020', width=2.5)))
    apply_chart(fig, 330)
    fig.update_layout(barmode='overlay',
                       yaxis =dict(title='Fire Count',   **GRID),
                       yaxis2=dict(title='Avg FRP (MW)', overlaying='y', side='right', **GRID),
                       xaxis =dict(**GRID))
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# PAGE 2 — FIRE MAP
# ═══════════════════════════════════════════════════════════
elif page == "🗺   Fire Hotspot Map":
    page_header("🗺 Fire Hotspot Map", "Australia · Real-time Wildfire Detection · NASA FIRMS")

    c1,c2,c3,c4 = st.columns([2,1,1,1])
    with c1: risk_f    = st.multiselect("Risk Level Filter", ['High','Medium','Low'], default=['High','Medium','Low'])
    with c2: n_pts     = st.slider("Max Points", 500, 8000, 3000, 500)
    with c3: mstyle    = st.selectbox("Map Style", ["carto-darkmatter","open-street-map","stamen-terrain"])
    with c4: show_heat = st.checkbox("Heatmap Mode", value=False)

    fdf = df[df['risk_label'].isin(risk_f)].sample(min(n_pts,len(df)), random_state=42)
    if show_heat:
        fig = px.density_mapbox(fdf, lat='latitude', lon='longitude', z='frp',
                                 radius=10, zoom=3.5, center={"lat":-25,"lon":134},
                                 mapbox_style=mstyle, color_continuous_scale=CS_FIRE)
    else:
        fig = px.scatter_mapbox(fdf, lat='latitude', lon='longitude',
                                 color='risk_label', size='frp', size_max=20, opacity=0.85,
                                 color_discrete_map=RISK_COLORS, hover_name='acq_date',
                                 hover_data={'brightness':':.1f','frp':':.1f','confidence':True,
                                             'risk_label':True,'latitude':False,'longitude':False},
                                 zoom=3.5, center={"lat":-25,"lon":134}, mapbox_style=mstyle)
    fig.update_layout(height=620, **CHART, legend=dict(font=dict(color='#7a90b8')))
    st.plotly_chart(fig, use_container_width=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Points Shown",  f"{len(fdf):,}")
    c2.metric("Avg FRP",       f"{fdf['frp'].mean():.1f} MW")
    c3.metric("Max FRP",       f"{fdf['frp'].max():.1f} MW")
    c4.metric("High Risk %",   f"{len(fdf[fdf['risk_label']=='High'])/len(fdf)*100:.1f}%")

# ═══════════════════════════════════════════════════════════
# PAGE 3 — TRENDS
# ═══════════════════════════════════════════════════════════
elif page == "📈  Trend Analysis":
    page_header("📈 Advanced Trend Analysis", "Statistical Deep-Dive · Fire Patterns · 2019")

    daily = df.groupby('acq_date').agg(
        count=('frp','count'), avg_frp=('frp','mean'), max_frp=('frp','max'),
        avg_conf=('confidence','mean'), high=('risk_num', lambda x:(x==2).sum())
    ).reset_index()

    t1,t2,t3,t4 = st.tabs(["📅 Daily Trends","🔬 Statistical Analysis","🛰 Satellite","⏰ Day vs Night"])

    with t1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily['acq_date'], y=daily['count'], fill='tozeroy', name='Fire Events',
                                  line=dict(color='#4db8ff',width=2.5), fillcolor='rgba(77,184,255,0.10)'))
        fig.add_trace(go.Scatter(x=daily['acq_date'], y=daily['high'],  fill='tozeroy', name='High Risk',
                                  line=dict(color='#ff2d20',width=2),   fillcolor='rgba(255,45,32,0.12)'))
        apply_chart(fig, 320)
        st.plotly_chart(fig, use_container_width=True)
        c1,c2 = st.columns(2)
        with c1:
            fig = px.bar(daily, x='acq_date', y='avg_frp', color='avg_frp',
                          color_continuous_scale=CS_FIRE, title="Avg FRP Per Day")
            apply_chart(fig, 280); st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.line(daily, x='acq_date', y='avg_conf', color_discrete_sequence=['#4db8ff'],
                           title="Daily Detection Confidence")
            fig.add_hline(y=70, line_dash="dash", line_color="#ff2d20",
                          annotation_text="High Risk Threshold (70%)", annotation_font_color="#ff2d20")
            apply_chart(fig, 280); st.plotly_chart(fig, use_container_width=True)

    with t2:
        c1,c2 = st.columns(2)
        with c1:
            sec("📐 DESCRIPTIVE STATISTICS")
            st.dataframe(df[['brightness','frp','confidence','bright_t31','delta_b']].describe().round(2),
                          use_container_width=True)
        with c2:
            sec("🔗 CORRELATION HEATMAP")
            corr = df[['brightness','frp','confidence','bright_t31','scan','track','delta_b']].corr()
            fig  = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r')
            apply_chart(fig, 350); st.plotly_chart(fig, use_container_width=True)
        sec("📉 BRIGHTNESS vs FRP — REGRESSION ANALYSIS")
        samp = df.sample(min(5000,len(df)))
        fig  = px.scatter(samp, x='brightness', y='frp', color='risk_label', opacity=0.35,
                           color_discrete_map=RISK_COLORS, trendline='ols',
                           trendline_scope='overall', trendline_color_override='#00d4ff')
        apply_chart(fig, 380); fig.update_layout(yaxis_range=[0,400])
        st.plotly_chart(fig, use_container_width=True)

    with t3:
        sat = df.groupby('satellite').agg(Count=('frp','count'), Avg_FRP=('frp','mean'),
                                           Max_FRP=('frp','max'), Avg_Conf=('confidence','mean')).round(2)
        st.dataframe(sat, use_container_width=True)
        fig = px.box(df, x='satellite', y='frp', color='satellite',
                      color_discrete_sequence=['#4db8ff','#ffb020'], title="FRP by Satellite")
        apply_chart(fig, 360); fig.update_layout(yaxis_range=[0,400])
        st.plotly_chart(fig, use_container_width=True)

    with t4:
        dn = df.groupby('daynight').agg(Count=('frp','count'), Avg_FRP=('frp','mean'),
                                         Avg_Conf=('confidence','mean')).round(2)
        dn.index = ['Daytime' if x=='D' else 'Nighttime' for x in dn.index]
        st.dataframe(dn, use_container_width=True)
        fig = px.violin(df, y='frp', x='daynight', color='daynight', box=True,
                         color_discrete_map={'D':'#ffb020','N':'#4db8ff'}, title="FRP: Day vs Night")
        apply_chart(fig, 360); fig.update_layout(yaxis_range=[0,300])
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# PAGE 4 — 3D VISUALIZATION
# ═══════════════════════════════════════════════════════════
elif page == "🌐  3D Visualization":
    page_header("🌐 3D Fire Visualization", "Immersive Satellite Data · 3D Scatter · Globe · Animation")

    t1,t2,t3 = st.tabs(["🌐 3D Scatter","🌀 Globe View","🔥 Fire Animation"])

    with t1:
        samp = df.sample(min(4000,len(df)))
        fig  = px.scatter_3d(samp, x='longitude', y='latitude', z='brightness',
                              color='frp', size='frp', size_max=9, color_continuous_scale=CS_FIRE,
                              opacity=0.75, hover_data=['confidence','risk_label'])
        fig.update_layout(height=640, **CHART,
                          scene=dict(bgcolor='rgba(6,8,20,1)',
                                     xaxis=dict(title='Longitude', gridcolor='rgba(77,184,255,0.12)', color='#3a4a6a'),
                                     yaxis=dict(title='Latitude',  gridcolor='rgba(77,184,255,0.12)', color='#3a4a6a'),
                                     zaxis=dict(title='Brightness (K)', gridcolor='rgba(77,184,255,0.12)', color='#3a4a6a')))
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        samp2 = df.sample(min(3000,len(df)))
        fig   = go.Figure(go.Scattergeo(
            lon=samp2['longitude'], lat=samp2['latitude'], mode='markers',
            marker=dict(size=samp2['frp'].clip(0,200)/12+2, color=samp2['frp'],
                        colorscale=CS_FIRE, opacity=0.8,
                        colorbar=dict(title='FRP MW', tickfont=dict(color='#7a90b8'))),
            text=samp2.apply(lambda r: f"FRP:{r['frp']:.0f} | Conf:{r['confidence']}%", axis=1),
            hoverinfo='text'))
        fig.update_layout(
            geo=dict(projection_type='orthographic', showland=True, landcolor='#0d1428',
                     showocean=True, oceancolor='#06080f', showcountries=True,
                     countrycolor='rgba(77,184,255,0.25)', showcoastlines=True,
                     coastlinecolor='rgba(77,184,255,0.4)', bgcolor='rgba(0,0,0,0)',
                     center=dict(lat=-25,lon=134), projection_rotation=dict(lat=-25,lon=134)),
            **CHART, height=640)
        st.plotly_chart(fig, use_container_width=True)

    with t3:
        sec("🔥 FIRE SPREAD ANIMATION — TEMPORAL PROGRESSION")
        da = df.copy()
        da['date_str'] = da['acq_date'].dt.strftime('%Y-%m-%d')
        samp3 = da.sample(min(5000,len(da))).sort_values('date_str')
        fig = px.scatter_mapbox(samp3, lat='latitude', lon='longitude',
                                 color='risk_label', size='frp', size_max=15,
                                 color_discrete_map=RISK_COLORS, animation_frame='date_str',
                                 zoom=3.5, center={"lat":-25,"lon":134}, mapbox_style="carto-darkmatter")
        fig.update_layout(height=600, **CHART, legend=dict(font=dict(color='#7a90b8')))
        fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 700
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# PAGE 5 — ALERT SYSTEM
# ═══════════════════════════════════════════════════════════
elif page == "🚨  Alert System":
    page_header("🚨 Alert System", "Real-time Fire Risk Monitoring · Threshold Management")

    sec("📡 ACTIVE SYSTEM ALERTS")
    for atype, amsg in alerts:
        alert_box(atype, amsg)

    st.markdown("---")
    sec("⚙️ CUSTOM ALERT THRESHOLD CONFIGURATOR")
    c1,c2 = st.columns(2)
    with c1: t_frp  = st.slider("FRP Alert Threshold (MW)", 50, 500, 200, 10)
    with c2: t_conf = st.slider("Confidence Threshold (%)", 50, 100,  80,  5)

    custom = df[(df['frp']>t_frp) & (df['confidence']>t_conf)]
    st.metric("Events matching your criteria", f"{len(custom):,}")
    if len(custom) > 0:
        alert_box('CRITICAL', f"{len(custom):,} high-priority events match your alert criteria!")
        fig = px.scatter_mapbox(custom.sample(min(1000,len(custom))),
                                 lat='latitude', lon='longitude', color='frp',
                                 color_continuous_scale=CS_FIRE, size='frp', size_max=15,
                                 zoom=3.5, center={"lat":-25,"lon":134}, mapbox_style="carto-darkmatter")
        fig.update_layout(height=450, **CHART)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    sec("🔝 TOP 20 EXTREME FIRE EVENTS")
    top = df.nlargest(20,'frp')[['acq_date','latitude','longitude','frp','brightness','confidence','risk_label']].copy()
    top['acq_date'] = top['acq_date'].dt.strftime('%Y-%m-%d')
    st.dataframe(top, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# PAGE 6 — ML MODEL LAB
# ═══════════════════════════════════════════════════════════
elif page == "🧠  ML Model Lab":
    page_header("🧠 ML Model Laboratory", "3 Algorithms Compared · Feature Analysis · Syllabus Mapping")

    sec("🏆 MODEL PERFORMANCE COMPARISON")
    c1,c2,c3 = st.columns(3)
    for col,(nm,mr) in zip([c1,c2,c3], model_res.items()):
        with col:
            is_best   = (nm==best_name)
            acc_color = '#00e5a0' if is_best else '#4db8ff'
            badge     = '🥇 BEST MODEL' if is_best else '✅ TRAINED'
            extra_cls = 'best-model' if is_best else ''
            st.markdown(f"""
            <div class="model-card {extra_cls}">
              <div class="model-name">{nm.upper()}</div>
              <div class="model-acc" style="color:{acc_color}">{mr['acc']*100:.1f}%</div>
              <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:{acc_color};margin-top:6px">{badge}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    c1,c2 = st.columns(2)
    with c1:
        sec("📊 ACCURACY COMPARISON")
        names  = list(model_res.keys())
        accs   = [model_res[n]['acc']*100 for n in names]
        colors = ['#00e5a0' if n==best_name else '#4db8ff' for n in names]
        fig = go.Figure(go.Bar(x=names, y=accs, marker_color=colors, marker_line_width=0,
                                text=[f"{a:.1f}%" for a in accs], textposition='outside',
                                textfont=dict(color='white', size=13)))
        apply_chart(fig, 300); fig.update_layout(yaxis_range=[0,115])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        sec("🔍 CONFUSION MATRIX")
        sel = st.selectbox("Select Model", list(model_res.keys()))
        cm  = model_res[sel]['cm']
        fig = px.imshow(cm, text_auto=True, color_continuous_scale=CS_BLUE,
                         x=['Low','Medium','High'], y=['Low','Medium','High'],
                         labels=dict(x="Predicted", y="Actual"))
        apply_chart(fig, 300); st.plotly_chart(fig, use_container_width=True)

    sec("📌 FEATURE IMPORTANCE — RANDOM FOREST")
    rf = model_res['Random Forest']['model']
    fi = pd.DataFrame({'Feature':FEATS,'Importance':rf.feature_importances_}).sort_values('Importance')
    fig = px.bar(fi, x='Importance', y='Feature', orientation='h',
                  color='Importance', color_continuous_scale=CS_BLUE)
    apply_chart(fig, 280); st.plotly_chart(fig, use_container_width=True)

    sec("📚 SYLLABUS CONCEPTS MAPPING")
    st.markdown("""
| Syllabus Topic | Project Application |
|----------------|---------------------|
| **Introduction to AI** | AI-powered wildfire risk prediction system |
| **Fundamentals of Statistics** | Mean, std deviation of FRP, brightness |
| **Descriptive Statistics** | Summary stats, histograms, box plots |
| **Inferential Statistics** | Sample data → population prediction |
| **Hypothesis Testing** | Confidence ≥ 70% = High Risk threshold |
| **Statistical Analysis** | Correlation matrix, OLS regression trendline |
| **Concept of ML** | Pattern learning from 36,000+ records |
| **Types of ML Algorithms** | Logistic Regression, Random Forest, Gradient Boosting |
| **Components of ML** | Features, Labels, Train/Test split, Evaluation metrics |
    """)

# ═══════════════════════════════════════════════════════════
# PAGE 7 — PREDICT RISK
# ═══════════════════════════════════════════════════════════
elif page == "🎯  Predict Risk":
    page_header("🎯 Fire Risk Predictor", "Enter Satellite Sensor Readings · AI Predicts Risk Level")

    sel_m  = st.selectbox("🤖 Select ML Model", list(model_res.keys()),
                           index=list(model_res.keys()).index(best_name))
    active = model_res[sel_m]['model']

    col_a,col_b,col_c = st.columns(3)
    with col_a:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        sec("🌡 SENSOR INPUTS — GROUP A")
        brightness = st.number_input("Brightness (Kelvin)", 280.0,500.0,320.0,0.5)
        frp_in     = st.number_input("FRP (MW)",             0.0,1000.0, 45.0,0.5)
        confidence = st.slider("Confidence (%)", 0, 100, 65)
        st.markdown('</div>', unsafe_allow_html=True)
    with col_b:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        sec("📡 SENSOR INPUTS — GROUP B")
        bright_t31 = st.number_input("Brightness T31 (K)", 270.0,350.0,298.0,0.5)
        scan       = st.number_input("Scan",                0.5,  5.0,  1.0, 0.1)
        track      = st.number_input("Track",               0.5,  3.0,  1.0, 0.1)
        st.markdown('</div>', unsafe_allow_html=True)
    with col_c:
        db        = brightness - bright_t31
        frp_cat   = 'EXTREME' if frp_in>200 else 'HIGH' if frp_in>100 else 'MEDIUM' if frp_in>30 else 'LOW'
        det_qual  = 'EXCELLENT' if confidence>80 else 'GOOD' if confidence>60 else 'FAIR'
        db_color  = '#ff2d20' if db>30 else '#ffb020' if db>15 else '#00e5a0'
        frp_color = '#ff2d20' if frp_in>100 else '#ffb020' if frp_in>30 else '#00e5a0'
        st.markdown(f"""
        <div class="panel">
          <p class="sec-label">⚙️ COMPUTED VALUES</p>
          <div style="display:flex;flex-direction:column;gap:14px;margin-top:8px">
            <div>
              <div style="font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:#3a4a6a;letter-spacing:2px">DELTA BRIGHTNESS</div>
              <div style="font-family:'Orbitron',monospace;font-size:1.3rem;font-weight:700;color:{db_color}">{db:+.1f} K</div>
            </div>
            <div>
              <div style="font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:#3a4a6a;letter-spacing:2px">FRP INTENSITY</div>
              <div style="font-family:'Orbitron',monospace;font-size:1.3rem;font-weight:700;color:{frp_color}">{frp_cat}</div>
            </div>
            <div>
              <div style="font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:#3a4a6a;letter-spacing:2px">DETECTION QUALITY</div>
              <div style="font-family:'Orbitron',monospace;font-size:1.3rem;font-weight:700;color:#4db8ff">{det_qual}</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("⚡  ANALYZE FIRE RISK", use_container_width=True):
        inp  = np.array([[brightness,frp_in,confidence,bright_t31,scan,track,db]])
        pred = active.predict(scaler.transform(inp))[0]
        prob = active.predict_proba(scaler.transform(inp))[0]
        rmap = {0:'LOW RISK',1:'MEDIUM RISK',2:'HIGH RISK — CRITICAL'}
        emod = {0:'🟢',1:'🟡',2:'🔴'}
        clr  = {0:'#00e5a0',1:'#ffb020',2:'#ff2d20'}
        adv  = {0:"Sensor readings are within normal parameters. No immediate action required.",
                1:"Elevated conditions detected. Recommend continued monitoring and standby alert.",
                2:"CRITICAL — High brightness + elevated FRP indicates active wildfire. Immediate response required!"}
        st.markdown(f"""
        <div class="panel" style="text-align:center;border-color:{clr[pred]};box-shadow:0 0 40px {clr[pred]}22;margin-top:12px">
          <div style="font-size:3.5rem;margin-bottom:8px">{emod[pred]}</div>
          <div style="font-family:'Orbitron',monospace;font-size:2rem;font-weight:900;
            color:{clr[pred]};letter-spacing:3px">{rmap[pred]}</div>
          <div style="color:#7a90b8;font-size:0.9rem;margin-top:10px;
            font-family:'Space Grotesk',sans-serif">{adv[pred]}</div>
        </div>""", unsafe_allow_html=True)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Risk Score",   f"{prob[pred]*100:.1f}%")
        c2.metric("FRP Input",    f"{frp_in} MW")
        c3.metric("Δ Brightness", f"{db:+.1f} K")
        c4.metric("Confidence",   f"{confidence}%")

        fig = go.Figure(go.Bar(
            x=['🟢 Low','🟡 Medium','🔴 High'],
            y=[p*100 for p in prob],
            marker_color=[RISK_COLORS['Low'],RISK_COLORS['Medium'],RISK_COLORS['High']],
            marker_line_width=0,
            text=[f'{p*100:.1f}%' for p in prob], textposition='outside',
            textfont=dict(color='white', size=13)))
        apply_chart(fig, 320); fig.update_layout(yaxis_range=[0,115], title=f"Risk Probability — {sel_m}")
        st.plotly_chart(fig, use_container_width=True)

        sec("🔀 ALL MODELS CONSENSUS")
        c1,c2,c3 = st.columns(3)
        for col,(mn,mr) in zip([c1,c2,c3], model_res.items()):
            with col:
                p2  = mr['model'].predict(scaler.transform(inp))[0]
                pr2 = mr['model'].predict_proba(scaler.transform(inp))[0]
                c2x = clr[p2]
                st.markdown(f"""
                <div class="model-card {'best-model' if mn==best_name else ''}">
                  <div class="model-name">{mn}</div>
                  <div style="font-family:'Orbitron',monospace;font-size:1.3rem;font-weight:900;color:{c2x}">{['LOW','MEDIUM','HIGH'][p2]}</div>
                  <div style="color:#3a4a6a;font-size:0.7rem;font-family:'Share Tech Mono',monospace;margin-top:6px">{max(pr2)*100:.1f}% confidence</div>
                </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# PAGE 8 — DOWNLOAD REPORT
# ═══════════════════════════════════════════════════════════
elif page == "📄  Download Report":
    page_header("📄 Analysis Report", "Export Complete Fire Intelligence Report")

    report_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    high_days   = df.groupby('acq_date')['frp'].mean().sort_values(ascending=False).head(5)

    report_text = f"""
╔══════════════════════════════════════════════════════════╗
║       FIRESENSE AI — WILDFIRE INTELLIGENCE REPORT        ║
║          NASA MODIS SATELLITE DATA ANALYSIS              ║
╚══════════════════════════════════════════════════════════╝
Generated  : {report_time}
Dataset    : Australia Wildfire Data (NASA FIRMS MODIS)
Period     : August 1 – September 30, 2019

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  EXECUTIVE SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Total Fire Events   : {len(df):,}
  High Risk Events    : {len(df[df['risk_label']=='High']):,}  ({len(df[df['risk_label']=='High'])/len(df)*100:.1f}%)
  Medium Risk Events  : {len(df[df['risk_label']=='Medium']):,}  ({len(df[df['risk_label']=='Medium'])/len(df)*100:.1f}%)
  Low Risk Events     : {len(df[df['risk_label']=='Low']):,}  ({len(df[df['risk_label']=='Low'])/len(df)*100:.1f}%)
  Days Monitored      : {df['acq_date'].nunique()}
  Daytime Fires       : {len(df[df['daynight']=='D']):,}
  Nighttime Fires     : {len(df[df['daynight']=='N']):,}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STATISTICAL ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  FRP:  Avg={df['frp'].mean():.2f} | Med={df['frp'].median():.2f} | Max={df['frp'].max():.2f} MW
  Brightness: Avg={df['brightness'].mean():.2f} | Max={df['brightness'].max():.2f} K
  Confidence: Avg={df['confidence'].mean():.1f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  MACHINE LEARNING RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    for nm,mr in model_res.items():
        flag = " ◄ BEST" if nm==best_name else ""
        report_text += f"  {nm:<28}: {mr['acc']*100:.2f}%{flag}\n"

    report_text += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n  TOP 5 HIGH RISK DAYS\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    for date,frp_val in high_days.items():
        report_text += f"  {date.strftime('%Y-%m-%d')}  →  {frp_val:.1f} MW avg FRP\n"

    report_text += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n  ACTIVE ALERTS\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    for atype,amsg in alerts:
        report_text += f"  [{atype}] {amsg}\n"
    report_text += "\n╚══════════════════════════════════════════════════════════╝\n"

    sec("📋 REPORT PREVIEW")
    st.code(report_text, language=None)
    st.markdown("---")

    c1,c2 = st.columns(2)
    with c1:
        b64  = base64.b64encode(report_text.encode()).decode()
        href = (f'<a href="data:file/txt;base64,{b64}" download="firesense_report_{datetime.now().strftime("%Y%m%d_%H%M")}.txt" '
                f'style="display:inline-block;background:linear-gradient(135deg,#0b3d91,#0288d1);color:white;'
                f'padding:12px 28px;border-radius:8px;text-decoration:none;font-family:Space Grotesk,sans-serif;'
                f'font-size:0.9rem;font-weight:600;letter-spacing:1px;box-shadow:0 4px 16px rgba(11,61,145,0.4)">📄 Download Report (.txt)</a>')
        st.markdown(href, unsafe_allow_html=True)
    with c2:
        csv_buf = df.to_csv(index=False)
        b64c    = base64.b64encode(csv_buf.encode()).decode()
        href2   = (f'<a href="data:file/csv;base64,{b64c}" download="fire_data_processed.csv" '
                   f'style="display:inline-block;background:linear-gradient(135deg,#006633,#00c896);color:white;'
                   f'padding:12px 28px;border-radius:8px;text-decoration:none;font-family:Space Grotesk,sans-serif;'
                   f'font-size:0.9rem;font-weight:600;letter-spacing:1px;box-shadow:0 4px 16px rgba(0,200,150,0.3)">📊 Download Data (.csv)</a>')
        st.markdown(href2, unsafe_allow_html=True)