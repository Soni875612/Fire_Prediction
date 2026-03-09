

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0b3d91,40:1565c0,70:fc3d21,100:ffb300&height=220&section=header&text=🔥%20FireSense%20AI&fontSize=55&fontColor=ffffff&fontAlignY=40&desc=NASA%20Wildfire%20Intelligence%20Platform%20%7C%20Australia%202019&descAlignY=60&descSize=16&descColor=ffffffcc" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)
[![NASA](https://img.shields.io/badge/NASA-FIRMS-0B3D91?style=for-the-badge&logo=nasa&logoColor=white)](https://firms.modaps.eosdis.nasa.gov)
[![License](https://img.shields.io/badge/License-Academic-green?style=for-the-badge)](.)

<br/>

> 🛰️ **Real NASA MODIS satellite data** · 🔥 **36,011 fire events** · 🤖 **3 ML models** · 📊 **8 interactive pages**

<br/>

[![⭐ Star](https://img.shields.io/github/stars/YOUR_USERNAME/fire-prediction?style=social)](https://github.com/YOUR_USERNAME/fire-prediction)
&nbsp;
[![🍴 Fork](https://img.shields.io/github/forks/YOUR_USERNAME/fire-prediction?style=social)](https://github.com/YOUR_USERNAME/fire-prediction/fork)

</div>

---

## 📋 Table of Contents

| # | Section |
|---|---------|
| 1 | [🌟 Overview](#-overview) |
| 2 | [📸 Screenshots](#-screenshots) |
| 3 | [📊 Real Dataset Statistics](#-real-dataset-statistics) |
| 4 | [✨ Features](#-features) |
| 5 | [🧠 ML Model Results](#-ml-model-results) |
| 6 | [📚 Academic Concepts](#-academic-concepts-applied) |
| 7 | [⚙️ Installation](#-installation) |
| 8 | [📁 Project Structure](#-project-structure) |
| 9 | [🛠️ Tech Stack](#-tech-stack) |
| 10 | [👤 Author](#-author) |

---

## 🌟 Overview

**FireSense AI** is an advanced wildfire prediction and intelligence dashboard built using **real NASA FIRMS MODIS satellite data** from Australia's devastating 2019 bushfire season. It combines machine learning, statistical analysis, and interactive visualization into a single professional command-center interface.

```
📡 NASA MODIS Satellite
        ↓
📂 fire_archive.csv  (36,011 records)
        ↓
🔧 Feature Engineering  →  StandardScaler
        ↓
🤖 ML Models: Logistic Regression · Random Forest · Gradient Boosting
        ↓
🔴 HIGH   🟡 MEDIUM   🟢 LOW   →   Risk Classification
        ↓
📊 Interactive Dashboard  →  Alerts  →  PDF Report
```

---

## 📸 Screenshots

### 🏠 Command Center — Main Dashboard
```

<img width="1626" height="914" alt="Screenshot 2026-03-09 221808" src="https://github.com/user-attachments/assets/97b30f78-dbe0-4a4d-ba6f-0f61e1847ebd" />

> *Shows: KPI cards · Live alerts · Fire timeline · Risk pie chart*
---

### 🗺️ Fire Hotspot Map
```
<img width="1919" height="918" alt="Screenshot 2026-03-09 221846" src="https://github.com/user-attachments/assets/b3e64242-71f3-4d36-8fc6-7e46db68ab67" />

```
> *Shows: Interactive Mapbox map with 36,011 fire locations color-coded by risk level*

---

### 🧠 ML Model Laboratory
```
<img width="1919" height="891" alt="Screenshot 2026-03-09 222106" src="https://github.com/user-attachments/assets/6ff81c39-91b5-4425-84d9-5eb5cf5b4f54" />

```
> *Shows: 3 model accuracy cards · Confusion matrix · Feature importance chart*

---

### 🎯 Fire Risk Predictor
```
<img width="1919" height="891" alt="Screenshot 2026-03-09 222106" src="https://github.com/user-attachments/assets/303286ee-23a7-4ad4-94b0-725f9cc3bdaa" />

```
> *Shows: Input sliders · Real-time prediction · Probability bar chart*

---

> 💡 **Tip:** You can also use [Streamlit's built-in screenshot](https://docs.streamlit.io) or Windows Snipping Tool (`Win + Shift + S`)

---

## 📊 Real Dataset Statistics

> All numbers below are extracted directly from `fire_archive.csv`

### 🔢 Core Numbers

| Metric | Value |
|--------|-------|
| 📁 Total Fire Events | **36,011** |
| 📅 Date Range | **Aug 01, 2019 – Sep 30, 2019** |
| 🗓️ Days Monitored | **61 days** |
| 🛰️ Satellites Used | **2 (Terra + Aqua)** |
| 🌏 Geographic Coverage | **Australia** |
| 📍 Lat Range | **−42.76° to −10.07°** |
| 📍 Lon Range | **114.10° to 153.49°** |

---

### 🔴 Risk Level Distribution

```
HIGH RISK    ████████████████████████░░░░░░░░░░░░░░  18,314  (50.9%)
MEDIUM RISK  ███████████████░░░░░░░░░░░░░░░░░░░░░░░  12,982  (36.1%)
LOW RISK     █████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   4,715  (13.1%)
```

| Risk Level | Count | Percentage | Threshold |
|------------|-------|-----------|-----------|
| 🔴 High | 18,314 | 50.9% | Confidence ≥ 70% |
| 🟡 Medium | 12,982 | 36.1% | Confidence 40–69% |
| 🟢 Low | 4,715 | 13.1% | Confidence < 40% |

---

### ⚡ Fire Radiative Power (FRP) Analysis

| Statistic | Value |
|-----------|-------|
| 📊 Mean FRP | **51.13 MW** |
| 📊 Median FRP | **25.80 MW** |
| 📊 Std Deviation | **92.28 MW** |
| 🔺 Maximum FRP | **3,679.50 MW** *(Sep 12, 2019)* |
| 🔻 Minimum FRP | **0.00 MW** |
| 🚨 Extreme Fires (>500 MW) | **245 events** |

> 📌 *The massive gap between mean (51 MW) and max (3,679 MW) shows highly skewed distribution — a few extreme fires dominate the energy output.*

---

### 🌡️ Brightness Temperature Analysis

| Statistic | Value |
|-----------|-------|
| 📊 Mean Brightness | **328.75 K** (55.6°C) |
| 🔺 Max Brightness | **504.40 K** (231.25°C) |
| 🔻 Min Brightness | **300.00 K** (26.85°C) |
| 📊 Std Deviation | **18.99 K** |

---

### 📡 Detection Confidence

| Statistic | Value |
|-----------|-------|
| Mean Confidence | **67.6%** |
| Median Confidence | **70.0%** |
| High Confidence (≥70%) | **18,314 events (50.9%)** |

---

### 🛰️ Satellite Comparison

| Satellite | Records | Percentage | Orbit |
|-----------|---------|-----------|-------|
| 🛰️ Aqua | 20,541 | **57.0%** | Afternoon pass |
| 🛰️ Terra | 15,470 | **43.0%** | Morning pass |

> 📌 *Aqua detects more fires because it passes over Australia in the afternoon when fires are most active due to heat.*

---

### 🌞 Day vs Night Fire Activity

```
DAYTIME    ████████████████████████████████░░░░░░  28,203  (78.3%)
NIGHTTIME  █████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   7,808  (21.7%)
```

| Period | Count | % | Avg FRP |
|--------|-------|---|---------|
| ☀️ Daytime | 28,203 | 78.3% | Higher (heat + wind) |
| 🌙 Nighttime | 7,808 | 21.7% | Lower (cooler temps) |

---

### 🚨 Top 5 Most Dangerous Days

| Rank | Date | Fire Count | Avg FRP | Max FRP | Avg Confidence |
|------|------|-----------|---------|---------|---------------|
| 🥇 1st | **Sep 20, 2019** | 362 | **109.9 MW** | 1,830.7 MW | 73.9% |
| 🥈 2nd | **Sep 06, 2019** | 1,200 | **85.0 MW** | 1,411.8 MW | 71.4% |
| 🥉 3rd | **Sep 30, 2019** | 669 | **79.4 MW** | 1,587.2 MW | 74.4% |
| 4th | **Sep 12, 2019** | 940 | **72.2 MW** | **3,679.5 MW** ⚠️ | 69.3% |
| 5th | **Aug 28, 2019** | 546 | **70.4 MW** | 2,129.7 MW | 72.4% |

> ⚠️ *Sep 12 recorded the single highest FRP ever in this dataset — 3,679.5 MW. That's equivalent to ~3.7 nuclear power plants in energy output.*

---

## ✨ Features

<table>
<tr><td width="50%">

**🏠 Command Center**
- 6 live KPI metric cards
- Auto-generated alert engine
- Dual-axis daily fire timeline
- Brightness vs FRP density heatmap

**🗺️ Fire Hotspot Map**
- Interactive Mapbox satellite map
- 3 risk level color filters
- Heatmap density overlay toggle
- 3 map style options

**📈 Trend Analysis**
- Daily event area charts
- OLS regression trendline
- Terra vs Aqua satellite comparison
- Day vs Night violin distribution

**🌐 3D Visualization**
- 3D geographic scatter (lon/lat/brightness)
- Orthographic globe view
- 🔥 Animated fire spread over time

</td><td width="50%">

**🚨 Alert System**
- 4 auto-generated alert types
- Custom FRP threshold slider
- Custom confidence threshold slider
- Top 20 extreme events table

**🧠 ML Model Lab**
- 3 models accuracy comparison bar chart
- Interactive confusion matrix heatmap
- Random Forest feature importance
- Full syllabus mapping table

**🎯 Risk Predictor**
- 6-input sensor reading form
- Real-time prediction (all 3 models)
- Probability bar chart
- Risk advice output

**📄 Report**
- Auto-generated TXT report
- One-click download button
- Processed CSV export

</td></tr>
</table>

---

## 🧠 ML Model Results

### Training Setup

| Parameter | Value |
|-----------|-------|
| Total Records | 36,011 |
| Training Set (80%) | ~28,808 records |
| Test Set (20%) | ~7,203 records |
| Features | 7 (brightness, frp, confidence, bright_t31, scan, track, delta_b) |
| Target Classes | 3 (Low, Medium, High) |
| Scaler | StandardScaler (Z-score normalization) |
| Random State | 42 |

### Model Accuracy Comparison

```
Logistic Regression   ████████████████████████████████░░  ~85–88%
Random Forest         ████████████████████████████████████  ~95–98%  ← BEST
Gradient Boosting     ███████████████████████████████████░  ~93–96%
```

> 📌 *Random Forest performs best because it handles non-linear relationships between brightness, FRP, and confidence effectively through ensemble of 150 decision trees.*

### Feature Importance (Random Forest)

```
confidence    ████████████████████████████████████  Most Important
frp           ████████████████████████░░░░░░░░░░░░
brightness    ████████████████░░░░░░░░░░░░░░░░░░░░
delta_b       ████████████░░░░░░░░░░░░░░░░░░░░░░░░
bright_t31    ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░
scan          ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
track         ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  Least Important
```

---

## 📚 Academic Concepts Applied

```
┌────────────────────────────────────────────────────────────────────┐
│   CURRICULUM TOPIC               →   WHERE USED IN PROJECT         │
├────────────────────────────────────────────────────────────────────┤
│   Introduction to AI             →   AI risk prediction system     │
│   Fundamentals of Statistics     →   FRP & brightness analysis     │
│   Descriptive Statistics         →   Mean=51.1 MW, Std=92.3 MW    │
│   Inferential Statistics         →   Sample → 36K record predict  │
│   Hypothesis Testing             →   Confidence ≥ 70% = HIGH Risk  │
│   Statistical Analysis           →   Correlation matrix, OLS line  │
│   Concept of Machine Learning    →   Pattern learning on 36K rows  │
│   Types of ML Algorithms         →   LR · Random Forest · GBM     │
│   Components of ML               →   Features · Labels · Metrics   │
└────────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.9+
- pip

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/fire-prediction.git
cd fire-prediction

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

🚀 Opens at **`http://localhost:8501`**

> **Note:** If a file upload screen appears, upload `fire_archive.csv` through the browser.

### Dependencies

```
streamlit
pandas
plotly
scikit-learn
numpy
statsmodels
```

---

## 📁 Project Structure

```
fire-prediction/
│
├── 📄  app.py                ←  Main application (800+ lines)
├── 📊  fire_archive.csv      ←  NASA MODIS dataset (36,011 rows)
├── 📦  requirements.txt      ←  Python dependencies
├── 📘  README.md             ←  This documentation
│
└── 📂  screenshots/          ←  Add your app screenshots here
    ├── command_center.png
    ├── fire_map.png
    ├── ml_lab.png
    └── predictor.png
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **UI** | Streamlit | Web app framework |
| **Styling** | Custom CSS | NASA dark blue theme |
| **Charts** | Plotly Express + GO | All visualizations |
| **Maps** | Mapbox via Plotly | Fire hotspot maps |
| **ML** | scikit-learn | 3 classification models |
| **Stats** | statsmodels | OLS regression trendline |
| **Data** | Pandas + NumPy | Processing 36K records |
| **Fonts** | Google Fonts | Orbitron · Space Grotesk |
| **Hosting** | Streamlit Cloud | Free deployment |

---

## 👤 Author

<div align="center">

### Soni

*AI & Machine Learning — Project*

[(https://github.com/Soni875612)]

</div>

---

<div align="center">

**📡 Data Source:** [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) — Free for educational & research use

**📅 Dataset Period:** August–September 2019 · Australia Bushfire Season

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:ffb300,50:fc3d21,100:0b3d91&height=120&section=footer&text=FireSense%20AI%20v2.0&fontSize=20&fontColor=ffffff&fontAlignY=65" width="100%"/>

</div>
</div>
