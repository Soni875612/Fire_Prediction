<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0b3d91,50:fc3d21,100:ffb300&height=200&section=header&text=🔥%20FireSense%20AI&fontSize=50&fontColor=ffffff&fontAlignY=38&desc=NASA%20Wildfire%20Intelligence%20Platform&descAlignY=58&descSize=18&descColor=ffffff" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Plotly](https://img.shields.io/badge/Plotly-Visualization-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)
[![NASA](https://img.shields.io/badge/NASA-FIRMS%20Data-0B3D91?style=for-the-badge&logo=nasa&logoColor=white)](https://firms.modaps.eosdis.nasa.gov)

<br/>

> **An AI-powered wildfire prediction & intelligence system**
> Built on real NASA MODIS satellite data · Australia 2019 · 36,011 fire events

</div>

---

## 📋 Table of Contents

- [🌟 Overview](#-overview)
- [✨ Features](#-features)
- [🧠 Machine Learning](#-machine-learning)
- [📊 Dataset](#-dataset)
- [📚 Academic Concepts](#-academic-concepts-applied)
- [⚙️ Installation](#-installation)
- [📁 Project Structure](#-project-structure)
- [🛠️ Tech Stack](#-tech-stack)
- [👤 Author](#-author)

---

## 🌟 Overview

**FireSense AI** is a full-stack data science project that transforms raw NASA satellite readings into an intelligent wildfire risk assessment system. Using machine learning classification algorithms and advanced interactive visualizations, it provides real-time fire risk prediction across Australia's 2019 bushfire season.

```
📡 NASA MODIS Satellite  →  📊 Data Processing  →  🤖 ML Models  →  🔥 Risk Prediction
```

| Metric | Value |
|--------|-------|
| 🔥 Total Fire Events | 36,011 |
| 📅 Date Range | Aug 1 – Sep 30, 2019 |
| 🛰️ Satellites | Terra + Aqua (MODIS) |
| 🌏 Region | Australia |
| 🤖 ML Models | 3 (LR, RF, GBM) |
| 📄 Pages | 8 interactive pages |

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🏠 Command Center
- Live KPI dashboard (6 metrics)
- Real-time alert engine
- Dual-axis fire activity timeline
- Brightness vs FRP density heatmap

### 🗺️ Fire Hotspot Map
- Interactive satellite map (Mapbox)
- Risk level color filtering
- Heatmap density overlay toggle
- Multiple map style options

### 📈 Trend Analysis
- Daily fire event area charts
- OLS regression trendline
- Satellite comparison (Terra vs Aqua)
- Day vs Night violin plots

### 🌐 3D Visualization
- 3D geographic scatter plot
- Orthographic globe view
- 🔥 Animated fire spread over time

</td>
<td width="50%">

### 🚨 Alert System
- Auto-generated critical alerts
- Custom FRP + confidence thresholds
- Alert heatmap visualization
- Top extreme fire events table

### 🧠 ML Model Laboratory
- 3 models side-by-side comparison
- Confusion matrix heatmap
- Feature importance chart (Random Forest)
- Full syllabus concepts mapping table

### 🎯 Fire Risk Predictor
- Real-time sensor input prediction
- All 3 models compared simultaneously
- Probability bar chart output
- Actionable risk advice

### 📄 Report Download
- Auto-generated analysis report (TXT)
- Processed CSV data export
- Complete statistical summary

</td>
</tr>
</table>

---

## 🧠 Machine Learning

### Training Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                         │
│                                                             │
│  36,011 records  →  Feature Engineering  →  StandardScaler  │
│                                                             │
│  Train 80%  ──┬──  Logistic Regression                     │
│               ├──  Random Forest      (150 trees)           │
│               └──  Gradient Boosting  (100 estimators)      │
│                                                             │
│  Test  20%  →  Accuracy · Confusion Matrix · F1 Score      │
└─────────────────────────────────────────────────────────────┘
```

### Features Used

| Feature | Description | Role |
|---------|-------------|------|
| `brightness` | Fire brightness temperature (Kelvin) | 🔴 Primary |
| `frp` | Fire Radiative Power (Megawatts) | 🔴 Primary |
| `confidence` | Detection confidence (%) | 🔴 Primary |
| `bright_t31` | Background brightness temperature (K) | 🟡 Secondary |
| `delta_b` | brightness − bright_t31 (engineered) | 🟡 Secondary |
| `scan` | Along-scan pixel size | 🟢 Supporting |
| `track` | Along-track pixel size | 🟢 Supporting |

### Risk Classes

```
Confidence ≥ 70%   →   🔴 HIGH RISK
Confidence 40–69%  →   🟡 MEDIUM RISK
Confidence  < 40%  →   🟢 LOW RISK
```

---

## 📊 Dataset

**Source:** [NASA FIRMS — Fire Information for Resource Management System](https://firms.modaps.eosdis.nasa.gov/)

```
Dataset : MODIS_C6_Australia_2019
Records : 36,011 fire detections
Period  : August 1 – September 30, 2019
Format  : CSV  (fire_archive.csv)
Size    : ~3 MB
```

<details>
<summary><b>📋 Click to see all column descriptions</b></summary>

<br/>

| Column | Type | Description |
|--------|------|-------------|
| `latitude` | float | Fire location — latitude |
| `longitude` | float | Fire location — longitude |
| `brightness` | float | Channel 21/22 brightness temperature (K) |
| `scan` | float | Along-scan pixel size |
| `track` | float | Along-track pixel size |
| `acq_date` | date | Acquisition date (YYYY-MM-DD) |
| `acq_time` | int | Acquisition time (HHMM UTC) |
| `satellite` | string | Terra or Aqua |
| `instrument` | string | MODIS |
| `confidence` | int | Detection confidence score (0–100) |
| `bright_t31` | float | Channel 31 brightness temperature (K) |
| `frp` | float | Fire Radiative Power in Megawatts |
| `daynight` | string | D = Daytime, N = Nighttime |

</details>

---

## 📚 Academic Concepts Applied

```
┌──────────────────────────────────────────────────────────────────────┐
│   CURRICULUM TOPIC                →   PROJECT APPLICATION            │
├──────────────────────────────────────────────────────────────────────┤
│   Introduction to AI              →   End-to-end AI prediction app   │
│   Fundamentals of Statistics      →   FRP & brightness distribution  │
│   Descriptive Statistics          →   Mean, median, std, histograms  │
│   Inferential Statistics          →   Sample → full dataset predict  │
│   Hypothesis Testing              →   Confidence ≥ 70% = High Risk   │
│   Statistical Analysis            →   Correlation matrix, OLS line   │
│   Concept of Machine Learning     →   Pattern learning on 36K rows   │
│   Types of ML Algorithms          →   LR  ·  Random Forest  ·  GBM  │
│   Components of ML                →   Features · Labels · Metrics    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Installation

### Requirements
- Python 3.9+
- pip

### Steps

```bash
# Step 1 — Clone repository
git clone https://github.com/YOUR_USERNAME/fire-prediction.git
cd fire-prediction

# Step 2 — Install dependencies
pip install -r requirements.txt

# Step 3 — Run the app
streamlit run app.py
```

✅ App opens at **`http://localhost:8501`**

> **Note:** If a file upload screen appears, upload `fire_archive.csv` through the browser.

---

## 📁 Project Structure

```
fire-prediction/
│
├── 📄  app.py               ←  Main Streamlit application
├── 📊  fire_archive.csv     ←  NASA MODIS satellite dataset
├── 📦  requirements.txt     ←  Python dependencies
└── 📘  README.md            ←  Project documentation
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **UI Framework** | Streamlit |
| **Styling** | Custom CSS — NASA Dark Blue Theme |
| **Charts** | Plotly Express + Plotly Graph Objects |
| **Maps** | Mapbox (via Plotly) |
| **Machine Learning** | scikit-learn |
| **Statistics** | statsmodels (OLS trendline) |
| **Data Processing** | Pandas + NumPy |
| **Fonts** | Orbitron · Space Grotesk · Space Mono |
| **Hosting** | Streamlit Cloud |

---

## 👤 Author

<div align="center">

### Soni

*AI & Machine Learning — Project*

[(https://github.com/Soni875612)]
</div>

---

<div align="center">

**Data:** [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) — Free for educational use

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:ffb300,50:fc3d21,100:0b3d91&height=100&section=footer" width="100%"/>

</div>
