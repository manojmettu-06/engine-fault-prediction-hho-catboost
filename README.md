# 🚀 Engine Fault Prediction and Maintenance Decision Support using Harris Hawks Optimized CatBoost

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-CatBoost-orange)
![Optimization](https://img.shields.io/badge/Optimization-Harris%20Hawks-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---


## 🚀 Live Demo

https://engine-fault-prediction-hho-catboost-cgpf8f8g8pojdrtxk5ptcc.streamlit.app/

# 📌 Project Overview

This project presents an intelligent **Engine Fault Prediction and Maintenance Decision Support System** using a **CatBoost Classifier** optimized with the **Harris Hawks Optimization (HHO)** algorithm.

The framework predicts the health condition of an engine using multiple sensor readings and provides maintenance recommendations based on the prediction. It also includes feature importance analysis for model explainability.

---

# ❗ Problem Statement

Modern industrial engines continuously generate sensor data. Detecting faults at an early stage helps reduce maintenance costs, avoid unexpected failures, and improve operational efficiency.

Traditional maintenance strategies are either:

- Reactive (after failure)
- Scheduled (regardless of condition)

Both approaches increase operational cost.

This project develops an intelligent predictive maintenance system capable of identifying engine conditions from sensor data and recommending maintenance actions.

---

# 🎯 Objectives

- Predict engine health using machine learning.
- Optimize CatBoost hyperparameters using Harris Hawks Optimization.
- Improve prediction performance.
- Identify important engine sensor features.
- Generate maintenance recommendations.
- Build an end-to-end predictive maintenance framework.

---

# 📊 Dataset

The dataset contains **10,000 engine samples** with multiple sensor measurements.

### Features

- Vibration Amplitude
- RMS Vibration
- Vibration Frequency
- Surface Temperature
- Exhaust Temperature
- Acoustic dB
- Acoustic Frequency
- Intake Pressure
- Exhaust Pressure
- Frequency Band Energy
- Amplitude Mean

### Target

**Engine_Condition**

- 0 → Normal
- 1 → Warning
- 2 → Fault

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- CatBoost
- Joblib
- Harris Hawks Optimization (Custom Implementation)

---

# 📁 Project Structure

```
engine_fault/
│
├── dataset/
│   └── engine_fault_detection_dataset.csv
│
├── hho/
│   ├── __init__.py
│   ├── population.py
│   ├── levy.py
│   ├── fitness.py
│   ├── update.py
│   └── hho.py
│
├── models/
│   └── baseline_catboost.pkl
│
├── outputs/
│   ├── feature_importance.csv
│   └── feature_importance.png
│
├── dataanalysis.py
├── preproccesing.py
├── baseline_catboost.py
├── optimize_catboost.py
├── explainability.py
├── recommendation.py
├── main.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Project Workflow

```
Dataset
      │
      ▼
Data Analysis
      │
      ▼
Preprocessing
      │
      ▼
Baseline CatBoost Model
      │
      ▼
Harris Hawks Optimization
      │
      ▼
Optimized CatBoost Model
      │
      ▼
Feature Importance
      │
      ▼
Maintenance Recommendation
      │
      ▼
Final Engine Health Report
```

---

# 🦅 Harris Hawks Optimization (HHO)

Harris Hawks Optimization is a nature-inspired optimization algorithm based on the cooperative hunting strategy of Harris hawks.

In this project, HHO is used to optimize CatBoost hyperparameters including:

- Iterations
- Learning Rate
- Tree Depth
- L2 Leaf Regularization
- Random Strength
- Bagging Temperature

The optimizer searches for the best parameter combination that maximizes prediction accuracy.

---

# 🌳 CatBoost Classifier

CatBoost is a gradient boosting algorithm that efficiently handles structured tabular data.

Advantages:

- High prediction accuracy
- Robust against overfitting
- Fast training
- Excellent performance on tabular datasets

---

# 📈 Explainability

Model explainability is achieved using CatBoost Feature Importance.

The project identifies the most influential engine parameters affecting predictions.

Top Important Features:

- Vibration Frequency
- Vibration Amplitude
- Intake Pressure
- Frequency Band Energy
- Exhaust Pressure

Feature importance plots are automatically generated and saved in the **outputs/** directory.

---

# 🔧 Maintenance Recommendation

Based on the predicted engine condition, the system provides maintenance suggestions.

### Normal

- Continue routine maintenance
- Monitor sensor readings
- Perform scheduled inspections

### Warning

- Inspect bearings
- Check lubrication
- Monitor vibration
- Inspect exhaust pressure

### Fault

- Inspect shaft alignment
- Replace damaged bearings
- Check intake and exhaust pressure
- Perform complete maintenance

---

# 📊 Experimental Results

## Baseline CatBoost

| Metric | Value |
|---------|--------|
| Accuracy | 58.90% |
| Precision | 45.57% |
| Recall | 58.90% |
| F1 Score | 46.21% |

---

## HHO Optimized CatBoost

Best Accuracy:

```
59.70%
```

Best Parameters:

```
Iterations           : 63
Learning Rate        : 0.078
Depth                : 8
L2 Leaf Regularization : 2
Random Strength      : 1.924
Bagging Temperature  : 1.084
```

---

# ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/engine-fault-prediction.git

cd engine_fault
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the complete project:

```bash
python main.py
```

---

# 📌 Outputs

The project automatically generates:

- Trained CatBoost Model
- Feature Importance CSV
- Feature Importance Plot
- Maintenance Recommendation
- Engine Health Report

---

# 🔮 Future Scope

- Deep Learning-based fault prediction
- Real-time IoT sensor integration
- Remaining Useful Life (RUL) estimation
- Streamlit Web Dashboard
- Explainability using SHAP/LIME
- Cloud deployment
- Mobile monitoring application

---

# 👨‍💻 Author

**Mettu Manoj Babu**

B.Tech – Computer Science and Business Systems (CSBS)

Machine Learning | AI | Data Science | Predictive Maintenance

---

# ⭐ If you found this project useful, consider giving it a star!
