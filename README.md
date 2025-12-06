## NYRA Horse Racing Prediction — GPS-Based Spatiotemporal ML

This project builds a **full machine learning pipeline** for predicting thoroughbred race outcomes using **high-resolution GPS tracking data** provided by the New York Racing Association (NYRA).  
It includes **end-to-end data preparation, exploratory analysis, feature engineering, model development, evaluation, and interpretability analysis** with SHAP explanations.

---

## Dataset: High-Frequency GPS Trajectories

This project uses the official dataset from **Kaggle Big Data Derby 2022**:

🔗 **Dataset link:**  
https://www.kaggle.com/competitions/big-data-derby-2022/data

- **0.1-second interval GPS tracking data**
- Each race includes:  
  - Latitude / longitude positions  
  - Instantaneous speed  
  - Course type  
  - Odds & implied probability  
  - Race distance & configuration  
- Each horse's trajectory forms a **spatiotemporal sequence** requiring careful time-aware preprocessing.

---

## Requirements
```
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
scikit-learn>=1.3.0
xgboost>=2.0.0
category-encoders>=2.6.0
shap>=0.42.0
```

---

## Exploratory Data Analysis (EDA)

The EDA examines:

- Trajectory patterns across different horses  
- Speed curves, deceleration events, and cornering behavior  
- Spatial movement characteristics (e.g., lateral variance)  
- Feature correlations and nonlinear relationships  
- Distribution shifts across course types and distances  
- Visualization of race paths and turning efficiency  

These findings shaped the feature engineering strategy and model design.

---

## Feature Engineering

To capture both spatial and temporal structure, the project computes domain-specific features:

- **Path Length** (actual distance traveled)  
- **Path Efficiency** (optimal vs. actual trajectory)  
- **Average Speed**  
- **Jockey Performance Metrics** (average finish, average odds)
- **Implied Probability** (derived from betting odds)
- **Course Type Encoding**  
- **Race Hour** (temporal feature)

These features were critical in improving predictive performance.

---

## Machine Learning Pipeline

The full ML pipeline includes:

- **Time-series–aware train/validation/test split** (chronological, no shuffling)
- Standardization and preprocessing  
- Model training with six algorithms:  
  - **Ridge Regression**  
  - **Lasso Regression**  
  - **ElasticNet**
  - **Random Forest**
  - **Gradient Boosting Regressor**  
  - **XGBoost**  

Hyperparameters were tuned using GridSearchCV with time-series cross-validation.

---

## Performance Summary

| Rank | Model | Test RMSE | Test R² | Improvement over Baseline |
|------|-------|-----------|---------|---------------------------|
| 1 | **Lasso** | **2.367** | **0.285** | **+16.69%** |
| 2 | Gradient Boosting | 2.371 | 0.282 | +16.55% |
| 3 | XGBoost | 2.372 | 0.282 | +16.51% |
| 4 | ElasticNet | 2.385 | 0.274 | +16.07% |
| 5 | Random Forest | 2.387 | 0.273 | +15.98% |
| 6 | Ridge | 2.497 | 0.204 | +12.12% |

> **Baseline RMSE: 2.841**

### Winner Prediction Accuracy
- **Predicted 1st → Actual 1st:** 33.8%
- **Predicted 1st → Actual Top 3:** 66.7%

---

## Feature Importance & Explainability

Four complementary methods were applied:

### **1. Permutation Importance (Model Agnostic)**  
Captures how feature shuffling impacts predictions.

### **2. Tree-Based Feature Importances (XGBoost / GBM / RF)**  
Captures information gain and split value importance.

### **3. Linear Coefficients (Ridge / Lasso / ElasticNet)**  
Provides interpretable weights.

### **4. SHAP Values (Local Interpretability)**
Explains individual predictions and feature contributions.

### **Top Features (Aggregated Across Methods)**

1. **Implied Probability**  
2. **Average Speed**  
3. **Odds**  
4. **Path Length**  
5. **Jockey Average Finish**

These findings highlight that **betting market expectations (odds/implied probability) are the strongest predictors**, followed by physical performance metrics.

---

## Project Structure
```
├── NYRA_Horse_Racing_Prediction.ipynb   # Main notebook
├── README.md
├── requirements.txt
├── data/                                 # Raw data files
│   ├── nyra_race_table.csv
│   ├── nyra_start_table.csv
│   └── nyra_tracking_table.csv
└── outputs/
    ├── preprocessed/                     # Preprocessed data
    ├── models/                           # Trained models
    ├── evaluation_results/               # Test results
    ├── feature_importance/               # Importance CSVs & SHAP plots
    ├── figures/                          # Visualizations
    └── final_results/                    # Summary & executive report
```

---
