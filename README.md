## NYRA Horse Racing Prediction — GPS-Based Spatiotemporal ML

This project builds a **full machine learning pipeline** for predicting thoroughbred race outcomes using **high-resolution GPS tracking data** provided by the New York Racing Association (NYRA).  
It includes **end-to-end data preparation, exploratory analysis, feature engineering, model development, evaluation, and global interpretability**, with local SHAP explanations to be added soon.

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
- Each horse’s trajectory forms a **spatiotemporal sequence** requiring careful time-aware preprocessing.

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
- **Deceleration Pattern Metrics**  
- **Lateral Variance** (stability of trajectory)  
- **Straight-Distance Coverage**  
- **Course Type Encoding**  
- **Tactical Positioning Indicators**  

These features were critical in improving predictive performance.

---

## Machine Learning Pipeline

The full ML pipeline includes:

- Time-series–aware train/validation/test split  
- Standardization and preprocessing  
- Model training with six algorithms:  
  - **XGBoost**  
  - **Gradient Boosting Regressor**  
  - **Random Forest**  
  - **Ridge Regression**  
  - **Lasso Regression**  
  - **ElasticNet**

Hyperparameters were tuned using systematic search and performance variance was measured across multiple seeds/splits.

---

## Performance Summary

Across all models, **XGBoost achieved the strongest predictive results**:

| Model | Test RMSE | Test R² | Improvement over Baseline |
|------|-----------|---------|----------------------------|
| **XGBoost** | **1.96** | **0.509** | **+30.95%** |
| Gradient Boosting | 2.01 | 0.485 | +29.32% |
| Random Forest | 2.02 | 0.476 | +28.69% |
| Ridge | 2.07 | 0.451 | +27.02% |
| Lasso | 2.12 | 0.428 | +25.50% |

> **The model explains ~51% of the variance in race outcomes using only GPS-derived features**, which is considered strong performance in high-noise sports prediction settings.

---

## Feature Importance & Explainability

Three complementary methods were applied:

### **1. Permutation Importance (Model Agnostic)**  
Captures how feature shuffling impacts predictions.

### **2. Tree-Based Feature Importances (XGBoost / GBM / RF)**  
Captures information gain and split value importance.

### **3. Linear Coefficients (Ridge / Lasso / ElasticNet)**  
Provides interpretable weights.

### **Aggregated Importance (Top Features)**  
Across all models, the most influential predictors were consistently:

1. **Path Length**  
2. **Average Speed**  
3. **Efficiency Score**  
4. **Implied Probability**  
5. **Odds**  
6. **Straight-Distance Coverage**

These findings highlight the importance of both physical trajectory metrics and market-derived expectations.

