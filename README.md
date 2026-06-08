# Traffic Demand Prediction

A machine learning solution for predicting normalized traffic demand across road segments using geospatial, temporal, road, and weather features. Built for the **Flipkart Challenge**, the pipeline combines three gradient boosting models in a stacked ensemble with pseudo-label semi-supervised learning.

---

## Project Structure

```
├── Traffic_Demand_Prediction.ipynb   # End-to-end ML pipeline notebook
├── train.csv                         # Labeled training data
├── test.csv                          # Unlabeled test data for submission
└── README.md
```

---

## Problem Statement

Predict the **traffic demand** (a normalized continuous value) for a given road segment, time slot, and set of environmental conditions. Accurate demand forecasting enables smarter urban traffic management and infrastructure planning.

---

## Dataset

### Features

| Column | Description |
|--------|-------------|
| `geohash` | Encoded geographic location of the road segment |
| `day` | Day identifier |
| `timestamp` | Time of observation (HH:MM format) |
| `RoadType` | Type of road (e.g., Residential) — has missing values |
| `NumberofLanes` | Number of lanes on the road |
| `LargeVehicles` | Whether large vehicles are allowed (`Allowed` / `Not Allowed`) |
| `Landmarks` | Whether landmarks are nearby (`Yes` / `No`) |
| `Temperature` | Recorded temperature — has missing values |
| `Weather` | Weather condition (Sunny, Rainy, Snowy, Foggy, etc.) — has missing values |

### Target

| Column | Description |
|--------|-------------|
| `demand` | Normalized traffic demand (train set only) |

---

## Pipeline Overview

```
Raw Data
   │
   ▼
Missing Value Imputation
(Temperature → median, Weather/RoadType → "Unknown")
   │
   ▼
Geohash Decoding → latitude, longitude
+ Engineered features: lat×lon product, distance from center
   │
   ▼
Timestamp Parsing → time_slot (0–95, one per 15-min interval)
   │
   ▼
Label Encoding
(Weather, RoadType, LargeVehicles, Landmarks)
   │
   ▼
Base Models: XGBoost | LightGBM | CatBoost
trained with 5-Fold Cross-Validation (OOF predictions)
   │
   ▼
Meta-Learner: Ridge Regression
stacked on base model predictions
   │
   ▼
Pseudo-Label Semi-Supervised Refinement
(lowest-uncertainty 20% test predictions used to augment training)
   │
   ▼
Final Stacked Prediction
```

---

## Tech Stack

| Category | Library |
|----------|---------|
| Data Handling | `pandas`, `numpy` |
| Visualization | `matplotlib` |
| Geospatial | `pygeohash` |
| Preprocessing | `scikit-learn` (LabelEncoder) |
| Gradient Boosting | `xgboost`, `lightgbm`, `catboost` |
| Meta-Learner | `scikit-learn` (Ridge) |
| Evaluation | `scikit-learn` (RMSE, MAE, R²) |

---

## Model Architecture

### Base Models (5-Fold KFold OOF)

All three base models share consistent hyperparameters for a fair ensemble:

| Hyperparameter | Value |
|----------------|-------|
| `n_estimators` / `iterations` | 1000 |
| `learning_rate` | 0.05 |
| `max_depth` | 8 |
| `min_child_weight` / `min_data_in_leaf` | 5 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |

### Meta-Learner
- **Ridge Regression** (`alpha=1.0`) trained on validation set predictions from the three base models.

### Pseudo-Label Refinement
After the initial stack, test samples with the **lowest 20% prediction variance** (across the 3 models) are treated as high-confidence pseudo-labels and added to the training data. All base models and the meta-learner are then retrained on this augmented dataset.

---

## Evaluation Metrics

```python
def evaluate(label, y_true, y_pred):
    rmse = root_mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"{label:<25}  RMSE: {rmse:.4f}  MAE: {mae:.4f}  R²: {r2:.4f}")
```

Four checkpoints are evaluated:
1. OOF XGBoost
2. OOF LightGBM
3. OOF CatBoost
4. OOF Simple Average (baseline ensemble)
5. Stacked Ridge on validation set
6. Stacked Ridge **after** pseudo-label retraining

---

## Getting Started

### 1. Install Dependencies

```bash
pip install pandas numpy matplotlib pygeohash scikit-learn xgboost lightgbm catboost
```

### 2. Clone & Run

```bash
git clone https://github.com/your-username/traffic-demand-prediction.git
cd traffic-demand-prediction
jupyter notebook Traffic_Demand_Prediction.ipynb
```

Make sure `train.csv` and `test.csv` are in the same directory as the notebook.

---

## Key Design Decisions

- **Geohash decoding**: Raw geohash strings are decoded into latitude/longitude coordinates. Two derived features are added — the `lat×lon` product and the Euclidean distance from the geographic center of the training set — to capture spatial structure without one-hot encoding thousands of location strings.
- **Time slot encoding**: Timestamps are converted into a single integer `time_slot` (0–95) representing each 15-minute window in a 24-hour day, preserving ordinal time structure in a single feature.
- **OOF stacking**: Out-of-fold predictions prevent target leakage from the base models into the meta-learner.
- **Pseudo-labeling**: High-confidence test predictions (low inter-model variance) are used to expand the training set, leveraging the unlabeled data in a semi-supervised fashion.

---
