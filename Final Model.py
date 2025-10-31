# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 19:58:18 2025

@author: ABHI
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 20:05:13 2025

@author: ABHI
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, classification_report, confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings("ignore")
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

xgboost_available = True
lightgbm_available = True
imblearn_available = True


DATA_PATH = Path(r"D:\Smart Accident Risk Classification Using Traffic and Signal Data\Model\Project_dataset.csv")     # adjust if file in another path
PREFERRED_TARGET = "accident_occurred"  # will fall back if missing
TEST_SIZE = 0.25
RANDOM_STATE = 42
N_SPLITS = 5
N_ITER_SEARCH = 30   # randomized search iterations


# Load & inspect dataset
df = pd.read_csv(DATA_PATH)
print("Loaded dataset:", DATA_PATH, " shape:", df.shape)
print("\nColumns:", list(df.columns)[:32])
print("\nSample rows:")
display(df.head())

print("\nMissing counts (top 20):")
print(df.isnull().sum().sort_values(ascending=False).head(20))

# Picking target 
if PREFERRED_TARGET in df.columns and df[PREFERRED_TARGET].notna().sum() > 10:
    target = PREFERRED_TARGET
else:
    target = None

    
    cand = []
    for c in df.columns:
        non_null_frac = df[c].notna().mean()
        nunique = df[c].nunique(dropna=True)
        cand.append((c, non_null_frac, nunique))
    cand_df = pd.DataFrame(cand, columns=["col","non_null_frac","nunique"]).sort_values(['non_null_frac','nunique'], ascending=[False, True])
    # choose first with non_null_frac >= .7 and nunique between 2 and 20
    target = None
    for _, r in cand_df.iterrows():
        if r['non_null_frac'] >= 0.7 and 2 <= r['nunique'] <= 20:
            target = r['col']; break
    if target is None:
        target = df.columns[-1]   # fallback
print("\nChosen target column:", target)
print(df[target].value_counts(dropna=False).head(20))

# Drop rows missing target
df = df.dropna(subset=[target]).reset_index(drop=True)
print("After dropping missing target rows shape:", df.shape)

# Define X, y and drop obvious leakage columns
y = df[target]
X = df.drop(columns=[target])

# Drop common leakage columns if present
leakage_guess = ['is_peak', 'peak', 'timestamp', 'time_of_day', 'datetime']
for c in leakage_guess:
    if c in X.columns:
        print("Dropping leakage-like column:", c)
        X = X.drop(columns=[c])

# Basic dtype cleaning
# Convert boolean/object target to numeric classes
if y.dtype == 'bool' or y.dtype == object:
    y = pd.factorize(y)[0]

# Make sure we have samples
if X.shape[0] == 0 or len(y) == 0:
    raise ValueError("No samples left after preprocessing. Inspect the dataset and target selection.")

# Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
print("Train / Test shapes:", X_train.shape, X_test.shape)
print("Train class distribution:\n", pd.Series(y_train).value_counts(normalize=True))
print("Test class distribution:\n", pd.Series(y_test).value_counts(normalize=True))

# Determine numeric & categorical columns
numeric_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object','category','bool']).columns.tolist()

# avoid exploding OHE for very high-cardinality cat columns (like IDs)
high_cardinality = [c for c in categorical_cols if X[c].nunique() > 100]
if high_cardinality:
    print("Dropping high-cardinality categorical columns (example ids/timestamps):", high_cardinality[:10])
    for c in high_cardinality:
        categorical_cols.remove(c)
        X_train = X_train.drop(columns=[c], errors='ignore')
        X_test = X_test.drop(columns=[c], errors='ignore')

print("Numeric cols count:", len(numeric_cols), "Categorical cols count:", len(categorical_cols))

# Preprocessing pipelines
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, numeric_cols),
    ('cat', cat_pipeline, categorical_cols)
], remainder='drop')


# Models & parameter grids 
models_and_params = []

# Logistic Regression 
lr = LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=200)
pipe_lr = ImbPipeline([('pre', preprocessor), ('sm', SMOTE(random_state=RANDOM_STATE)), ('clf', lr)])

params_lr = {
    'clf__C': np.logspace(-3, 2, 10),
}

models_and_params.append(("LogisticRegression", pipe_lr, params_lr))

# Random Forest
rf = RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1)
pipe_rf = ImbPipeline([('pre', preprocessor), ('sm', SMOTE(random_state=RANDOM_STATE)), ('clf', rf)])

params_rf = {
    'clf__n_estimators': [100, 200, 400],
    'clf__max_depth': [6, 8, 12, None],
    'clf__min_samples_leaf': [1, 3, 5]
}
models_and_params.append(("RandomForest", pipe_rf, params_rf))

# XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE)
pipe_xgb = ImbPipeline([('pre', preprocessor), ('sm', SMOTE(random_state=RANDOM_STATE)), ('clf', xgb)])
params_xgb = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [3, 5, 7],
    'clf__learning_rate': [0.01, 0.05, 0.1],
    'clf__subsample': [0.7, 0.9]
}
models_and_params.append(("XGBoost", pipe_xgb, params_xgb))

# LightGBM
lgbm = LGBMClassifier(random_state=RANDOM_STATE)
pipe_lgb = ImbPipeline([('pre', preprocessor), ('sm', SMOTE(random_state=RANDOM_STATE)), ('clf', lgbm)])
params_lgb = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [6, 8, -1],
    'clf__learning_rate': [0.01, 0.05, 0.1]
}
models_and_params.append(("LightGBM", pipe_lgb, params_lgb))

# Cross-validated randomized search & evaluation
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

cv_results = []
best_models = {}
target_metrics = {'accuracy':0.85,'precision':0.80,'recall':0.75,'f1':0.77,'roc_auc':0.85}

for name, pipeline_obj, param_dist in models_and_params:
    print("\n\n=== Training candidate:", name, "===")
    rs = RandomizedSearchCV(
        estimator=pipeline_obj,
        param_distributions=param_dist,
        n_iter=N_ITER_SEARCH if len(param_dist)>0 else 1,
        scoring='f1_weighted',
        cv=skf,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=True
    )
    start = time.time()
    try:
        rs.fit(X_train, y_train)
    except Exception as e:
        print("Warning: randomized search failed for", name, " — trying quick fit. Error:", e)
        pipeline_obj.fit(X_train, y_train)
        best = pipeline_obj
    else:
        best = rs.best_estimator_
        print(f"{name} best params:", rs.best_params_)
        print(f"{name} best CV score (f1_weighted):", rs.best_score_)
    end = time.time()
    print(f"{name} training time: {end-start:.1f}s")

    # Evaluate on test set
    y_pred = best.predict(X_test)
    try:
        y_prob = best.predict_proba(X_test)
        if y_prob.shape[1] == 2:
            y_prob = y_prob[:,1]
        else:
            y_prob = None
    except Exception:
        y_prob = None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    roc = None
    try:
        if y_prob is not None and len(np.unique(y_test))==2:
            roc = roc_auc_score(y_test, y_prob)
    except Exception:
        roc = None

    print(f"Test Metrics for {name}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, ROC-AUC={roc}")
    cv_results.append({'name':name, 'estimator':best, 'accuracy':acc, 'precision':prec, 'recall':rec, 'f1':f1, 'roc_auc':roc})
    best_models[name] = best

# Collect & display results
res_df = pd.DataFrame(cv_results).sort_values('f1', ascending=False).reset_index(drop=True)
print("\n\n=== Summary of candidate models (test set) ===") 
display(res_df[['name','accuracy','precision','recall','f1','roc_auc']])

# To save the best Logistic Regression model
log_reg_model = best_models.get("LogisticRegression")
joblib.dump(log_reg_model, "logistic_regression_model.pkl")
print("Logistic Regression model saved as logistic_regression_model.pkl")
