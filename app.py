# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 20:50:24 2025

@author: ABHI
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Paths

MODEL_PATH = Path("logistic_regression_model.pkl")
DATASET_PATH = Path(r"D:\Smart Accident Risk Classification Using Traffic and Signal Data\DataSets\Project_dataset.csv")  # uses the dataset you provided

st.set_page_config(page_title="🚦 Smart Accident Risk Classification", layout="wide")
st.title("🚦 Smart Accident Risk Classification")


# Load model and dataset

if not MODEL_PATH.exists():
    st.error(" Trained model not found. Make sure logistic_regression_model.pkl is in this folder.")
    st.stop()
if not DATASET_PATH.exists():
    st.error(" Dataset not found. Put Project_dataset.csv in this folder.")
    st.stop()

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATASET_PATH)

# Identify target & features (exclude target from inputs)
TARGET = "accident_occurred" if "accident_occurred" in df.columns else None
feature_cols = [c for c in df.columns if c != TARGET]

# Build default values from dataset (for hidden/unwanted fields)
defaults = {}
for c in feature_cols:
    s = df[c]
    if s.dtype == bool:
        defaults[c] = bool(s.mode(dropna=True).iloc[0]) if not s.dropna().empty else False
    elif pd.api.types.is_numeric_dtype(s):
        defaults[c] = float(s.median()) if not s.dropna().empty else 0.0
    else:
        # categorical / object
        defaults[c] = str(s.mode(dropna=True).iloc[0]) if not s.dropna().empty else ""


# Curated inputs 

DISPLAY_COLS = [
    "avg_speed_kmph",
    "vehicle_count_per_hr",
    "speed_limit_kmph",
    "lane_count",
    "has_signal",
    "signal_status",
    "day_of_week",
    "hour_of_day",
    "road_type",
    "weather",
    "lighting",
    "enforcement_level",
    "green_duration_s",
    "red_duration_s",
    "yellow_duration_s",
    "cycle_time_s",
    "violations_count",
]
# Keep only columns that actually exist in this dataset
DISPLAY_COLS = [c for c in DISPLAY_COLS if c in feature_cols]

def choices(col):
    s = df[col].dropna().unique().tolist()
    # Booleans sometimes come in as object with True/False values
    if all(x in [True, False] for x in s):
        return [False, True]
    return sorted(s)


#  Accident Risk Predictor 

st.header("🚘 Accident Risk Predictor")

left, right = st.columns(2)
user_inputs = {}

for i, col in enumerate(DISPLAY_COLS):
    with (left if i % 2 == 0 else right):
        series = df[col]
        if series.dtype == bool:
            # Checkbox for booleans
            user_inputs[col] = st.checkbox(col, value=bool(defaults[col]))
        elif pd.api.types.is_numeric_dtype(series):
            # Special handling for discrete small-cardinality numerics
            unique_vals = series.dropna().unique()
            is_intlike = pd.api.types.is_integer_dtype(series) or np.all(np.mod(unique_vals, 1) == 0)
            if col == "hour_of_day":
                user_inputs[col] = st.slider(col, min_value=0, max_value=23, value=int(np.median(unique_vals)))
            elif col == "day_of_week":
              
                opts = sorted([int(v) for v in series.dropna().unique().tolist()])
                user_inputs[col] = st.selectbox(col, options=opts, index=opts.index(int(np.median(opts))))
            elif is_intlike and len(unique_vals) <= 24:
              
                opts = sorted([int(v) for v in unique_vals])
                default_val = int(defaults[col]) if col in defaults else opts[0]
                default_idx = opts.index(default_val) if default_val in opts else 0
                user_inputs[col] = st.selectbox(col, options=opts, index=default_idx)
            else:
                # Continuous numeric
                min_v = float(np.nanmin(series)) if series.notna().any() else 0.0
                max_v = float(np.nanmax(series)) if series.notna().any() else 1000.0
                default_val = float(defaults[col])
                user_inputs[col] = st.number_input(col, value=default_val, min_value=min_v, max_value=max_v)
        else:
            # Categorical/object -> dropdown from dataset values
            opts = choices(col)
            default_val = defaults[col] if defaults[col] in opts else (opts[0] if opts else "")
            # Use selectbox; if only 1 option, it will be fixed
            user_inputs[col] = st.selectbox(col, options=opts if opts else [""], index=(opts.index(default_val) if opts else 0))

# Build full input row: use user inputs for displayed fields, dataset defaults for hidden fields
full_row = {c: user_inputs.get(c, defaults[c]) for c in feature_cols}
input_df = pd.DataFrame([full_row], columns=feature_cols)

if st.button(" Predict Risk"):
    try:
        pred = model.predict(input_df)[0]
        st.success(f"**Prediction:** {'High Risk' if int(pred) == 1 else 'Low Risk'}")
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)
            if proba.ndim == 2 and proba.shape[1] == 2:
                st.write(f"Probability (High Risk): {float(proba[0,1]):.4f}")
            elif proba.ndim == 2:
                st.write("Class probabilities:", [float(x) for x in proba[0]])
    except Exception as e:
        st.error(f"Prediction failed: {e}")


# 📂 Batch Prediction

st.header("📂 Batch Prediction (CSV)")

uploaded = st.file_uploader("Upload a CSV with any subset of the above fields; missing ones will be auto-filled.", type=["csv"])
if uploaded is not None:
    try:
        batch_df = pd.read_csv(uploaded)

        # Align: add missing columns with defaults; keep only training features order
        for c in feature_cols:
            if c not in batch_df.columns:
                batch_df[c] = defaults[c]
        batch_df = batch_df[feature_cols]

        preds = model.predict(batch_df)
        out = batch_df.copy()
        out["prediction"] = preds

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(batch_df)
            if proba.ndim == 2 and proba.shape[1] == 2:
                out["probability_high_risk"] = proba[:, 1]

        st.subheader("Preview")
        st.dataframe(out.head(50))
        st.download_button(
            "Download Predictions CSV",
            out.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.error(f"Batch prediction failed: {e}")



