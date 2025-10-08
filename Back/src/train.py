import os
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import time

# Config
FEAT_DIR = r"C:\Users\htagi\bacteria_features"
OUT_DIR = r"C:\Users\htagi\bacterias_models"
os.makedirs(OUT_DIR, exist_ok=True)
RANDOM_STATE = 42
N_JOBS = -1

# Load data
print("Loading data...")
X = np.load(os.path.join(FEAT_DIR, "features_train.npy"))
y_raw = np.load(os.path.join(FEAT_DIR, "labels_train.npy"))

# Label encode
le = LabelEncoder()
y = le.fit_transform(y_raw)
joblib.dump(le, os.path.join(OUT_DIR, "label_encoder.pkl"))

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Classes: {dict(zip(*np.unique(y_raw, return_counts=True)))}")

# CV setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# ==========================================
# Model 1: Random Forest (Fast baseline)
# ==========================================
print("\n" + "=" * 60)
print("Training Random Forest (fast baseline)...")
start = time.time()

pipe_rf = ImbPipeline(
    [
        ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=3)),
        (
            "clf",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS,
                verbose=0,
            ),
        ),
    ]
)

# Manual CV to track performance
rf_scores = []
for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y), 1):
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    pipe_rf.fit(X_tr, y_tr)
    y_pred = pipe_rf.predict(X_val)
    f1 = f1_score(y_val, y_pred, average="macro")
    rf_scores.append(f1)
    print(f"  Fold {fold}: F1-macro = {f1:.4f}")

rf_mean = np.mean(rf_scores)
print(f"RF Mean F1-macro: {rf_mean:.4f} (±{np.std(rf_scores):.4f})")
print(f"Time: {(time.time() - start)/60:.1f} min")

# Refit on all data
pipe_rf.fit(X, y)

# ==========================================
# Model 2: XGBoost (Better performance)
# ==========================================
print("\n" + "=" * 60)
print("Training XGBoost (optimized for imbalanced data)...")
start = time.time()

# Calculate scale_pos_weight for each class
class_counts = np.bincount(y)
scale_weights = len(y) / (len(class_counts) * class_counts)

pipe_xgb = ImbPipeline(
    [
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=3)),
        (
            "clf",
            XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS,
                tree_method="hist",  # Much faster
                verbosity=0,
            ),
        ),
    ]
)

# Manual CV
xgb_scores = []
for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y), 1):
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    pipe_xgb.fit(X_tr, y_tr)
    y_pred = pipe_xgb.predict(X_val)
    f1 = f1_score(y_val, y_pred, average="macro")
    xgb_scores.append(f1)
    print(f"  Fold {fold}: F1-macro = {f1:.4f}")

xgb_mean = np.mean(xgb_scores)
print(f"XGB Mean F1-macro: {xgb_mean:.4f} (±{np.std(xgb_scores):.4f})")
print(f"Time: {(time.time() - start)/60:.1f} min")

# Refit on all data
pipe_xgb.fit(X, y)

# ==========================================
# Select best model
# ==========================================
print("\n" + "=" * 60)
if xgb_mean > rf_mean:
    best_model = pipe_xgb
    best_name = "XGBoost"
    best_score = xgb_mean
else:
    best_model = pipe_rf
    best_name = "RandomForest"
    best_score = rf_mean

print(f"Best model: {best_name} (F1-macro={best_score:.4f})")

# Save
MODEL_PATH = os.path.join(OUT_DIR, "automl_best_model.pkl")
joblib.dump(best_model, MODEL_PATH)
print(f"Saved to: {MODEL_PATH}")

# Final validation on full dataset
y_pred_final = best_model.predict(X)
print("\nFull dataset performance:")
print(classification_report(y, y_pred_final, target_names=le.classes_))
