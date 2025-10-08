import os, sys
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)
from pathlib import Path
import pandas as pd


#  pour que sa marche soyez sure que vous lancer depuis le chemin relatif à partir du répertoire actuel

REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
print(REPO_ROOT)
MODELS_DIR = Path(REPO_ROOT) / "models"
print(MODELS_DIR)
FEAT_DIR = Path(REPO_ROOT) / "features"
MODEL_PATH = MODELS_DIR / "automl_best_one.pkl"

OUT_DIR = os.path.join(MODELS_DIR, "validation_results")
os.makedirs(OUT_DIR, exist_ok=True)

# ==========================================
# LOAD DATA & MODEL
# ==========================================
print("Loading validation data...")
X_val = np.load(os.path.join(FEAT_DIR, "features_val.npy"))
y_val_raw = np.load(os.path.join(FEAT_DIR, "labels_val.npy"))

print("Loading model and label encoder...")
model = joblib.load(os.path.join(MODELS_DIR, "automl_best_one.pkl"))
le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))

# Encode validation labels
y_val = le.transform(y_val_raw)

print(f"\nValidation set: {X_val.shape[0]} samples, {X_val.shape[1]} features")
print(f"Classes: {dict(zip(*np.unique(y_val_raw, return_counts=True)))}")

# ==========================================
# PREDICTIONS
# ==========================================
print("\nMaking predictions...")
y_pred = model.predict(X_val)
y_pred_proba = model.predict_proba(X_val)

# ==========================================
# METRICS
# ==========================================
print("\n" + "=" * 60)
print("VALIDATION METRICS")
print("=" * 60)

# Overall metrics
accuracy = accuracy_score(y_val, y_pred)
f1_macro = f1_score(y_val, y_pred, average="macro")
f1_weighted = f1_score(y_val, y_pred, average="weighted")
precision_macro = precision_score(y_val, y_pred, average="macro")
recall_macro = recall_score(y_val, y_pred, average="macro")

print(f"\nOverall Performance:")
print(f"  Accuracy:          {accuracy:.4f}")
print(f"  F1-Score (macro):  {f1_macro:.4f}")
print(f"  F1-Score (weighted): {f1_weighted:.4f}")
print(f"  Precision (macro): {precision_macro:.4f}")
print(f"  Recall (macro):    {recall_macro:.4f}")

# Per-class report
print("\n" + "-" * 60)
print("Per-Class Performance:")
print("-" * 60)
report = classification_report(
    y_val, y_pred, target_names=le.classes_, digits=4, output_dict=True
)
print(classification_report(y_val, y_pred, target_names=le.classes_, digits=4))

# Save report as CSV
df_report = pd.DataFrame(report).transpose()
report_path = os.path.join(OUT_DIR, "classification_report.csv")
df_report.to_csv(report_path)
print(f"\nSaved report to: {report_path}")

# ==========================================
# CONFUSION MATRIX
# ==========================================
print("\nGenerating confusion matrix...")
cm = confusion_matrix(y_val, y_pred)
cm_normalized = confusion_matrix(y_val, y_pred, normalize="true")

# Plot confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Raw counts
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=le.classes_,
    yticklabels=le.classes_,
    ax=axes[0],
    cbar_kws={"label": "Count"},
)
axes[0].set_title("Confusion Matrix (Counts)", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Predicted Label", fontsize=12)
axes[0].set_ylabel("True Label", fontsize=12)

# Normalized
sns.heatmap(
    cm_normalized,
    annot=True,
    fmt=".2f",
    cmap="Greens",
    xticklabels=le.classes_,
    yticklabels=le.classes_,
    ax=axes[1],
    cbar_kws={"label": "Proportion"},
    vmin=0,
    vmax=1,
)
axes[1].set_title("Confusion Matrix (Normalized)", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Predicted Label", fontsize=12)
axes[1].set_ylabel("True Label", fontsize=12)

plt.tight_layout()
cm_path = os.path.join(OUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=300, bbox_inches="tight")
print(f"Saved confusion matrix to: {cm_path}")
plt.close()

# ==========================================
# PER-CLASS F1 SCORES (Bar plot)
# ==========================================
print("\nGenerating per-class F1 scores plot...")
f1_per_class = [report[cls]["f1-score"] for cls in le.classes_]

plt.figure(figsize=(10, 6))
bars = plt.bar(le.classes_, f1_per_class, color="steelblue", edgecolor="black")
plt.axhline(
    y=f1_macro,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Macro Avg: {f1_macro:.3f}",
)
plt.xlabel("Class", fontsize=12, fontweight="bold")
plt.ylabel("F1-Score", fontsize=12, fontweight="bold")
plt.title("Per-Class F1-Scores on Validation Set", fontsize=14, fontweight="bold")
plt.xticks(rotation=45, ha="right")
plt.ylim(0, 1.05)
plt.legend()
plt.grid(axis="y", alpha=0.3)

# Add value labels on bars
for bar, score in zip(bars, f1_per_class):
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.02,
        f"{score:.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )

plt.tight_layout()
f1_path = os.path.join(OUT_DIR, "per_class_f1_scores.png")
plt.savefig(f1_path, dpi=300, bbox_inches="tight")
print(f"Saved F1 scores plot to: {f1_path}")
plt.close()

# ==========================================
# CONFIDENCE DISTRIBUTION
# ==========================================
print("\nGenerating confidence distribution plot...")
confidences = np.max(y_pred_proba, axis=1)
correct = y_val == y_pred

plt.figure(figsize=(10, 6))
plt.hist(
    confidences[correct],
    bins=30,
    alpha=0.7,
    label="Correct",
    color="green",
    edgecolor="black",
)
plt.hist(
    confidences[~correct],
    bins=30,
    alpha=0.7,
    label="Incorrect",
    color="red",
    edgecolor="black",
)
plt.xlabel("Prediction Confidence", fontsize=12, fontweight="bold")
plt.ylabel("Frequency", fontsize=12, fontweight="bold")
plt.title("Prediction Confidence Distribution", fontsize=14, fontweight="bold")
plt.legend(fontsize=11)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

conf_path = os.path.join(OUT_DIR, "confidence_distribution.png")
plt.savefig(conf_path, dpi=300, bbox_inches="tight")
print(f"Saved confidence plot to: {conf_path}")
plt.close()

# ==========================================
# MISCLASSIFICATION ANALYSIS
# ==========================================
print("\nAnalyzing misclassifications...")
misclassified = np.where(y_val != y_pred)[0]
print(
    f"\nTotal misclassifications: {len(misclassified)} / {len(y_val)} ({len(misclassified)/len(y_val)*100:.2f}%)"
)

if len(misclassified) > 0:
    # Top confused pairs
    from collections import Counter

    confused_pairs = []
    for idx in misclassified:
        true_label = le.inverse_transform([y_val[idx]])[0]
        pred_label = le.inverse_transform([y_pred[idx]])[0]
        confused_pairs.append((true_label, pred_label))

    top_confused = Counter(confused_pairs).most_common(5)
    print("\nMost Common Misclassifications:")
    for (true_cls, pred_cls), count in top_confused:
        print(f"  {true_cls} → {pred_cls}: {count} times")

# ==========================================
# SUMMARY
# ==========================================
print("\n" + "=" * 60)
print("VALIDATION COMPLETE")
print("=" * 60)
print(f"\nAll results saved to: {OUT_DIR}")
print(f"\nKey Metrics:")
print(f"  - Accuracy:  {accuracy:.4f}")
print(f"  - F1-macro:  {f1_macro:.4f}")
print(f"  - Precision: {precision_macro:.4f}")
print(f"  - Recall:    {recall_macro:.4f}")
print("\nGenerated files:")
print(f"  1. classification_report.csv")
print(f"  2. confusion_matrix.png")
print(f"  3. per_class_f1_scores.png")
print(f"  4. confidence_distribution.png")
