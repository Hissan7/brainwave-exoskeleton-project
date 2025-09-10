import numpy as np
import random
from joblib import load
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from load_bci_data import (
    load_and_segment_trials, filter_left_right,
    notch_filter, bandpass_8_30, crop_time_window,
    FS, BEST_START, BEST_DUR
)

# ---------------------------
# Load model
# ---------------------------
MODEL_PATH = "models/csp_svm_final.joblib"
payload = load(MODEL_PATH)
csp = payload["csp"]
clf = payload["clf"]

print(f"✓ Loaded model from {MODEL_PATH}")
print(f"Best window: {payload['best_start']}–{payload['best_start']+payload['best_dur']}s")
print(f"Params: nc={payload['n_components']}, C={payload['C']}, cov={payload['cov_est']}, reg={payload['reg']}")

# ---------------------------
# Load & preprocess dataset
# ---------------------------
X, y = load_and_segment_trials("data/A01T.mat")
X, y = filter_left_right(X, y)
X = notch_filter(X, fs=FS, freqs=(60, 120))
X = bandpass_8_30(X, fs=FS)
X = crop_time_window(X, fs=FS, start_s=BEST_START, dur_s=BEST_DUR)

# ---------------------------
# Evaluate on ALL trials
# ---------------------------
X_csp = csp.transform(X)
y_pred = clf.predict(X_csp)

acc = accuracy_score(y, y_pred)
print(f"\n=== Final Model Evaluation on ALL Trials ===")
print(f"Accuracy: {acc:.3f}")

cm = confusion_matrix(y, y_pred)
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y, y_pred, target_names=["Left", "Right"]))

# Plot confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred Left", "Pred Right"],
            yticklabels=["True Left", "True Right"])
plt.title("Confusion Matrix (Final Model)")
plt.ylabel("True")
plt.xlabel("Predicted")
plt.tight_layout()
plt.show()

# ---------------------------
# Test on random trials
# ---------------------------
print("\n=== Testing random trials ===")
for idx in random.sample(range(len(y)), 10):  # pick 10 random trials
    trial = X[idx:idx+1]  # keep shape (1, n_channels, n_time)
    true_label = "Left" if y[idx] == 1 else "Right"

    X_csp = csp.transform(trial)
    pred = clf.predict(X_csp)[0]
    prob = clf.predict_proba(X_csp)[0]

    pred_label = "Left" if pred == 1 else "Right"
    print(f"True: {true_label:>5} | Pred: {pred_label:>5} | Prob={prob}")
