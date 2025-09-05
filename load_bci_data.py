import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import os
from joblib import dump, load
from scipy.signal import butter, filtfilt, iirnotch
from mne.decoding import CSP
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold

# ---------------------------
# Constants
# ---------------------------
FS = 250               # Hz (BCI IV-2a)
BEST_START = 1.5       # seconds (window start you found)
BEST_DUR = 2.5         # seconds (window duration you found)
DEFAULT_COV = 'concat' # <<< change to 'oas' or 'ledoit_wolf' later if you want
DEFAULT_REG = 'ledoit_wolf'


# ---------------------------
# 1) Load and segment trials
# ---------------------------
def load_and_segment_trials(filepath):
    mat = scipy.io.loadmat(filepath, struct_as_record=False, squeeze_me=True)
    all_runs = mat['data']
    all_X, all_y = [], []

    #Only use runs 3 to 8 (runs 0–2 are warmup/setup and don’t have usable labels).
    #So we're processing 6 actual motor imagery runs.
    for run_idx in range(3, 9):
        run = all_runs[run_idx]
        eeg = run.X              # (samples, channels)
        labels = run.y           # (48,)
        trial_starts = run.trial # (48,)
        n_trials = len(labels)
        n_channels = eeg.shape[1]
        n_timepoints = 1125      # 4.5s * 250Hz

        trials = np.zeros((n_trials, n_channels, n_timepoints))
        for i in range(n_trials):
            start = int(trial_starts[i])
            trials[i] = eeg[start:start + n_timepoints].T  # transpose to (channels, time)

        all_X.append(trials)
        all_y.append(labels)

    X = np.concatenate(all_X, axis=0)  # (288, 25, 1125)
    y = np.concatenate(all_y, axis=0)  # (288,)
    print("Final EEG shape:", X.shape)
    print("Final label shape:", y.shape)
    print("Unique classes:", np.unique(y))
    return X, y

# ---------------------------------------
# 2) Keep only left(1) / right(2) trials
# ---------------------------------------

# function to filter the data to keep only left-hand and right-hand trials (labels 1 and 2)
# also plot a few EEG signals from channels C3 and C4, which are over the motor cortex.
# y is a numpy array of all the labels e.g. [1,2,4,2,1]
# hence we need to filter out the trials that only correspond to 1 & 2
# Labels: 1 = left hand, 2 = right hand
def filter_left_right(X, y):
    mask = (y == 1) | (y == 2)

# Xf, yf = X[mask], y[mask]
# Xf = X[mask] indexing array with boolean mask. NumPy keeps only elements where mask is True i.e its label 1 or label 2
# yf = y[mask] indexing array with the corresponding labels 
    Xf, yf = X[mask], y[mask]
    print(f"Filtered EEG shape: {Xf.shape}")
    print(f"Filtered labels: {np.unique(yf)}")
    return Xf, yf

# -----------------------
# 3) Preprocessing helpers
# -----------------------
def notch_filter(data, fs=250, freqs=(50, 100), Q=30):
    """Notch out mains and first harmonic (UK/EU: 50/100; US: 60/120)."""
    out = data.copy()
    for f0 in freqs:
        b, a = iirnotch(w0=f0/(fs/2), Q=Q)
        out = filtfilt(b, a, out, axis=-1)
    return out

def bandpass_8_30(data, fs=250, low=8, high=30, order=4):
    """Keep motor bands (mu/beta)."""
    b, a = butter(order, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, data, axis=-1)

def zscore_per_channel(X):
    """Standardize each trial/channel across time."""
    mean = X.mean(axis=-1, keepdims=True)
    std  = X.std(axis=-1, keepdims=True) + 1e-8
    return (X - mean) / std

def crop_time_window(X, fs=250, start_s=0.5, dur_s=2.0):
    start = int(start_s * fs)
    length = int(dur_s * fs)
    return X[:, :, start:start+length]   # (trials, channels, time_cropped)

# -----------------------
# 4) (Optional) Plotting
# -----------------------
# A list of the 25 channel names used in BCI IV 2a dataset (in order)
# getting C3 and C4 because these are responsible for left and right hand movements
def get_channel_indices():
    names = [
        'Fz','FC1','FC2','Cz','C3','C4','CP1','CP2','Pz',
        'C5','C1','C2','C6','CP5','CP3','CP4','CP6',
        'FC5','FC3','FC4','FC6','POz','O1','Oz','O2'
    ]
    return names.index('C3'), names.index('C4')

# Now due to contralateral control we should expect : 
# Stronger signal in C4 when imagining left-hand movement
# Stronger signal in C3 when imagining right-hand movement
def plot_example_trial(X, y, trial_idx):
    c3_idx, c4_idx = get_channel_indices()
    sig = X[trial_idx]; label = y[trial_idx]
    plt.figure(figsize=(10,4))
    plt.plot(sig[c3_idx], label='C3 (left motor cortex)')
    plt.plot(sig[c4_idx], label='C4 (right motor cortex)')
    plt.title(f"Trial {trial_idx} – {'Left' if label==1 else 'Right'} hand")
    plt.xlabel("Time (samples)"); plt.ylabel("EEG (µV)")
    plt.legend(); plt.tight_layout(); plt.show()

# -----------------------
# 5) Baseline classifiers
# -----------------------
def csp_lda_cv(X, y, n_components=6, cov_est=DEFAULT_COV, reg=DEFAULT_REG):
    csp = CSP(n_components=n_components, reg=reg, log=True, cov_est=cov_est)
    clf = LDA()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tr, te in cv.split(X, y):
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        Xtr_csp = csp.fit_transform(Xtr, ytr)
        Xte_csp = csp.transform(Xte)
        clf.fit(Xtr_csp, ytr)
        scores.append(clf.score(Xte_csp, yte))
    return float(np.mean(scores)), float(np.std(scores))

def csp_svm_cv(X, y, n_components=6, C=2.0, cov_est=DEFAULT_COV, reg=DEFAULT_REG):
    csp = CSP(n_components=n_components, reg=reg, log=True, cov_est=cov_est)
    clf = SVC(kernel='rbf', C=C, gamma='scale')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tr, te in cv.split(X, y):
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        Xtr_csp = csp.fit_transform(Xtr, ytr)
        Xte_csp = csp.transform(Xte)
        clf.fit(Xtr_csp, ytr)
        scores.append(clf.score(Xte_csp, yte))
    return float(np.mean(scores)), float(np.std(scores))

# Goal: Given EEG data X and labels y, measure how well CSP+SVM can classify left vs right.
# CSP (Common Spatial Patterns): Finds channel combinations that maximize the variance difference between classes — basically “spatial filters” tuned to left vs right motor activity.
# SVM (Support Vector Machine): Takes the CSP features and learns a decision boundary to separate the two classes.
# Cross-validation: StratifiedKFold splits the dataset into train/test sets multiple times to avoid overfitting and to give a reliable average accuracy.
# Output: Mean accuracy and standard deviation across folds.

# So this function answers:
# “If I use these CSP settings (n_components) and SVM settings (C), how accurate is the model on this data?”

def eval_csp_svm(X, y, n_components=6, C=2.0, cov_est=DEFAULT_COV, reg=DEFAULT_REG):
    """Helper for grid search; identical to csp_svm_cv but named for clarity."""
    return csp_svm_cv(X, y, n_components=n_components, C=C, cov_est=cov_est, reg=reg)


# Goal: Figure out which time slice of each trial gives the best accuracy.
# Trials are a few seconds long, but motor imagery might only be strongest in a specific window (e.g., starting 0.5s after the cue for 2 seconds).
# starts: List of possible start times after the trial begins.
# durs: Possible window lengths.
# For each (start, duration) combination:
# Crop the EEG to that time range using crop_time_window.
# Run eval_csp_svm to see how accurate classification is in that window.
# Keep track of the best combination.
# Output: Prints accuracy for every tested window and the best one found.

# So this answers:
# “When in time should I look at the EEG for the clearest left vs right separation?”

# -----------------------
# 6) Time-window grid search (optional)
# -----------------------

def grid_search_time(X_full, y, fs=FS):
    starts = [0.0, 0.5, 1.0, 1.5]
    durs   = [1.5, 2.0, 2.5, 3.0]
    best = None
    for s in starts:
        for d in durs:
            Xc = crop_time_window(X_full, fs, s, d)
            if Xc.shape[-1] <= 10:
                continue
            mean, std = eval_csp_svm(
                Xc, y, n_components=6, C=2.0,
                cov_est=DEFAULT_COV, reg=DEFAULT_REG
            )
            print(
                f"start={s:.1f}s dur={d:.1f}s -> acc={mean:.3f} ± {std:.3f} "
                f"(nT={Xc.shape[0]}, nCh={Xc.shape[1]}, nTps={Xc.shape[2]})"
            )
            if best is None or mean > best[0]:
                best = (mean, std, s, d)
    if best:
        print(f"BEST window: start={best[2]:.1f}s dur={best[3]:.1f}s -> {best[0]:.3f} ± {best[1]:.3f}")
    return best


# -----------------------
# 7) Train & save final model
# -----------------------

def train_and_save_final_model(
    X, y, n_components=8, C=2.0, cov_est=DEFAULT_COV, reg=DEFAULT_REG,
    save_path="models/csp_svm_final.joblib"
):
    """
    Train CSP+SVM on ALL (preprocessed, CROPPED) trials and save the fitted objects + metadata.
    """
    csp = CSP(n_components=n_components, reg=reg, log=True, cov_est=cov_est)
    X_csp = csp.fit_transform(X, y)

    clf = SVC(kernel='rbf', C=C, gamma='scale', probability=True)
    clf.fit(X_csp, y)

    payload = {
        "csp": csp,
        "clf": clf,
        "fs": FS,
        "best_start": BEST_START,
        "best_dur": BEST_DUR,
        "cov_est": cov_est,   # pooling used ('concat' or 'epoch')
        "reg": reg,           # shrinkage used (None/OAS/LW/float)
        "n_components": n_components,
        "C": C,
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dump(payload, save_path)
    print(f"✓ Saved final model to {save_path}")

def load_model_and_predict(trials, model_path="models/csp_svm_final.joblib"):
    """
    trials: (n_trials, n_channels, n_time) already preprocessed & CROPPED the same way.
    returns: predicted labels and probabilities
    """
    payload = load(model_path)
    csp = payload["csp"]; clf = payload["clf"]
    X_csp = csp.transform(trials)
    preds = clf.predict(X_csp)
    probs = clf.predict_proba(X_csp) if hasattr(clf, "predict_proba") else None
    return preds, probs


# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    # 1) Load & keep only left/right
    X, y = load_and_segment_trials("data/A01T.mat")
    X, y = filter_left_right(X, y)

    # 2) Minimal preprocessing (US mains)
    X = notch_filter(X, fs=FS, freqs=(60, 120))  # US
    X = bandpass_8_30(X, fs=FS)

    # 3) Baselines on full 4.5 s trials (no crop)
    print("\n=== Baseline (no crop) ===")
    m, s = csp_lda_cv(X, y, n_components=6, cov_est=DEFAULT_COV, reg=DEFAULT_REG)
    print(f"CSP+LDA (nc=6, cov={DEFAULT_COV}, reg={DEFAULT_REG}): {m:.3f} ± {s:.3f}")
    m, s = csp_svm_cv(X, y, n_components=6, C=2.0, cov_est=DEFAULT_COV, reg=DEFAULT_REG)
    print(f"CSP+SVM (nc=6, C=2.0, cov={DEFAULT_COV}, reg={DEFAULT_REG}): {m:.3f} ± {s:.3f}")

    # 4) (Optional) quick crop search (you already found 1.5s/2.5s)
    DO_CROP_SEARCH = False
    if DO_CROP_SEARCH:
        print("\n=== Quick crop window search ===")
        grid_search_time(X, y, fs=FS)

    # 5) Apply your best window (lock 1.5–4.0 s)
    X = crop_time_window(X, fs=FS, start_s=BEST_START, dur_s=BEST_DUR)
    print(f"\nCropped EEG shape: {X.shape} (start={BEST_START}s, dur={BEST_DUR}s)")

    # 6) Baselines on cropped trials
    print("\n=== Baseline (cropped) ===")
    m, s = csp_lda_cv(X, y, n_components=8, cov_est=DEFAULT_COV, reg=DEFAULT_REG)
    print(f"CSP+LDA (nc=8, cov={DEFAULT_COV}, reg={DEFAULT_REG}): {m:.3f} ± {s:.3f}")
    for nc in [4, 6, 8]:
        for C in [0.5, 1.0, 2.0, 4.0]:
            m, s = csp_svm_cv(X, y, n_components=nc, C=C, cov_est=DEFAULT_COV, reg=DEFAULT_REG)
            print(f"CSP+SVM (nc={nc}, C={C}, cov={DEFAULT_COV}, reg={DEFAULT_REG}): {m:.3f} ± {s:.3f}")

    # 7) Train on ALL cropped data and save the final model
    BEST_NC = 8
    BEST_C = 2.0
    print("\n=== Train final model on all cropped data ===")
    train_and_save_final_model(
        X, y,
        n_components=BEST_NC,
        C=BEST_C,
        cov_est=DEFAULT_COV,      # pooling ('concat' now)
        reg=DEFAULT_REG,          # shrinkage (None now)
        save_path="models/csp_svm_final.joblib"
    )

    # Optional sanity plot:
    # plot_example_trial(X, y, trial_idx=0)
    









