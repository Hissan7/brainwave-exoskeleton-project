# ROUGH CODE JUST TO SHOW THE KEYS OF THE DATA SET ------------------

# import scipy.io
# import numpy as np

# def load_bci_data(filepath):
#     mat = scipy.io.loadmat(filepath)

#     print("Available keys:", mat.keys())  # ← Add this line

#     # You can return early just to see the keys for now
#     return None, None

# if __name__ == "__main__":
#     load_bci_data("data/A01T.mat")

# ROUGH SCRIPT TO SHOW WHAT TYPE OF DATASTRUCTURE IT IS ------------------

# import scipy.io
# import numpy as np

# def load_bci_data(filepath):
#     mat = scipy.io.loadmat(filepath, struct_as_record=False, squeeze_me=True)

#     print("Top-level keys:", mat.keys())
#     print("Type of mat['data']:", type(mat['data']))
#     print("mat['data'] content:", mat['data'])

#     return None, None

# if __name__ == "__main__":
#     load_bci_data("data/A01T.mat")

# ------------------------------------------------------------------------

# import scipy.io
# import numpy as np

# def load_bci_data(filepath):
#     # Load the .mat file
#     mat = scipy.io.loadmat(filepath, struct_as_record=False, squeeze_me=True)

#     for i in range(9):
#         # Extract the first run (you can loop through others later)
#         run = mat['data'][i]  # First of 9 runs

#         # Extract EEG data and labels
#         X = run.X             # EEG data: shape (trials, channels, time)
#         y = run.y             # Labels: shape (trials,)

#         print("EEG shape:", X.shape)  # e.g., (48, 22, 1125)
#         print("Labels shape:", y.shape)
#         print("Unique classes:", np.unique(y))
#         print(f"That index was {i}")
#         print("")

#     return X, y
    

# if __name__ == "__main__":
#     X, y = load_bci_data("data/A01T.mat")

# ------------------------------------------------------------------------

# import scipy.io
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import butter, filtfilt, iirnotch
# from mne.decoding import CSP
# from sklearn.svm import SVC
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# from sklearn.model_selection import StratifiedKFold

# def load_and_segment_trials(filepath):
#     mat = scipy.io.loadmat(filepath, struct_as_record=False, squeeze_me=True)
    
#     all_runs = mat['data']
#     all_X = []
#     all_y = []

#     #Only use runs 3 to 8 (runs 0–2 are warmup/setup and don’t have usable labels).
#     #So we're processing 6 actual motor imagery runs.
#     for run_idx in range(3, 9): 
#         run = all_runs[run_idx]

#         eeg = run.X              # shape: (samples, channels)
#         labels = run.y           # shape: (48,)
#         trial_starts = run.trial # shape: (48,)
#         n_trials = len(labels)
#         n_channels = eeg.shape[1]
#         n_timepoints = 1125  # each trial is 4.5s * 250Hz = 1125 samples

#         # Preallocate: (trials, channels, time)
#         trials = np.zeros((n_trials, n_channels, n_timepoints))

#         for i in range(n_trials):
#             start = trial_starts[i]
#             trials[i] = eeg[start:start+n_timepoints].T  # transpose to (channels, time)

#         all_X.append(trials)
#         all_y.append(labels)

#     # Concatenate all 6 runs together
#     X = np.concatenate(all_X, axis=0)  # shape: (288, 25, 1125)
#     y = np.concatenate(all_y, axis=0)  # shape: (288,)

#     print("Final EEG shape:", X.shape)
#     print("Final label shape:", y.shape)
#     print("Unique classes:", np.unique(y))

#     return X, y


# # function to filter the data to keep only left-hand and right-hand trials (labels 1 and 2)
# # also plot a few EEG signals from channels C3 and C4, which are over the motor cortex.
# # y is a numpy array of all the labels e.g. [1,2,4,2,1]
# # hence we need to filter out the trials that only correspond to 1 & 2
# def filter_left_right(X, y):
#     # Labels: 1 = left hand, 2 = right hand
#     mask = (y == 1) | (y == 2)

    
#     X_filtered = X[mask] # indexing array with boolean mask. NumPy keeps only elements where mask is True i.e its label 1 or label 2
#     y_filtered = y[mask] # indexing array with the corresponding labels 

#     print(f"Filtered EEG shape: {X_filtered.shape}")
#     print(f"Filtered labels: {np.unique(y_filtered)}")

#     return X_filtered, y_filtered

# def get_channel_indices():
#     # A list of the 25 channel names used in BCI IV 2a dataset (in order)
#     # getting C3 and C4 because these are responsible for left and right hand movements
#     channel_names = [
#         'Fz', 'FC1', 'FC2', 'Cz', 'C3', 'C4', 'CP1', 'CP2', 'Pz',
#         'C5', 'C1', 'C2', 'C6', 'CP5', 'CP3', 'CP4', 'CP6',
#         'FC5', 'FC3', 'FC4', 'FC6', 'POz', 'O1', 'Oz', 'O2'
#     ]
#     return channel_names.index('C3'), channel_names.index('C4')

# #plotting the data
# def plot_example_trial(X, y, trial_idx):
#     c3_idx, c4_idx = get_channel_indices()

#     signal = X[trial_idx]
#     label = y[trial_idx]

#     plt.figure(figsize=(10, 4))
#     plt.plot(signal[c3_idx], label='C3 (left motor cortex)')
#     plt.plot(signal[c4_idx], label='C4 (right motor cortex)')
#     plt.title(f"Trial {trial_idx} - Label: {'Left Hand' if label == 1 else 'Right Hand'}")
#     plt.xlabel("Time (samples)")
#     plt.ylabel("EEG amplitude (µV)")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

#     # Now due to contralateral control we should expect : 
#     # Stronger signal in C4 when imagining left-hand movement
#     # Stronger signal in C3 when imagining right-hand movement
    
# def bandpass_8_30(data, fs=250, low=8, high=30, order=4):
#     """
#     data: (trials, channels, time)
#     returns filtered data with the same shape
#     """
#     b, a = butter(order, [low/(fs/2), high/(fs/2)], btype='band')
#     # filtfilt along time axis
#     return filtfilt(b, a, data, axis=-1)

# def csp_lda_baseline(X, y, n_components=6):
#     csp = CSP(n_components=n_components, reg=None, log=True, cov_est='concat')
#     clf = LDA()
#     cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#     scores = []
#     for train_idx, test_idx in cv.split(X, y):
#         X_tr, X_te = X[train_idx], X[test_idx]
#         y_tr, y_te = y[train_idx], y[test_idx]

#         X_tr_csp = csp.fit_transform(X_tr, y_tr)
#         X_te_csp = csp.transform(X_te)

#         clf.fit(X_tr_csp, y_tr)
#         scores.append(clf.score(X_te_csp, y_te))

#     print(f"CSP+LDA 5-fold accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

# def csp_svm_baseline(X, y, n_components=6):
#     csp = CSP(n_components=n_components, reg=None, log=True, cov_est='concat')
#     clf = SVC(kernel='rbf', C=1.0, gamma='scale')  # good default

#     cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     scores = []
#     for tr, te in cv.split(X, y):
#         Xtr, Xte = X[tr], X[te]
#         ytr, yte = y[tr], y[te]
#         Xtr_csp = csp.fit_transform(Xtr, ytr)
#         Xte_csp = csp.transform(Xte)
#         clf.fit(Xtr_csp, ytr)
#         scores.append(clf.score(Xte_csp, yte))
#     print(f"CSP+SVM 5-fold accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

# # in the UK/EU, mains = 50 Hz (and its harmonic 100 Hz). In the US, use 60/120 instead.
# def notch_filter(data, fs=250, freqs=(60, 120), Q=30):
#     # data: (trials, channels, time)
#     out = data.copy()
#     for f0 in freqs:
#         b, a = iirnotch(w0=f0/(fs/2), Q=Q)
#         out = filtfilt(b, a, out, axis=-1)
#     return out

# def zscore_per_channel(X):
#     # X: (trials, channels, time)
#     mean = X.mean(axis=-1, keepdims=True)
#     std  = X.std(axis=-1, keepdims=True) + 1e-8
#     return (X - mean) / std


# if __name__ == "__main__":
#     X, y = load_and_segment_trials("data/A01T.mat")
#     X, y = filter_left_right(X, y)
#     X = notch_filter(X,fs=250,freqs=(60,120))
#     X = bandpass_8_30(X)
#     X = zscore_per_channel(X)
#     plot_example_trial(X, y, trial_idx=0)
#     plot_example_trial(X, y, trial_idx=1)
#     csp_lda_baseline(X, y)
#     csp_svm_baseline(X, y)

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch
from mne.decoding import CSP
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold

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
def csp_lda_baseline(X, y, n_components=6):
    csp = CSP(n_components=n_components, reg=None, log=True, cov_est='concat')
    clf = LDA()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tr, te in cv.split(X, y):
        Xtr, Xte = X[tr], X[te]; ytr, yte = y[tr], y[te]
        Xtr_csp = csp.fit_transform(Xtr, ytr)
        Xte_csp = csp.transform(Xte)
        clf.fit(Xtr_csp, ytr)
        scores.append(clf.score(Xte_csp, yte))
    print(f"CSP+LDA 5-fold accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

def csp_svm_baseline(X, y, n_components=6, C = 2.0):
    csp = CSP(n_components=n_components, reg=None, log=True, cov_est='concat')
    clf = SVC(kernel='rbf', C=C, gamma='scale')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    accuracies = []
    for tr, te in cv.split(X, y):
        Xtr, Xte = X[tr], X[te]; ytr, yte = y[tr], y[te]
        Xtr_csp = csp.fit_transform(Xtr, ytr)
        Xte_csp = csp.transform(Xte)
        clf.fit(Xtr_csp, ytr)
        scores.append(clf.score(Xte_csp, yte))
    print(f"CSP+SVM 5-fold accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}")


# Goal: Given EEG data X and labels y, measure how well CSP+SVM can classify left vs right.
# CSP (Common Spatial Patterns): Finds channel combinations that maximize the variance difference between classes — basically “spatial filters” tuned to left vs right motor activity.
# SVM (Support Vector Machine): Takes the CSP features and learns a decision boundary to separate the two classes.
# Cross-validation: StratifiedKFold splits the dataset into train/test sets multiple times to avoid overfitting and to give a reliable average accuracy.
# Output: Mean accuracy and standard deviation across folds.

# So this function answers:
# “If I use these CSP settings (n_components) and SVM settings (C), how accurate is the model on this data?”

def eval_csp_svm(X, y, n_components=6, C=2.0, cov_est='concat'):
    csp = CSP(n_components=n_components, reg=None, log=True, cov_est=cov_est)
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
    
def grid_search_time(X_full, y, fs=250):
    starts = [0.0, 0.5, 1.0, 1.5]      # seconds after trial start
    durs   = [1.5, 2.0, 2.5, 3.0]      # window length (s)
    best = None
    for s in starts:
        for d in durs:
            Xc = crop_time_window(X_full, fs, s, d)
            if Xc.shape[-1] <= 10:   # too short -> skip
                continue
            mean, std = eval_csp_svm(Xc, y, n_components=6, C=2.0, cov_est='concat')
            print(f"start={s:.1f}s dur={d:.1f}s -> acc={mean:.3f} ± {std:.3f} (nT={Xc.shape[0]}, nCh={Xc.shape[1]}, nTps={Xc.shape[2]})")
            if best is None or mean > best[0]:
                best = (mean, std, s, d)
    print(f"BEST window: start={best[2]:.1f}s dur={best[3]:.1f}s -> {best[0]:.3f} ± {best[1]:.3f}")

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    # 1) Load & keep only left/right
    X, y = load_and_segment_trials("data/A01T.mat")
    X, y = filter_left_right(X, y)

    # 2) Minimal preprocessing (US mains)
    X = notch_filter(X, fs=250, freqs=(60, 120))  # US
    X = bandpass_8_30(X, fs=250)

    # 3) Baselines
    print("\n=== Baseline (no crop, no z-score) ===")
    csp_lda_baseline(X, y)
    csp_svm_baseline(X, y, n_components=6)  # default

    # 4) Small, tidy sweep for CSP+SVM (components × C)
    print("\n=== CSP+SVM small sweep ===")
    for nc in [4, 6, 8]:
        for C in [0.5, 1.0, 2.0, 4.0]:
            mean, std = eval_csp_svm(X, y, n_components=nc, C=C, cov_est='concat')
            print(f"nc={nc}, C={C} -> {mean:.3f} ± {std:.3f}")

    # 5) (Optional) quick crop search — toggle this flag if you want to try windows
    DO_CROP_SEARCH = False
    if DO_CROP_SEARCH:
        print("\n=== Quick crop window search ===")
        starts = [0.0, 0.5, 1.0, 1.5]
        durs   = [1.5, 2.0, 2.5, 3.0]
        best = None
        for s in starts:
            for d in durs:
                Xc = crop_time_window(X, fs=250, start_s=s, dur_s=d)
                if Xc.shape[-1] < 100:  # ignore tiny windows
                    continue
                mean, std = eval_csp_svm(Xc, y, n_components=6, C=2.0)
                print(f"start={s:.1f}s, dur={d:.1f}s -> {mean:.3f} ± {std:.3f}")
                if best is None or mean > best[0]:
                    best = (mean, std, s, d)
        if best:
            print(f"BEST: start={best[2]:.1f}s, dur={best[3]:.1f}s -> {best[0]:.3f} ± {best[1]:.3f}")









