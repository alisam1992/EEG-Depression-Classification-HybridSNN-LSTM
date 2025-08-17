"""
EEG preprocessing & modeling.
- Loads .mat via SciPy/MNE
- Extracts real channel names and DROPS CB1, CB2, HEOG, VEOG -> 62 channels
- Splits continuous EEG into eyes-open / eyes-closed using events
- Applies basic MNE preprocessing
- Saves subject HDF5 under ./datasets/
- Aggregates "open" into ./open data/data.h5
- Windows and trains a CNN+TCN model (4-class by default)
"""

import os
import gc
from typing import Tuple, Optional, List

import numpy as np
import mne
from scipy.io import loadmat
import scipy.signal
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

# Keras / TF
from keras.layers import (
    Dense, concatenate, Dropout, Conv1D, Flatten, Conv2D, Reshape,
    LSTM, LeakyReLU, MaxPooling1D, Input
)
from keras.models import Model
from keras.initializers import glorot_normal
from tcn import TCN
import keras.backend as K
from sklearn.utils.class_weight import compute_class_weight
from keras import optimizers, metrics
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from keras.backend import clear_session
from tensorflow.keras import optimizers

# Metrics
from imblearn.metrics import sensitivity_specificity_support
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error, r2_score

import warnings
warnings.filterwarnings('ignore')

# ------------------------------
# Configuration
# ------------------------------
BAD_CHS = {"CB1", "CB2", "HEOG", "VEOG"}   # will be dropped to enforce 62 channels
OPEN_CODES  = {2, 4, 6, 12, 14, 16}
CLOSE_CODES = {1, 3, 5, 11, 13, 15}

RAW_DIR = './dataset/'          # where raw .mat files live
SUBJ_H5_DIR = './datasets/'     # where per-subject H5 is saved
OPEN_COMBINED_H5 = './open data/data.h5'   # combined open data
LABELS_XLSX = './labels.xlsx'   # labels file

os.makedirs(SUBJ_H5_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OPEN_COMBINED_H5), exist_ok=True)
os.makedirs('./model results/csv results', exist_ok=True)

# Subject list & filename pattern
matNums = [*range(507, 544), *range(545, 571), *range(573, 629)]
name_tpl = '{}_Depression_REST'  # -> 507_Depression_REST
EXT = '.mat'                     # adjust if using .set
TARGET_FS_DEFAULT = 250.0        # fallback if sampling rate missing


# ------------------------------
# Loader utilities (MNE / SciPy)
# ------------------------------
def _events_from_eeglab_struct(EEG) -> Optional[np.ndarray]:
    """Extract MNE-style events array from EEGLAB struct if present."""
    events = None
    try:
        ev = EEG.event
        onset = []
        codes = []
        for e in np.atleast_1d(ev):
            lat = getattr(e, "latency", None)
            typ = getattr(e, "type", None)
            if lat is not None and typ is not None:
                try:
                    onset.append(int(lat))
                    codes.append(int(typ) if str(typ).isdigit() else hash(str(typ)) % 1000)
                except Exception:
                    continue
        if onset:
            events = np.c_[np.array(onset, int), np.zeros(len(onset), int), np.array(codes, int)]
    except Exception:
        pass
    return events


def _chan_names_from_chanlocs(chanlocs, n_ch: int) -> List[str]:
    """Extract channel names from EEG.chanlocs if available."""
    ch_names: List[str] = []
    if isinstance(chanlocs, np.ndarray):
        for c in chanlocs:
            lab = getattr(c, "labels", None)
            if lab is None:
                lab = getattr(c, "label", None)
            ch_names.append(str(lab) if lab is not None else f"ch{len(ch_names)+1}")
    if not ch_names:
        ch_names = [f"ch{i+1}" for i in range(n_ch)]
    return ch_names


def load_eeg_file(path: str, preload: bool = True) -> Tuple[np.ndarray, float, Optional[np.ndarray], List[str]]:
    """
    Load EEG from .set (via MNE) or .mat (via SciPy). Returns:
      data    : (n_channels, n_samples)
      sfreq   : float
      events  : (n_events, 3) or None (MNE style [onset, 0, code])
      ch_names: list of channel names (after dropping BAD_CHS)
    Applies name-based drop of BAD_CHS so outputs have consistent channel count (62).
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".set":
        raw = mne.io.read_raw_eeglab(path, preload=preload, verbose="ERROR")
        # drop bad channels by name if present
        bads = [x for x in BAD_CHS if x in raw.ch_names]
        if bads:
            raw.drop_channels(bads)
        data = raw.get_data()
        sfreq = float(raw.info.get("sfreq", np.nan))
        try:
            ev, _ = mne.events_from_annotations(raw, verbose="ERROR")
        except Exception:
            ev = None
        ch_names = list(raw.ch_names)
        return data, sfreq, (ev if ev is not None and len(ev) else None), ch_names

    elif ext == ".mat":
        mat = loadmat(path, squeeze_me=True, struct_as_record=False)

        if "EEG" not in mat:
            # fallback: try common direct keys (without names)
            data = None
            sfreq = np.nan
            for key in ["data", "signal", "X", "eeg"]:
                if key in mat and hasattr(mat[key], "ndim") and mat[key].ndim == 2:
                    data = np.asarray(mat[key])
                    break
            if data is None:
                raise ValueError(f"Expected 'EEG' struct or 2D array in {path}")
            ch_names = [f"ch{i+1}" for i in range(data.shape[0])]
            events = None
        else:
            EEG = mat["EEG"]
            data = np.asarray(EEG.data)                    # (n_channels, n_samples)
            sfreq = float(np.array(EEG.srate).squeeze())
            ch_names = _chan_names_from_chanlocs(getattr(EEG, "chanlocs", None), data.shape[0])
            events = _events_from_eeglab_struct(EEG)

        # drop bad channels by name if present
        if any(b in ch_names for b in BAD_CHS):
            keep_idx = [i for i, nm in enumerate(ch_names) if nm not in BAD_CHS]
            data = data[keep_idx, :]
            ch_names = [ch_names[i] for i in keep_idx]

        return data, (float(sfreq) if np.isfinite(sfreq) else np.nan), events, ch_names

    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def raw_from_array(data: np.ndarray, sfreq: float, ch_names: Optional[List[str]] = None):
    """Create an MNE Raw from (channels, samples) with real names if provided."""
    if ch_names is None or len(ch_names) != data.shape[0]:
        ch_names = [f"ch{i+1}" for i in range(int(data.shape[0]))]
    info = mne.create_info(ch_names=ch_names, sfreq=float(sfreq), ch_types="eeg")
    return mne.io.RawArray(data, info, verbose="ERROR")


def split_open_close(data: np.ndarray, events: Optional[np.ndarray]):
    """
    Split continuous EEG (channels, samples) into eyes-open and eyes-closed segments
    using events of shape (n_events, 3): [onset_sample, 0, code].
    Returns (open_data, close_data), each as (channels, samples) or None if not present.
    If events is None: returns (data, None) i.e., treat entire recording as eyes-open.
    """
    if events is None or len(events) == 0:
        return data, None

    open_chunks = []
    close_chunks = []

    ev = np.array(sorted(events, key=lambda x: x[0]))
    onsets = ev[:, 0].astype(int)
    codes = ev[:, 2].astype(int)

    for i in range(len(onsets)):
        tmin = max(onsets[i] - 1, 0)
        tmax = data.shape[1] if i == len(onsets) - 1 else max(onsets[i + 1] - 1, tmin + 1)
        seg = data[:, tmin:tmax]
        code = codes[i]
        if code in OPEN_CODES:
            open_chunks.append(seg)
        elif code in CLOSE_CODES:
            close_chunks.append(seg)

    open_data = np.concatenate(open_chunks, axis=1) if open_chunks else None
    close_data = np.concatenate(close_chunks, axis=1) if close_chunks else None
    return open_data, close_data


# ------------------------------
# Signal processing
# ------------------------------
def process_signal(raw: mne.io.BaseRaw):
    """Basic preprocessing: referencing, channel drop, baseline, notch, band, butter, ICA."""
    # Reference
    try:
        raw.set_eeg_reference(ref_channels=['M1', 'M2'])
    except Exception:
        raw.set_eeg_reference('average')

    # Drop bad channels if present (kept for safety, should already be dropped)
    dropable = [x for x in BAD_CHS if x in raw.ch_names]
    if dropable:
        raw.drop_channels(dropable)

    # Baseline (rescale whole record)
    raw._data = mne.baseline.rescale(raw.get_data(), raw.times, (None, None))

    # Notch & band filters
    try:
        raw.notch_filter(50.0, verbose='WARNING')
    except Exception:
        pass
    try:
        raw.filter(0.2, 45.0, method='iir', verbose='WARNING')
    except Exception:
        pass

    # Extra Butterworth band-pass
    try:
        lowcut = 1.0
        highcut = 50.0
        order = 5
        nyq = 0.5 * raw.info['sfreq']
        b, a = scipy.signal.butter(order, [lowcut / nyq, highcut / nyq], btype='band')
        raw._data = scipy.signal.lfilter(b, a, raw.get_data())
    except Exception:
        pass

    # ICA
    try:
        ica = mne.preprocessing.ICA(max_iter=100, random_state=1)
        ica.fit(raw)
        ica.apply(raw)
    except Exception:
        pass

    return raw


# ------------------------------
# Per-subject processing & save
# ------------------------------
def process_subjects():
    for matNum in matNums:
        base_name = name_tpl.format(matNum)
        raw_file = join(RAW_DIR, base_name + EXT)   # e.g., ./dataset/507_Depression_REST.mat
        out_h5   = join(SUBJ_H5_DIR, base_name + '.h5')

        if not os.path.isfile(raw_file):
            print(f"[WARN] missing file: {raw_file}")
            continue

        # Load continuous recording + events; drop BAD_CHS inside loader
        data_i, sfreq_i, events_i, ch_names_i = load_eeg_file(raw_file)
        if not (np.isfinite(sfreq_i) and sfreq_i > 0):
            sfreq_i = TARGET_FS_DEFAULT  # default if unknown

        # Split to eyes-open / eyes-closed
        open_data, close_data = split_open_close(data_i, events_i)

        # Process and save to subject-level H5
        with h5py.File(out_h5, 'w') as h5f:
            if open_data is not None and open_data.size > 0:
                raw_open = raw_from_array(open_data, sfreq=sfreq_i, ch_names=ch_names_i)
                raw_open = process_signal(raw_open)
                h5f.create_dataset('open', data=raw_open.get_data())

            if close_data is not None and close_data.size > 0:
                raw_close = raw_from_array(close_data, sfreq=sfreq_i, ch_names=ch_names_i)
                raw_close = process_signal(raw_close)
                h5f.create_dataset('close', data=raw_close.get_data())

        print(f"[OK] saved: {out_h5}")

    gc.collect()


# ------------------------------
# Helper functions for sequences
# ------------------------------
def pad_N_epochs(x, limit):
    out = []
    for y in x:
        while len(y) < limit:
            y = np.concatenate([y, y[-(limit - len(y)):]])
        out.append(y[:limit])
    return np.array(out)


def crop_N_epochs(x, limit):
    return np.array([y[:limit] for y in pad_N_epochs(x, limit)])


def func(temp):
    if 0 <= temp <= 13:
        return 0
    elif 14 <= temp <= 19:
        return 1
    elif 20 <= temp <= 28:
        return 2
    else:
        return 3


# ------------------------------
# Build combined './open data/data.h5'
# ------------------------------
def build_combined_open_h5():
    base_dir = SUBJ_H5_DIR
    onlyfiles = [f for f in listdir(base_dir) if isfile(join(base_dir, f)) and f.endswith('.h5')]
    label_path = LABELS_XLSX

    xl_file = pd.ExcelFile(label_path)
    dfs = {sheet_name: xl_file.parse(sheet_name) for sheet_name in xl_file.sheet_names}
    dfs = dfs['Depression Rest']  # expects columns: id, BDI

    seq_len = 60000
    idx = np.arange(0, seq_len, 2)   # decimate by 2
    Xs = []                          # collect (1, T, 62)
    patients_id = []
    labels = []
    shappe = []
    event_type = 'open'
    TARGET_CH = 62

    for i in onlyfiles:
        with h5py.File(join(base_dir, i), 'r') as h5f:
            if event_type not in h5f:
                print(f"[WARN] '{event_type}' not in {i}, skipping.")
                continue
            data = h5f[event_type][...]  # (channels=62, samples)

        # keep stats for plot
        shappe.append(data.shape[1])

        # ensure seq_len (crop/pad), then downsample by 2, then (1, T, ch)
        temp = crop_N_epochs(data, seq_len)        # (62, seq_len)
        temp = temp[:, idx]                        # (62, seq_len/2)
        temp = temp.reshape(1, temp.shape[0], temp.shape[1])  # (1, ch, T)
        temp_T = temp.transpose([0, 2, 1])         # (1, T, ch=62)

        # sanity enforce 62 channels if needed
        ch = temp_T.shape[2]
        if ch != TARGET_CH:
            if ch > TARGET_CH:
                temp_T = temp_T[:, :, :TARGET_CH]
                print(f"[INFO] {i}: truncated channels {ch} -> {TARGET_CH}")
            else:
                pad = np.zeros((temp_T.shape[0], temp_T.shape[1], TARGET_CH - ch), dtype=temp_T.dtype)
                temp_T = np.concatenate([temp_T, pad], axis=2)
                print(f"[INFO] {i}: padded channels {ch} -> {TARGET_CH}")

        Xs.append(temp_T)

        temp_id = int(i.split('_')[0])
        patients_id.append(temp_id)
        temp_label = dfs[dfs['id'] == temp_id].BDI.values[0]
        labels.append(temp_label)
        print("[OK] aggregated:", i)

    if not Xs:
        raise RuntimeError("No subject files aggregated. Check './datasets/' content and event_type presence.")

    X = np.concatenate(Xs, axis=0)  # (N, T, 62)
    patients_id = np.array(patients_id).astype('int16')
    labels = np.array(labels).astype('uint8')

    with h5py.File(OPEN_COMBINED_H5, 'w') as h5f:
        h5f.create_dataset('X', data=X)
        h5f.create_dataset('patients_id', data=patients_id)
        h5f.create_dataset('labels', data=labels)

    gc.collect()

    plt.scatter(range(len(shappe)), shappe, s=5, c=np.array(labels))
    plt.title("Per-subject aggregated length (samples)")
    plt.show()

    return X.shape[1]  # timestep


# ------------------------------
# Label distribution (report)
# ------------------------------
def report_label_distribution():
    xl_file = pd.ExcelFile(LABELS_XLSX)
    dfs = {sheet_name: xl_file.parse(sheet_name) for sheet_name in xl_file.sheet_names}
    labels_all = dfs['Depression Rest']['BDI'].values.astype('uint8')
    normal = np.sum((labels_all >= 0) & (labels_all <= 13))
    mild = np.sum((labels_all >= 14) & (labels_all <= 19))
    moderate = np.sum((labels_all >= 20) & (labels_all <= 28))
    severe = np.sum(labels_all >= 29)
    print('total patients:', normal + mild + moderate + severe)
    print('number of normal:', normal)
    print('number of mild:', mild)
    print('number of moderate:', moderate)
    print('number of severe:', severe)


# ------------------------------
# Windowing for modeling
# ------------------------------
def window_open_data():
    fs = 250
    length = 5   # seconds
    olp = 0.9    # overlap percentage

    with h5py.File(OPEN_COMBINED_H5, 'r') as h5f:
        X = h5f['X'][...]
        patients_id = h5f['patients_id'][...]
        labels = h5f['labels'][...]

    win_size = length * fs
    bias = int((1.0 - olp) * win_size)
    timestep = X.shape[1]

    # The original notebook used fixed counts per label bin:
    num_row = 1160 + 1152 + 1162 + 1140
    windowed_x = np.zeros((num_row, win_size, 62))  # 62 channels
    new_labels = np.zeros((num_row, 1))
    patients_ids = np.zeros((num_row, 1))
    start = 0

    for i in range(X.shape[0]):
        temp_label = func(int(labels[i]))
        if temp_label == 0:
            num_samp = 15
        elif temp_label == 1:
            num_samp = 83
        elif temp_label == 2:
            num_samp = 48
        else:
            num_samp = 232
        seg_data = np.zeros((num_samp, win_size, 62))
        start2 = 0
        for j in range(num_samp):
            seg_data[j, :, :] = X[i, start2:(start2 + win_size), :].reshape(1, win_size, 62)
            start2 += bias

        windowed_x[start:start + num_samp, :, :] = seg_data
        new_labels[start:start + num_samp] = labels[i] * np.ones((num_samp, 1))
        patients_ids[start:start + num_samp] = patients_id[i] * np.ones((num_samp, 1))
        start += num_samp
        gc.collect()

    new_labels = new_labels.astype('uint8')
    patients_ids = patients_ids.astype('int16')

    del X, labels, patients_id
    gc.collect()

    print("Windowed shapes:", windowed_x.shape, new_labels.shape, patients_ids.shape)
    return windowed_x, new_labels, patients_ids


# ------------------------------
# Label helpers / dataset loaders
# ------------------------------
def change_to_4(all_label):
    L = np.copy(all_label)
    a = np.unique(all_label)
    for i in a:
        idx = np.where(all_label == i)[0]
        if 0 <= i <= 13:
            L[idx] = 0
        elif 14 <= i <= 19:
            L[idx] = 1
        elif 20 <= i <= 28:
            L[idx] = 2
        else:
            L[idx] = 3
    return L


def change_to_2(data, label, mode):
    idx = np.zeros(1).astype(int)
    for m in mode:
        idx = np.append(idx, np.where(label == m)[0], axis=0)
    idx = np.delete(idx, 0, 0)
    idx = np.sort(idx)
    return data[idx], label[idx]


def prepare_data(data, labels, mode):
    if len(mode) == 1:  # 4-class classification or regression
        if mode[0] == 0:  # regression
            print('problem: regression')
            return data, labels
        else:             # 4-class classification
            print('problem: 4 class')
            labels = change_to_4(labels)
            return data, labels
    else:  # 2-class classification
        print('problem: 2 class')
        labels = change_to_4(labels)
        data, labels = change_to_2(data, labels, mode)
        return data, labels


def load_data2():
    return windowed_x, new_labels, patients_ids


# ------------------------------
# Modeling
# ------------------------------
def proposed_model(num_feats, timesteps, num_class):
    krl_init = glorot_normal(seed=42)
    i = Input(shape=(timesteps, num_feats), name='inp')
    h1 = TCN(return_sequences=True, nb_filters=64, kernel_size=5, kernel_initializer=krl_init,
             nb_stacks=1, padding='causal', dilations=[1, 2, 4, 8, 16, 32], activation='relu', name='tcn1')(i)
    h1 = Flatten(name='flt')(h1)
    h2 = LSTM(100, activation='relu', return_sequences=False, name='lstm')(i)
    h = concatenate([h1, h2])
    h = Dense(64, activation='relu', name='fc1')(h)
    if num_class == 1:
        h = Dense(1, name='fc2')(h)
    else:
        h = Dense(num_class, activation='softmax', name='fc3')(h)

    model = Model(inputs=i, outputs=h)
    adamopt = optimizers.adam(lr=0.0001)
    if num_class == 1:
        model.compile(optimizer=adamopt, loss='mse')
    else:
        model.compile(optimizer=adamopt, loss='categorical_crossentropy',
                      metrics=[metrics.Precision(name='precision_1'), metrics.Recall(name='recall_1'), 'acc'])
    return model


def DCNN(input_dim, timesteps, num_class):
    krl_init = glorot_normal(seed=42)
    i = Input(shape=(timesteps, input_dim, 1), name='inp')
    h = Conv2D(filters=64, kernel_size=5, strides=(3, 3), activation='relu',
               name='conv2_1', kernel_initializer=krl_init, data_format='channels_last')(i)
    h = Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu',
               name='conv2_2', kernel_initializer=krl_init, data_format='channels_last')(h)
    s = K.int_shape(h)
    h = Reshape((s[1], s[2] * s[3]))(h)
    h = TCN(return_sequences=True, nb_filters=50, kernel_size=3, kernel_initializer=krl_init,
            nb_stacks=1, padding='causal', dilations=[1, 2, 4], activation='relu', name='tcn1')(h)
    h = Flatten(name='flt')(h)
    h = Dropout(0.5, name='drp1')(h)
    h = Dense(512, activation='relu', name='fc1')(h)
    if num_class == 1:
        h = Dense(1, name='fc2')(h)
    else:
        h = Dense(num_class, activation='softmax', name='fc3')(h)

    model = Model(inputs=i, outputs=h)
    adamopt = optimizers.Adam(learning_rate=0.0001)
    if num_class == 1:
        model.compile(optimizer=adamopt, loss='mse')
    else:
        model.compile(optimizer=adamopt, loss='categorical_crossentropy',
                      metrics=['acc', metrics.Precision(name='precision_1'), metrics.Recall(name='recall_1')])
    return model


# ------------------------------
# Main execution
# ------------------------------
if __name__ == "__main__":
    # 1) Per-subject processing (saves ./datasets/<ID>_Depression_REST.h5)
    process_subjects()

    # 2) Combine open segments into ./open data/data.h5
    timestep = build_combined_open_h5()

    # 3) Report label distribution
    report_label_distribution()

    # 4) Window into fixed-length samples for modeling
    windowed_x, new_labels, patients_ids = window_open_data()

    # 5) Train / eval (10-fold)
    mode = [4]  # [0]=regression, [4]=4-class, or [i,j] for 2-class

    print('Loading data...')
    data, labels, patients = load_data2()
    gc.collect()

    print('preparing data...')
    data, labels = prepare_data(data, labels, mode)
    gc.collect()

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    sc = StandardScaler()
    # handle sklearn version differences for OneHotEncoder arg name
    try:
        encoder = OneHotEncoder(sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(sparse=False)

    counter = 0

    train_result, test_result = [], []
    train_mse, test_mse = [], []
    train_rmse, test_rmse = [], []
    train_mae, test_mae = [], []
    train_r2,  test_r2  = [], []
    train_conf_mat = np.zeros((2, 2))
    test_conf_mat  = np.zeros((2, 2))
    train_sensitivity, train_specificity = [], []
    test_sensitivity,  test_specificity  = [], []
    reg_pred = np.zeros((data.shape[0], 1))

    print('starting 10 fold...')
    for train_index, test_index in skf.split(data, labels):
        counter += 1
        print('Fold:', counter)
        train_x, test_x = data[train_index], data[test_index]
        train_y, test_y = labels[train_index], labels[test_index]
        print('splitting passed')

        # normalization per-sample
        for i in range(train_x.shape[0]):
            sc.partial_fit(train_x[i])
        for i in range(train_x.shape[0]):
            train_x[i] = sc.transform(train_x[i])
        for i in range(test_x.shape[0]):
            test_x[i] = sc.transform(test_x[i])
        print('normalization passed')

        if len(mode) == 1 and mode[0] == 0:
            num_class = 1
        else:
            train_y = encoder.fit_transform(train_y.reshape(-1, 1))
            test_y = encoder.transform(test_y.reshape(-1, 1))
            num_class = train_y.shape[1]
            print('num class:', num_class)

        (_, timesteps, num_feats) = train_x.shape
        print('timesteps:', timesteps, 'num_feats:', num_feats)
        train_x = train_x.reshape(train_x.shape[0], timesteps, num_feats, 1)
        test_x  = test_x.reshape(test_x.shape[0], timesteps, num_feats, 1)

        clear_session()
        model = DCNN(num_feats, timesteps, num_class)

        if len(mode) == 1 and mode[0] == 0:  # regression
            print('regression')
            history = model.fit(train_x, train_y, batch_size=10, epochs=50,
                                validation_data=(test_x, test_y), verbose=1)
        else:
            print('classification')
            history = model.fit(train_x, train_y, batch_size=16, epochs=50,
                                validation_data=(test_x, test_y), verbose=0)

        # (Optional) quick plots guarded — remove if running headless
        try:
            if 'acc' in history.history and 'val_acc' in history.history:
                plt.plot(history.history['acc']); plt.plot(history.history['val_acc'])
                plt.title('model accuracy'); plt.ylabel('accuracy'); plt.xlabel('epoch')
                plt.legend(['train', 'val']); plt.show()
            if 'recall_1' in history.history and 'val_recall_1' in history.history:
                plt.plot(history.history['recall_1']); plt.plot(history.history['val_recall_1'])
                plt.title('model recall'); plt.ylabel('recall'); plt.xlabel('epoch')
                plt.legend(['train', 'val']); plt.show()
            if 'precision_1' in history.history and 'val_precision_1' in history.history:
                plt.plot(history.history['precision_1']); plt.plot(history.history['val_precision_1'])
                plt.title('model precision'); plt.ylabel('precision'); plt.xlabel('epoch')
                plt.legend(['train', 'val']); plt.show()
            plt.plot(history.history['loss']); plt.plot(history.history['val_loss'])
            plt.title('model loss'); plt.ylabel('loss'); plt.xlabel('epoch')
            plt.legend(['train', 'val']); plt.show()
        except Exception:
            pass

        train_pred = model.predict(train_x)
        test_pred  = model.predict(test_x)

        if len(mode) == 1 and mode[0] == 0:  # regression
            train_mse.append(mean_squared_error(train_y, train_pred))
            test_mse.append(mean_squared_error(test_y, test_pred))
            train_rmse.append(np.sqrt(mean_squared_error(train_y, train_pred)))
            test_rmse.append(np.sqrt(mean_squared_error(test_y, test_pred)))
            train_mae.append(mean_absolute_error(train_y, train_pred))
            test_mae.append(mean_absolute_error(test_y, test_pred))
            train_r2.append(r2_score(train_y, train_pred))
            test_r2.append(r2_score(test_y, test_pred))
            reg_pred[test_index] = test_pred.reshape(-1, 1)
        else:
            train_pred = np.argmax(train_pred, axis=1)
            train_y    = np.argmax(train_y, axis=1)
            train_result.append(np.mean(train_pred == train_y))

            test_pred = np.argmax(test_pred, axis=1)
            test_y    = np.argmax(test_y, axis=1)
            test_result.append(np.mean(test_pred == test_y))

            if len(mode) == 2:  # binary
                temp1, temp2, _ = sensitivity_specificity_support(train_y, train_pred, average='binary')
                train_sensitivity.append(temp1); train_specificity.append(temp2)
                temp1, temp2, _ = sensitivity_specificity_support(test_y, test_pred, average='binary')
                test_sensitivity.append(temp1); test_specificity.append(temp2)
                train_conf_mat += confusion_matrix(train_y, train_pred)
                test_conf_mat  += confusion_matrix(test_y, test_pred)

        del test_x, test_y, train_x, train_y
        gc.collect()

    # ------------------------------
    # Aggregate metrics & save plots
    # ------------------------------
    plt.style.use('seaborn-whitegrid')

    if len(mode) == 1 and mode[0] == 0:
        # regression summaries
        def save_vec(vec, name):
            v = np.array(vec).reshape(-1, 1)
            plt.figure(figsize=(10, 8))
            plt.plot(range(1, v.shape[0] + 1), v, '.-', linewidth=2)
            plt.xticks(range(1, v.shape[0] + 1)); plt.xlabel('Fold number'); plt.ylabel(name)
            plt.savefig(f'./model results/{name}.jpg'); plt.close()
            df = pd.DataFrame(np.concatenate((v, np.mean(v).reshape(-1, 1),
                                              np.std(v).reshape(-1, 1)), axis=0), columns=[name])
            df.to_csv(f'./model results/csv results/{name}.csv', index=False)

        save_vec(train_mse, 'train_MSE');  save_vec(test_mse,  'test_MSE')
        save_vec(train_rmse,'train_RMSE'); save_vec(test_rmse, 'test_RMSE')
        save_vec(train_mae, 'train_MAE');  save_vec(test_mae,  'test_MAE')
        save_vec(train_r2,  'train_R2');   save_vec(test_r2,   'test_R2')

        # true vs pred (last fold)
        df = pd.DataFrame(np.concatenate([labels.reshape(-1, 1), reg_pred], axis=1),
                          columns=['True value', 'Predict value'])
        df.to_csv('./model results/csv results/true_vs_pred.csv', index=False)

    else:
        # classification accuracy per fold
        tr = np.array(train_result).reshape(-1, 1)
        te = np.array(test_result).reshape(-1, 1)

        def save_acc(vec, split):
            plt.figure(figsize=(10, 8))
            plt.plot(range(1, vec.shape[0] + 1), vec, '.-', linewidth=2)
            plt.xticks(range(1, vec.shape[0] + 1))
            plt.xlabel('Fold number'); plt.ylabel('Accuracy')
            if len(mode) == 2:
                plt.savefig(f'./model results/{split}_2class({mode[0]},{mode[1]})_accuracy.jpg')
                df = pd.DataFrame(np.concatenate((vec, np.mean(vec).reshape(-1, 1),
                                                  np.std(vec).reshape(-1, 1)), axis=0), columns=['accuracy'])
                df.to_csv(f'./model results/csv results/{split}_2class({mode[0]},{mode[1]})_accuracy.csv', index=False)
            else:
                plt.savefig(f'./model results/{split}_4class.jpg')
                df = pd.DataFrame(np.concatenate((vec, np.mean(vec).reshape(-1, 1),
                                                  np.std(vec).reshape(-1, 1)), axis=0), columns=['accuracy'])
                df.to_csv(f'./model results/csv results/{split}_4class.csv', index=False)
            plt.close()

        save_acc(tr, 'train'); save_acc(te, 'test')

        if len(mode) == 2:
            train_conf_mat /= 10.0
            test_conf_mat  /= 10.0
            print('train confusion matrix:\n', train_conf_mat)
            print('test confusion matrix:\n', test_conf_mat)
            pd.DataFrame(train_conf_mat).to_csv(f'./model results/csv results/train_2class({mode[0]},{mode[1]})_CM.csv', index=False)
            pd.DataFrame(test_conf_mat ).to_csv(f'./model results/csv results/test_2class({mode[0]},{mode[1]})_CM.csv',  index=False)

            # sensitivity / specificity plots
            tr_sen = np.array(train_sensitivity).reshape(-1, 1)
            tr_spe = np.array(train_specificity).reshape(-1, 1)
            te_sen = np.array(test_sensitivity).reshape(-1, 1)
            te_spe = np.array(test_specificity).reshape(-1, 1)

            plt.figure(figsize=(10, 8))
            plt.plot(range(1, tr.shape[0] + 1), tr, '.-', label='Accuracy', linewidth=2)
            plt.plot(range(1, tr_sen.shape[0] + 1), tr_sen, '.-', label='sensitivity', linewidth=2)
            plt.plot(range(1, tr_spe.shape[0] + 1), tr_spe, '.-', label='specificity', linewidth=2)
            plt.xlabel('Fold number'); plt.ylabel('Performance'); plt.legend()
            plt.savefig(f'./model results/train_2class({mode[0]},{mode[1]})_acc_sen_spe.jpg'); plt.close()

            plt.figure(figsize=(10, 8))
            plt.plot(range(1, tr_sen.shape[0] + 1), tr_sen, '.-', label='sensitivity', linewidth=2)
            plt.plot(range(1, tr_spe.shape[0] + 1), tr_spe, '.-', label='specificity', linewidth=2)
            plt.xlabel('Fold number'); plt.ylabel('Performance'); plt.legend()
            plt.savefig(f'./model results/train_2class({mode[0]},{mode[1]})_sen_spe.jpg'); plt.close()

            pd.DataFrame(np.concatenate((tr_sen, np.mean(tr_sen).reshape(-1,1), np.std(tr_sen).reshape(-1,1)), axis=0),
                         columns=['sensitivity']).to_csv(f'./model results/csv results/train_2class({mode[0]},{mode[1]})_sensitivity.csv', index=False)
            pd.DataFrame(np.concatenate((tr_spe, np.mean(tr_spe).reshape(-1,1), np.std(tr_spe).reshape(-1,1)), axis=0),
                         columns=['specificity']).to_csv(f'./model results/csv results/train_2class({mode[0]},{mode[1]})_specificity.csv', index=False)

            plt.figure(figsize=(10, 8))
            plt.plot(range(1, te.shape[0] + 1), te, '.-', label='Accuracy', linewidth=2)
            plt.plot(range(1, te_sen.shape[0] + 1), te_sen, '.-', label='sensitivity', linewidth=2)
            plt.plot(range(1, te_spe.shape[0] + 1), te_spe, '.-', label='specificity', linewidth=2)
            plt.xlabel('Fold number'); plt.ylabel('Performance'); plt.legend()
            plt.savefig(f'./model results/test_2class({mode[0]},{mode[1]})_acc_sen_spe.jpg'); plt.close()

            plt.figure(figsize=(10, 8))
            plt.plot(range(1, te_sen.shape[0] + 1), te_sen, '.-', label='sensitivity', linewidth=2)
            plt.plot(range(1, te_spe.shape[0] + 1), te_spe, '.-', label='specificity', linewidth=2)
            plt.xlabel('Fold number'); plt.ylabel('Performance'); plt.legend()
            plt.savefig(f'./model results/test_2class({mode[0]},{mode[1]})_sen_spe.jpg'); plt.close()

            pd.DataFrame(np.concatenate((te_sen, np.mean(te_sen).reshape(-1,1), np.std(te_sen).reshape(-1,1)), axis=0),
                         columns=['sensitivity']).to_csv(f'./model results/csv results/test_2class({mode[0]},{mode[1]})_sensitivity.csv', index=False)
            pd.DataFrame(np.concatenate((te_spe, np.mean(te_spe).reshape(-1,1), np.std(te_spe).reshape(-1,1)), axis=0),
                         columns=['specificity']).to_csv(f'./model results/csv results/test_2class({mode[0]},{mode[1]})_specificity.csv', index=False)

    clear_session()
    gc.collect()
    # model.save_weights('./weights_last_fold.h5')  # optional
