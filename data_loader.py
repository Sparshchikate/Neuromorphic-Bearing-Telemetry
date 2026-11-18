import os
import glob
import scipy.io as sio
import numpy as np
from delta_modulator import causal_delta_modulate, static_delta_modulate

# Standard 12 kHz Drive-End CWRU mapping: (class_id, load_hp) -> file_id
CWRU_FILE_MAP = {
    (0, 0): "97",  (0, 1): "98",  (0, 2): "99",  (0, 3): "100",  # Normal
    (1, 0): "105", (1, 1): "106", (1, 2): "107", (1, 3): "108",  # IR 0.007"
    (2, 0): "118", (2, 1): "119", (2, 2): "120", (2, 3): "121",  # Ball 0.007"
    (3, 0): "130", (3, 1): "131", (3, 2): "132", (3, 3): "133",  # OR@6 0.007"
    (4, 0): "169", (4, 1): "170", (4, 2): "171", (4, 3): "172",  # IR 0.014"
    (5, 0): "185", (5, 1): "186", (5, 2): "187", (5, 3): "188",  # Ball 0.014"
    (6, 0): "197", (6, 1): "198", (6, 2): "199", (6, 3): "200",  # OR@6 0.014"
    (7, 0): "209", (7, 1): "210", (7, 2): "211", (7, 3): "212",  # IR 0.021"
    (8, 0): "222", (8, 1): "223", (8, 2): "224", (8, 3): "225",  # Ball 0.021"
    (9, 0): "234", (9, 1): "235", (9, 2): "236", (9, 3): "237",  # OR@6 0.021"
}

CLASS_NAMES = [
    "Normal",
    "IR_007", "Ball_007", "OR_007",
    "IR_014", "Ball_014", "OR_014",
    "IR_021", "Ball_021", "OR_021",
]


def load_de_signal(mat_path):
    mat = sio.loadmat(mat_path)
    for k in mat:
        if k.endswith("DE_time"):
            return mat[k].ravel().astype(np.float64)
    raise KeyError(f"DE_time key not found in {mat_path}")


def add_awgn(signal, snr_db, rng):
    sig_power = np.mean(signal ** 2)
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=signal.shape)
    return signal + noise


def build_recording_records(data_dir, window_size=1024, stride=1024, mode="adaptive", snr_db=None, seed=42):
    rng = np.random.default_rng(seed)
    records = []

    for (cls_id, load_hp), file_id in sorted(CWRU_FILE_MAP.items()):
        matches = glob.glob(os.path.join(data_dir, f"*{file_id}*.mat"))
        if not matches:
            continue

        raw = load_de_signal(matches[0])
        if snr_db is not None:
            raw = add_awgn(raw, snr_db, rng)

        if mode == "adaptive":
            dual_spikes, _, _, _ = causal_delta_modulate(raw, window_size=window_size)
        elif mode == "static":
            dual_spikes, _ = static_delta_modulate(raw)
        else:
            dual_spikes = raw[:, None].astype(np.float32)

        windows = []
        for start in range(0, len(dual_spikes) - window_size + 1, stride):
            windows.append(dual_spikes[start : start + window_size])

        if windows:
            records.append({
                "file_id": file_id,
                "class_id": cls_id,
                "load_hp": load_hp,
                "X": np.stack(windows, axis=0),
                "y": np.full(len(windows), cls_id, dtype=np.int64),
                "loads": np.full(len(windows), load_hp, dtype=np.int64),
            })

    if not records:
        raise FileNotFoundError(f"No CWRU .mat files found in '{data_dir}'.")
    return records


def get_file_level_folds(records, n_folds=5, seed=42):
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(n_folds)]

    # Stratify assignment by (class_id, load_hp) across folds at the file/segment level
    by_stratum = {}
    for idx, rec in enumerate(records):
        key = (rec["class_id"], rec["load_hp"])
        by_stratum.setdefault(key, []).append(idx)

    for key, idxs in by_stratum.items():
        if len(idxs) >= n_folds:
            perm = rng.permutation(idxs)
            for i, rec_idx in enumerate(perm):
                folds[i % n_folds].append(records[rec_idx])
        else:
            # Partition continuous blocks sequentially into n_folds when 1 file exists per (class, load)
            for rec_idx in idxs:
                rec = records[rec_idx]
                n_win = len(rec["X"])
                chunk = n_win // n_folds
                for f in range(n_folds):
                    s = f * chunk
                    e = (f + 1) * chunk if f < n_folds - 1 else n_win
                    if e > s:
                        folds[f].append({
                            "file_id": rec["file_id"],
                            "class_id": rec["class_id"],
                            "load_hp": rec["load_hp"],
                            "X": rec["X"][s:e],
                            "y": rec["y"][s:e],
                            "loads": rec["loads"][s:e],
                        })

    split_data = []
    for f in range(n_folds):
        test_recs = folds[f]
        train_recs = [r for i in range(n_folds) if i != f for r in folds[i]]

        X_train = np.concatenate([r["X"] for r in train_recs], axis=0)
        y_train = np.concatenate([r["y"] for r in train_recs], axis=0)
        X_test = np.concatenate([r["X"] for r in test_recs], axis=0)
        y_test = np.concatenate([r["y"] for r in test_recs], axis=0)
        loads_test = np.concatenate([r["loads"] for r in test_recs], axis=0)

        split_data.append((X_train, y_train, X_test, y_test, loads_test))

    return split_data