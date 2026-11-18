import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from data_loader import build_recording_records, get_file_level_folds
from rlif_model import NeuromorphicRLIFClassifier


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_sparsity_and_energy(in_spikes, s1, s2):
    # in_spikes: (B, T, 2), s1: (B, T, 128), s2: (B, T, 64)
    batch_size, time_steps, _ = in_spikes.shape
    n_samples = batch_size * time_steps

    # Neuronal activation sparsity across 192 hidden neurons (Section IV-C1)
    total_neuron_steps = batch_size * time_steps * (128 + 64)
    hidden_spikes = s1.sum().item() + s2.sum().item()
    firing_rate = hidden_spikes / total_neuron_steps
    neuronal_sparsity = (1.0 - firing_rate) * 100.0

    # Input event rate
    input_events = in_spikes.sum().item()
    input_event_rate = input_events / (batch_size * time_steps)

    # Synaptic fan-out SOPs across 29,568 connections (Section IV-C2)
    # 2x128 + 128x128 + 128x64 + 64x64 + 64x10 = 29,568
    dense_ops_per_window = 29568 * time_steps
    active_sops_total = (
        in_spikes.sum().item() * 128
        + s1.sum().item() * (128 + 64)
        + s2.sum().item() * (64 + 10)
    )
    avg_sops_per_window = active_sops_total / batch_size
    synaptic_sparsity = (1.0 - (avg_sops_per_window / dense_ops_per_window)) * 100.0

    # 45 nm CMOS energy accounting (Eq. 13 & 14)
    e_syn_uj = (avg_sops_per_window * 0.9e-12) * 1e6
    e_frontend_uj = ((138.6e-12 * time_steps) + (4 * time_steps * 0.1e-12)) * 1e6
    e_total_uj = e_syn_uj + e_frontend_uj

    # IEEE 802.15.4 analytical battery model (Eq. 15-17)
    d_fault = (180.0 * 3.278e-3) / 3600.0
    d_heartbeat = 0.015 / 3600.0
    duty_cycle = d_fault + d_heartbeat
    i_avg_ma = (17.4 * duty_cycle) + 0.0012 + 0.0145 + 0.0040 + 0.003176
    effective_mah = 1000.0 * 0.77
    battery_years = effective_mah / (i_avg_ma * 8766.0)

    return {
        "input_event_rate_pct": input_event_rate * 100.0,
        "neuronal_sparsity_pct": neuronal_sparsity,
        "synaptic_sparsity_pct": synaptic_sparsity,
        "avg_sops_per_window": avg_sops_per_window,
        "e_syn_uj": e_syn_uj,
        "e_frontend_uj": e_frontend_uj,
        "e_total_uj": e_total_uj,
        "battery_years": battery_years,
    }


def train_fold(X_train, y_train, X_test, y_test, loads_test, device, epochs=50, batch_size=32, lr=1e-3, recurrent=True):
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test), torch.from_numpy(loads_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = NeuromorphicRLIFClassifier(
        in_channels=2, h1=128, h2=64, num_classes=10,
        beta=0.85, u_th=1.00, gamma=2.0, recurrent=recurrent
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

    model.eval()
    all_preds, all_targets, all_loads = [], [], []
    in_list, s1_list, s2_list = [], [], []

    with torch.no_grad():
        for bx, by, bl in test_loader:
            bx = bx.to(device)
            logits, stats = model(bx, return_stats=True)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.append(preds)
            all_targets.append(by.numpy())
            all_loads.append(bl.numpy())

            in_list.append(stats["in_spikes"].cpu())
            s1_list.append(stats["s1"].cpu())
            s2_list.append(stats["s2"].cpu())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    loads = np.concatenate(all_loads)

    overall_acc = np.mean(preds == targets) * 100.0
    load_accs = {}
    for hp in [0, 1, 2, 3]:
        mask = loads == hp
        if np.any(mask):
            load_accs[hp] = np.mean(preds[mask] == targets[mask]) * 100.0

    metrics = compute_sparsity_and_energy(
        torch.cat(in_list, dim=0),
        torch.cat(s1_list, dim=0),
        torch.cat(s2_list, dim=0),
    )
    metrics["overall_acc"] = overall_acc
    metrics["load_accs"] = load_accs
    return model, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/cwru")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mode", type=str, default="adaptive", choices=["adaptive", "static"])
    parser.add_argument("--feedforward_only", action="store_true")
    parser.add_argument("--snr_db", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    records = build_recording_records(
        args.data_dir, window_size=1024, stride=1024,
        mode=args.mode, snr_db=args.snr_db, seed=args.seed
    )
    folds = get_file_level_folds(records, n_folds=5, seed=args.seed)

    fold_metrics = []
    for f_idx, (X_tr, y_tr, X_te, y_te, l_te) in enumerate(folds):
        model, m = train_fold(
            X_tr, y_tr, X_te, y_te, l_te, device=device,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            recurrent=(not args.feedforward_only),
        )
        fold_metrics.append(m)
        print(
            f"Fold {f_idx + 1}/5 | Acc: {m['overall_acc']:.2f}% | "
            f"Synaptic Sparsity: {m['synaptic_sparsity_pct']:.2f}% | "
            f"Energy: {m['e_total_uj']:.3f} uJ"
        )
        if f_idx == 0:
            torch.save(model.state_dict(), os.path.join(args.out_dir, "rlif_fold1.pt"))

    accs = [m["overall_acc"] for m in fold_metrics]
    syn_sp = [m["synaptic_sparsity_pct"] for m in fold_metrics]
    neu_sp = [m["neuronal_sparsity_pct"] for m in fold_metrics]
    energies = [m["e_total_uj"] for m in fold_metrics]

    summary = {
        "macro_acc_mean": float(np.mean(accs)),
        "macro_acc_sd": float(np.std(accs)),
        "load_acc_mean": {
            f"{hp}HP": float(np.mean([m["load_accs"][hp] for m in fold_metrics if hp in m["load_accs"]]))
            for hp in [0, 1, 2, 3]
        },
        "neuronal_sparsity_mean": float(np.mean(neu_sp)),
        "synaptic_sparsity_mean": float(np.mean(syn_sp)),
        "energy_uj_mean": float(np.mean(energies)),
        "battery_years_proj": float(fold_metrics[0]["battery_years"]),
    }

    with open(os.path.join(args.out_dir, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n5-Fold Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
    