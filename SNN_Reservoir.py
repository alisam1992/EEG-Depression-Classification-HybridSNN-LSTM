"""
Spiking Reservoir (SNN) Pipeline
--------------------------------
- Loads eyes-open / eyes-closed HDF5 data
- Bins BDI labels into 4 categories
- Computes spike threshold from temporal derivatives
- Encodes spikes using snntorch.spikegen.delta
- Builds a 3D reservoir graph from Talairach coordinates
- Trains reservoir weights with a simple STDP rule
- Runs inference to collect spike counts & spike rasters
- Clusters reservoir neurons to electrodes and computes W_cut
- (Optional) Visualizes electrode graph and 3D connections
- Saves reservoir weights to HDF5

Inputs expected:
- open_data.h5     # datasets: X (N,T,Ch), labels (N,), patients_id (N,)
- close_data.h5    # datasets: X (N,T,Ch), labels (N,), patients_id (N,)
- talairach_brain_coordinates.csv
- koessler_mapping_checked_sorted.csv

Note: This script keeps your original algorithmic choices, but organizes and
      safeguards the flow with better structure and checks.

Author: (you)
"""

from __future__ import annotations

import os
import math
import random
import gc
from dataclasses import dataclass
from typing import Tuple, Optional, List

import numpy as np
import h5py
import pandas as pd
import scipy.spatial
import scipy.linalg
import matplotlib.pyplot as plt

# snn / torch (only what we actually use)
import torch
from snntorch import spikegen


# ------------------------------
# Configuration & Hyperparameters
# ------------------------------

@dataclass
class Paths:
    open_h5: str = "open_data.h5"    # expects datasets: X, labels, patients_id
    close_h5: str = "close_data.h5"  # expects datasets: X, labels, patients_id
    talairach_csv: str = "talairach_brain_coordinates.csv"
    koessler_csv: str = "koessler_mapping_checked_sorted.csv"
    out_reservoir_h5: str = "reservoir_weight_matrix_open.h5"
    out_wcut_h5: Optional[str] = None  # e.g., "Wcut_open.h5" if you want to save
    out_cube_png: Optional[str] = "cube_connections_open.png"  # 3D viz; None to skip


@dataclass
class HyperParams:
    # Encoding
    use_positive_spikes_only: bool = True  # positive polarity only (matches original)
    # Reservoir topology
    small_world_radius: float = 35.0       # neighbors within this distance can connect
    input_conn_scale: float = 3.0          # scale outgoing weights from input-neuron nodes
    # STDP / Neuron model
    dt: int = 1
    v_rest: float = 0.0
    v_thresh: float = 0.5
    refrac_period: int = 6
    stdp_rate: float = 1e-3
    tc_decay: float = 10.0                  # (not used as exp decay; we subtract constant)
    decay_subtract: Optional[float] = None  # if None, uses v_thresh/250
    epochs: int = 10
    # Training subset (optional trimming for speed/debug)
    limit_open_samples: Optional[int] = 5   # None = use all open samples
    # Plotting toggles
    plot_electrode_graph: bool = True
    plot_3d_connections: bool = True
    max_edges_3d: int = 500  # top-|weights| edges to draw in 3D


# ------------------------------
# Utilities
# ------------------------------

def bin_bdi_to_4(y: np.ndarray) -> np.ndarray:
    """Map BDI scores to 4 bins: 0:[0-13],1:[14-19],2:[20-28],3:[29+]."""
    y = np.asarray(y).astype(int)
    out = np.zeros_like(y, dtype=np.int32)
    out[(14 <= y) & (y <= 19)] = 1
    out[(20 <= y) & (y <= 28)] = 2
    out[y >= 29] = 3
    return out


def safe_inverse(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Elementwise inverse with zero on diagonal and eps guard for zeros."""
    y = np.zeros_like(x, dtype=float)
    mask = (np.abs(x) > eps)
    y[mask] = 1.0 / x[mask]
    return y


def ensure_2d(arr: np.ndarray) -> np.ndarray:
    """Make sure input is 2D (N,) -> (N,1)."""
    if arr.ndim == 1:
        return arr[:, None]
    return arr


# ------------------------------
# Data loading
# ------------------------------

def load_h5_triplet(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load (X, labels, patients_id) from an HDF5 path."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}")
    with h5py.File(path, "r") as f:
        X = f["X"][...]            # (N, T, Ch)
        labels = f["labels"][...]  # (N,)
        pids = f["patients_id"][...]
    return X, labels, pids


# ------------------------------
# Spike encoding
# ------------------------------

def compute_delta_threshold(X_open: np.ndarray) -> float:
    """
    Compute VT_mean from temporal absolute derivatives:
      VT = mean(mean(abs(diff(X)))_over_time_channels_per_sample over samples)
    Uses the original derivation: mean over samples of (mean(miu) + mean(gamma)),
    where miu=mean(|diff| across time per (ch,sample)), gamma=std(|diff| ...).
    """
    # Reorder to (time, channels, samples)
    X_openT0 = np.transpose(X_open, (1, 2, 0))  # (T, Ch, N)
    T = X_openT0.shape[0]

    # |diff| over time -> shape (T-1, Ch, N)
    X_diff = np.abs(np.diff(X_openT0, axis=0))

    # mean & std over time -> (Ch, N)
    miu = X_diff.mean(axis=0)
    gamma = X_diff.std(axis=0)

    # per-sample threshold per channel, then average over channels
    VT = miu.mean(axis=0) + gamma.mean(axis=0)  # (N,)
    VT_mean = float(VT.mean())                   # scalar
    return VT_mean


def encode_spikes_delta(X: np.ndarray, vt_mean: float, positive_only: bool = True) -> np.ndarray:
    """
    Encode spikes using snntorch.spikegen.delta.
    Input X as (N, T, Ch). Returns binary spikes as (N, T, Ch) in {0,1}.
    """
    # reorder to (T, N, Ch) for snntorch then back
    X_TNC = np.transpose(X, (1, 0, 2))
    spikes = spikegen.delta(
        torch.tensor(X_TNC, dtype=torch.float32),
        threshold=float(vt_mean),
        padding=False,
        off_spike=False  # match original
    ).numpy()
    spikes = np.transpose(spikes, (1, 0, 2))  # (N, T, Ch)
    if positive_only:
        spikes = (spikes > 0).astype(np.uint8)
    else:
        spikes = (spikes != 0).astype(np.uint8)
    return spikes


# ------------------------------
# Graph construction (reservoir)
# ------------------------------

@dataclass
class Reservoir:
    positions_reservoir: np.ndarray   # (Nr, 3)
    positions_input: np.ndarray       # (Ni, 3)
    input_to_reservoir_idx: List[int] # length Ni, each an index in [0..Nr-1]
    flag_matrix: np.ndarray           # (Nr, Nr) connectivity flags (0/1)
    weight_matrix: np.ndarray         # (Nr, Nr) real weights


def build_reservoir(paths: Paths, hp: HyperParams) -> Reservoir:
    # Load 3D coords
    res_pos = pd.read_csv(paths.talairach_csv, header=None).values.astype(float)     # (Nr,3)
    kmap = pd.read_csv(paths.koessler_csv, header=None, index_col=0, nrows=62).T     # electrodes x coord-cols
    inp_pos = kmap.T.values.astype(float)                                            # (Ni,3)

    Nr = res_pos.shape[0]
    Ni = inp_pos.shape[0]

    # Distances
    res_dist = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(res_pos))
    inp_to_res = scipy.spatial.distance.cdist(inp_pos, res_pos)

    # Flag matrix: connect if within radius; remove self-loops; remove edges to input nodes
    flag = (res_dist < hp.small_world_radius).astype(np.uint8)
    np.fill_diagonal(flag, 0)

    # Map each input electrode coordinate to its exact reservoir node index
    # (exact match assumed by original code)
    input_idx: List[int] = []
    pos_rows = {tuple(row): idx for idx, row in enumerate(res_pos)}
    for p in inp_pos:
        idx = pos_rows.get(tuple(p))
        if idx is None:
            raise ValueError("Input electrode position not found in reservoir positions."
                             " Check CSVs for exact coordinate matching.")
        input_idx.append(int(idx))

    # Zero out incoming edges to input nodes (as in the original code)
    for idx in input_idx:
        flag[:, idx] = 0

    # Break bidirectional duplicates randomly (keep one direction)
    rng = np.random.default_rng()
    for i in range(Nr):
        for j in range(i + 1, Nr):
            if flag[i, j] and flag[j, i]:
                if rng.random() > 0.5:
                    flag[i, j] = 0
                else:
                    flag[j, i] = 0

    # Weight init: random magnitude/sign, scaled by inverse distance and flag
    w = (np.random.rand(Nr, Nr) * np.sign(np.random.rand(Nr, Nr) - 0.2))
    res_dist_inv = safe_inverse(res_dist)
    np.fill_diagonal(res_dist_inv, 0.0)
    w = w * res_dist_inv * flag

    # Boost outgoing weights from input nodes
    for idx in input_idx:
        w[idx, :] = np.abs(w[idx, :]) * hp.input_conn_scale

    return Reservoir(
        positions_reservoir=res_pos,
        positions_input=inp_pos,
        input_to_reservoir_idx=input_idx,
        flag_matrix=flag,
        weight_matrix=w
    )


# ------------------------------
# STDP training (simple rule)
# ------------------------------

def stdp_train(
    reservoir: Reservoir,
    spikes_input: np.ndarray,   # (N, T, Ni) binary
    hp: HyperParams
) -> None:
    """In-place modifies reservoir.weight_matrix via simple STDP over epochs and samples."""
    Nr = reservoir.positions_reservoir.shape[0]
    Ni = reservoir.positions_input.shape[0]
    input_idx = reservoir.input_to_reservoir_idx
    W = reservoir.weight_matrix

    N, T, _ = spikes_input.shape

    decay_subtract = hp.decay_subtract if hp.decay_subtract is not None else (hp.v_thresh / 250.0)

    for epoch in range(hp.epochs):
        stdp_rate_n = float(round(hp.stdp_rate / math.sqrt(epoch + 1), 6))
        for n in range(N):
            v = np.full(Nr, hp.v_rest, dtype=float)
            refrac = np.zeros(Nr, dtype=np.int32)
            s = np.zeros(Nr, dtype=bool)
            last_s = np.zeros(Nr, dtype=np.int32)
            # input spikes for this sample: (T, Ni) -> bool
            inp_s = spikes_input[n].astype(bool)

            for t in range(T):
                dv = np.zeros(Nr, dtype=float)

                # contributions from input spikes at t
                if inp_s[t].any():
                    for i_ele in np.where(inp_s[t])[0]:
                        dv += W[input_idx[i_ele]]

                # contributions from reservoir spikes at t
                if s.any():
                    for pre in np.where(s)[0]:
                        dv += W[pre]

                # block refrac; update v
                dv = np.where(refrac > 0, 0.0, dv)
                v = np.maximum(0.0, v - decay_subtract + dv)

                # update refrac & spikes
                refrac = np.where(refrac > 0, refrac - hp.dt, 0)
                s = (v >= hp.v_thresh)
                refrac = np.where(s, hp.refrac_period, refrac)
                v = np.where(s, hp.v_rest, v)

                # STDP weight updates:
                #   - for presyn reservoirs that spiked: decrease outgoing weights slightly
                #   - for postsyn that spiked: increase incoming weights slightly
                #   dw scaled by time since last spike to avoid div-by-zero
                subtract = (t + 1) - last_s
                subtract = np.where(last_s > 0, subtract, 0)  # zero if never spiked
                with np.errstate(divide="ignore", invalid="ignore"):
                    dw_vec = np.zeros(Nr, dtype=float)
                    valid = subtract > 0
                    dw_vec[valid] = stdp_rate_n / subtract[valid]

                # LTP/LTD on existing edges only
                pres = np.where(s)[0]
                for pre in pres:
                    # LTD on outgoing from pre
                    mask = (reservoir.flag_matrix[pre] != 0)
                    W[pre, mask] -= dw_vec[mask]
                posts = pres
                for post in posts:
                    # LTP on incoming to post
                    mask = (reservoir.flag_matrix[:, post] != 0)
                    W[mask, post] += dw_vec[mask]

                # update last spike time
                last_s = np.where(s, t, last_s)


# ------------------------------
# Inference (spike counts / rasters)
# ------------------------------

def run_inference(
    reservoir: Reservoir,
    spikes_input: np.ndarray,  # (N, T, Ni)
    hp: HyperParams
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (spike_count [N,Nr], spike_raster [N,T,Nr]) after running reservoir forward."""
    Nr = reservoir.positions_reservoir.shape[0]
    input_idx = reservoir.input_to_reservoir_idx
    W = reservoir.weight_matrix

    N, T, Ni = spikes_input.shape
    decay_subtract = hp.decay_subtract if hp.decay_subtract is not None else (hp.v_thresh / 250.0)

    spike_count = np.zeros((N, Nr), dtype=np.uint32)
    raster = np.zeros((N, T, Nr), dtype=np.uint8)

    for n in range(N):
        v = np.full(Nr, hp.v_rest, dtype=float)
        refrac = np.zeros(Nr, dtype=np.int32)
        s = np.zeros(Nr, dtype=bool)
        inp_s = spikes_input[n].astype(bool)

        for t in range(T):
            spike_count[n] += s
            raster[n, t] += s.astype(np.uint8)

            dv = np.zeros(Nr, dtype=float)
            if inp_s[t].any():
                for i_ele in np.where(inp_s[t])[0]:
                    dv += W[input_idx[i_ele]]
                    # also count electrode “spikes” at their attached reservoir nodes
                    spike_count[n, input_idx[i_ele]] += 1
                    raster[n, t, input_idx[i_ele]] += 1

            if s.any():
                for pre in np.where(s)[0]:
                    dv += W[pre]

            dv = np.where(refrac > 0, 0.0, dv)
            v = np.maximum(0.0, v - decay_subtract + dv)

            refrac = np.where(refrac > 0, refrac - hp.dt, 0)
            s = (v >= hp.v_thresh)
            refrac = np.where(s, hp.refrac_period, refrac)
            v = np.where(s, hp.v_rest, v)

    return spike_count, raster


# ------------------------------
# Clustering & W_cut
# ------------------------------

def compute_wcut(
    reservoir: Reservoir,
    spike_count_total: np.ndarray  # (Nr,)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Wcut (Ni x Ni) by label propagation / spectral smoothing:
      - Build symmetric w from transmission (based on spike counts and flags)
      - Normalize, solve (I - 0.99 S) F = Y where Y maps input nodes to classes
      - Cluster labels from argmax(F) and compute inter-class cut weights
    Returns (Wcut2, label_cluster):
      Wcut2 is rounded-scaled Wcut (Ni x Ni), label_cluster is per-reservoir label idx (Nr,)
    """
    Nr = reservoir.positions_reservoir.shape[0]
    Ni = reservoir.positions_input.shape[0]
    F_flag = reservoir.flag_matrix

    # Symmetric “transmission” graph from spike counts (heuristic from original code)
    trans = np.repeat(spike_count_total[:, None], Nr, axis=1)
    trans = trans * F_flag
    w = trans + trans.T  # symmetric

    # normalized adjacency S = D^-1/2 w D^-1/2
    d = w.sum(axis=0)
    d = np.maximum(d, 1e-12)
    Dinv = np.diag(1.0 / np.sqrt(d))
    S = Dinv @ w @ Dinv

    # Y: one-hot for input nodes (labels are 0..Ni-1)
    Y = np.zeros((Nr, Ni), dtype=float)
    for i, idx in enumerate(reservoir.input_to_reservoir_idx):
        Y[idx, i] = 1.0

    F2 = np.eye(Nr) - 0.99 * S
    # Least-squares solve
    F = np.linalg.lstsq(F2, Y, rcond=None)[0]
    label_cluster = np.argmax(F, axis=1)  # (Nr,)

    # Aggregate inter-class weights between input classes
    Wcut = np.zeros((Ni, Ni), dtype=float)
    for i in range(Ni):
        Li = (label_cluster == i)
        Li = ensure_2d(Li.astype(float))
        for j in range(Ni):
            Lj = (label_cluster == j)
            Lj = ensure_2d(Lj.astype(float))
            Wcut[i, j] = float(Li.T @ w @ Lj)

    np.fill_diagonal(Wcut, 0.0)
    if Wcut.max() > 0:
        Wcut2 = np.round(Wcut / Wcut.max() * Ni)
    else:
        Wcut2 = Wcut.copy()
    return Wcut2, label_cluster


# ------------------------------
# Visualization
# ------------------------------

def plot_electrode_graph(Wcut2: np.ndarray, koessler_csv: str) -> None:
    """Circular electrode graph with edge widths scaled by Wcut2."""
    import networkx as nx

    labels = list(pd.read_csv(koessler_csv, header=None, index_col=0, nrows=62).columns)
    G = nx.Graph()
    n = Wcut2.shape[0]
    for i in range(n):
        G.add_node(i)

    for i in range(n):
        for j in range(i + 1, n):
            w = Wcut2[i, j]
            if w != 0:
                G.add_edge(i, j, weight=float(w))

    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    if edge_weights:
        mn, mx = min(edge_weights), max(edge_weights)
        span = max(mx - mn, 1e-9)
        scaled = [(w - mn) / span * 7 + 1 for w in edge_weights]
    else:
        scaled = []

    # Circular layout
    radius = 5.0
    pos = {}
    for i, node in enumerate(G.nodes()):
        ang = 2 * math.pi * i / n
        pos[node] = (radius * math.cos(ang), radius * math.sin(ang))

    # Random but reproducible node colors
    rng = np.random.default_rng(42)
    cx = plt.rcParams['axes.prop_cycle']
    palette = [c['color'] for c in cx]
    node_colors = [palette[int(rng.integers(0, len(palette)))] for _ in range(n)]

    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=200)
    nx.draw_networkx_edges(G, pos, width=scaled, edge_color='black', alpha=0.5)

    # Move labels slightly outside
    label_pos = {}
    for node, (x, y) in pos.items():
        ang = math.atan2(y, x)
        offset = 0.45
        label_pos[node] = ((radius + offset) * math.cos(ang), (radius + offset) * math.sin(ang))

    nx.draw_networkx_labels(G, label_pos, labels={i: lab for i, lab in enumerate(labels)},
                            font_size=8, font_color='black')

    plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_top_3d_connections(
    reservoir: Reservoir,
    weights: np.ndarray,
    spike_count_total: np.ndarray,
    max_edges: int = 500,
    out_png: Optional[str] = None
) -> None:
    """Plot a subset of strongest absolute connections in 3D."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    W = weights.copy()
    idx_nonzero = np.where(W != 0)
    if len(idx_nonzero[0]) == 0:
        print("[WARN] No nonzero reservoir connections to plot.")
        return

    wvals = np.abs(W[idx_nonzero])
    order = np.argsort(wvals)
    sel = order[-max_edges:] if max_edges < len(order) else order

    i0, j0 = idx_nonzero[0][sel], idx_nonzero[1][sel]
    pos = reservoir.positions_reservoir
    x, y, z = pos[i0, 0], pos[i0, 1], pos[i0, 2]
    x2, y2, z2 = pos[j0, 0], pos[j0, 1], pos[j0, 2]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Nodes colored by spike count
    sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                    c=spike_count_total, cmap="inferno", s=4, depthshade=True)

    # Draw edges, blue for positive, red for negative
    for i in range(len(sel)):
        w = W[i0[i], j0[i]]
        color = 'b' if w > 0 else 'r'
        ax.plot([x[i], x2[i]], [y[i], y2[i]], [z[i], z2[i]], color, linewidth=0.5)

    ax.set_xlabel('X', fontsize=8, labelpad=0)
    ax.set_ylabel('Y', fontsize=8, labelpad=0)
    ax.set_zlabel('Z', fontsize=8, labelpad=0)
    ax.grid(False)
    ax.tick_params(axis='both', which='major', labelsize=6, pad=0)
    plt.colorbar(sc, pad=0.09)

    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=600)
        print(f"[OK] saved 3D connections to {out_png}")
    plt.show()


# ------------------------------
# Main
# ------------------------------

def main(paths: Paths = Paths(), hp: HyperParams = HyperParams()) -> None:
    # 1) Load data
    X_open, y_open, pid_open = load_h5_triplet(paths.open_h5)
    X_close, y_close, pid_close = load_h5_triplet(paths.close_h5)

    # 2) Label binning -> 4 classes
    y_open4 = bin_bdi_to_4(y_open)
    y_close4 = bin_bdi_to_4(y_close)

    # 3) (Optional) restrict to one class / a few samples (matches your original open==3 then [:5])
    if hp.limit_open_samples is not None:
        mask_open_3 = (y_open4 == 3)
        if mask_open_3.any():
            X_open = X_open[mask_open_3]
            y_open4 = y_open4[mask_open_3]
        X_open = X_open[:hp.limit_open_samples]
        y_open4 = y_open4[:hp.limit_open_samples]

    # Sanity
    if X_open.ndim != 3:
        raise ValueError("X_open must be (N,T,Ch)")
    N_open, T, Ch = X_open.shape

    # 4) Spike threshold from open data
    vt_mean = compute_delta_threshold(X_open)

    # 5) Encode spikes for open/close
    spikes_open = encode_spikes_delta(X_open, vt_mean, positive_only=hp.use_positive_spikes_only)  # (N,T,Ch)
    spikes_close = encode_spikes_delta(X_close, vt_mean, positive_only=hp.use_positive_spikes_only)

    # 6) Load reservoir graph & init weights
    reservoir = build_reservoir(paths, hp)

    # Map electrode channels (Ch) to Ni: must match Koessler list length
    Ni = reservoir.positions_input.shape[0]
    if Ch != Ni:
        raise ValueError(f"Channel count in data ({Ch}) doesn't match electrode mapping ({Ni}).")

    # 7) Train reservoir with STDP on open spikes
    # reorder spikes to (N,T,Ni) already (spikes_open is (N,T,Ch==Ni))
    stdp_train(reservoir, spikes_open, hp)

    # 8) Inference on open (collect spike counts & raster)
    spike_count, raster = run_inference(reservoir, spikes_open, hp)
    spike_count_total = spike_count.sum(axis=0)  # (Nr,)

    # 9) Clustering and Wcut
    Wcut2, label_cluster = compute_wcut(reservoir, spike_count_total)

    # 10) Visualizations
    if hp.plot_electrode_graph:
        plot_electrode_graph(Wcut2, paths.koessler_csv)

    if hp.plot_3d_connections:
        plot_top_3d_connections(
            reservoir,
            weights=reservoir.weight_matrix,
            spike_count_total=spike_count_total,
            max_edges=hp.max_edges_3d,
            out_png=paths.out_cube_png
        )

    # 11) Save reservoir weights
    with h5py.File(paths.out_reservoir_h5, "w") as f:
        f.create_dataset("reservoir_weight_matrix", data=reservoir.weight_matrix)
        f.create_dataset("flag_matrix", data=reservoir.flag_matrix)
    if paths.out_wcut_h5:
        with h5py.File(paths.out_wcut_h5, "w") as f:
            f.create_dataset("Wcut", data=Wcut2)

    print("[DONE] Reservoir training + inference complete.")
    gc.collect()


if __name__ == "__main__":
    # Optional: set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    main()
