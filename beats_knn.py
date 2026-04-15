"""
BEATs + kNN Anomalous Sound Detection for DCASE 2026 Task 2.

Replaces the baseline AE with a pre-trained BEATs feature extractor
and kNN-based anomaly scoring.

Usage:
    python beats_knn.py --dataset DCASE2026T2ToyCar -d
    python beats_knn.py --dataset DCASE2026T2ToyCar -d --k 5
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


# ============================================================
# BEATs model loading
# ============================================================

def load_beats_model(checkpoint_path, device):
    """Load BEATs pre-trained model from checkpoint."""
    # Import BEATs (cloned repo must be on sys.path)
    beats_dir = Path(__file__).parent / "BEATs"
    if str(beats_dir) not in sys.path:
        sys.path.insert(0, str(beats_dir))
    from BEATs import BEATs, BEATsConfig

    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = BEATsConfig(ckpt["cfg"])
    model = BEATs(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.to(device)
    return model


def extract_beats_embedding(model, waveform, sr, device):
    """
    Extract a single embedding vector from a waveform using BEATs.

    Args:
        model: BEATs model
        waveform: numpy array (samples,) mono audio
        sr: sample rate
        device: torch device

    Returns:
        numpy array of shape (embed_dim,)
    """
    # BEATs expects 16kHz mono
    if sr != 16000:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)

    # Convert to tensor: (1, num_samples)
    wav_tensor = torch.from_numpy(waveform).float().unsqueeze(0).to(device)

    padding_mask = torch.zeros(wav_tensor.shape[0], wav_tensor.shape[1],
                               dtype=torch.bool, device=device)

    with torch.no_grad():
        # features shape: (1, num_frames, embed_dim)
        features = model.extract_features(wav_tensor, padding_mask=padding_mask)[0]

    # Mean-pool over time frames to get a single vector
    embedding = features.mean(dim=1).squeeze(0).cpu().numpy()
    return embedding


# ============================================================
# Data loading utilities
# ============================================================

def get_audio_files(data_dir, pattern="*.wav"):
    """Get sorted list of wav files in a directory."""
    return sorted(Path(data_dir).glob(pattern))


def parse_filename(filepath):
    """
    Parse DCASE filename to extract metadata.

    Example: section_00_source_test_normal_0002_noAttribute.wav
    Returns: (section_id, domain, split, label, index)
    """
    name = Path(filepath).stem
    parts = name.split("_")
    # section_00_source_test_normal_0002_noAttribute
    section_id = parts[1]  # "00"
    domain = parts[2]      # "source" or "target"
    split = parts[3]       # "train" or "test"
    label = parts[4]       # "normal" or "anomaly"
    return section_id, domain, split, label


def load_audio_mono(filepath, channel=0):
    """Load audio file, return mono waveform and sample rate."""
    audio, sr = librosa.load(filepath, sr=None, mono=False)
    if audio.ndim == 2:
        audio = audio[channel]  # Use near-mic (channel 0)
    return audio, sr


# ============================================================
# Machine type / dataset mapping
# ============================================================

DATASET_TO_MACHINE = {
    "DCASE2026T2ToyCar": ("dcase2026t2", "ToyCar"),
    "DCASE2026T2ToyCarEmu": ("dcase2026t2", "ToyCarEmu"),
    "DCASE2026T2bearingEmu": ("dcase2026t2", "bearingEmu"),
    "DCASE2026T2fan": ("dcase2026t2", "fan"),
    "DCASE2026T2gearboxEmu": ("dcase2026t2", "gearboxEmu"),
    "DCASE2026T2sliderEmu": ("dcase2026t2", "sliderEmu"),
    "DCASE2026T2valveEmu": ("dcase2026t2", "valveEmu"),
}


def get_data_dirs(dataset, data_root="./data", dev=True):
    """Get train and test directories for a dataset."""
    challenge, machine = DATASET_TO_MACHINE[dataset]
    split = "dev_data" if dev else "eval_data"
    base = Path(data_root) / challenge / split / "raw" / machine
    return base / "train", base / "test"


# ============================================================
# Main pipeline
# ============================================================

def extract_all_embeddings(model, audio_files, device, desc="Extracting"):
    """Extract BEATs embeddings for a list of audio files."""
    embeddings = []
    filenames = []
    for fpath in tqdm(audio_files, desc=desc):
        audio, sr = load_audio_mono(fpath)
        emb = extract_beats_embedding(model, audio, sr, device)
        embeddings.append(emb)
        filenames.append(fpath.name)
    return np.array(embeddings), filenames


def compute_knn_scores(train_embeddings, test_embeddings, k=2):
    """
    Compute anomaly scores using kNN distance.
    Score = mean distance to k nearest neighbors in training set.
    """
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    nn.fit(train_embeddings)
    distances, _ = nn.kneighbors(test_embeddings)
    # Mean distance to k neighbors as anomaly score
    scores = distances.mean(axis=1)
    return scores


def compute_train_knn_scores(train_embeddings, k=2):
    """
    Compute kNN scores for training data using leave-one-out.
    Fits with k+1 neighbors and drops the nearest (self-match).
    """
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
    nn.fit(train_embeddings)
    distances, _ = nn.kneighbors(train_embeddings)
    # Skip column 0 (distance to self = 0), use columns 1..k
    scores = distances[:, 1:].mean(axis=1)
    return scores


def save_csv(filepath, data):
    """Save data to CSV."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(data)


def evaluate_and_report(
    y_true, y_pred, domain_list, train_scores,
    filenames, dataset, section_id, result_dir, seed,
    decision_threshold_quantile=0.9, max_fpr=0.1
):
    """Evaluate scores and save results (compatible with baseline format)."""
    section_name = f"section_{section_id}"

    # Percentile-based threshold on training kNN scores
    # Training set is all normal, so the threshold captures the upper tail of normal distances
    decision_threshold = np.percentile(train_scores, decision_threshold_quantile * 100)
    print(f"Decision threshold (train {decision_threshold_quantile*100:.0f}th percentile): {decision_threshold:.6f}")

    # Save anomaly scores
    anomaly_score_list = [[fn, score] for fn, score in zip(filenames, y_pred)]
    anomaly_score_csv = result_dir / f"anomaly_score_{dataset}_{section_name}_test_seed{seed}.csv"
    save_csv(anomaly_score_csv, anomaly_score_list)
    print(f"anomaly score result ->  {anomaly_score_csv}")

    # Save decision results
    decision_result_list = [
        [fn, 1 if score > decision_threshold else 0]
        for fn, score in zip(filenames, y_pred)
    ]
    decision_result_csv = result_dir / f"decision_result_{dataset}_{section_name}_test_seed{seed}.csv"
    save_csv(decision_result_csv, decision_result_list)
    print(f"decision result ->  {decision_result_csv}")

    # Extract per-domain scores
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    domain_list = np.array(domain_list)

    source_mask = domain_list == "source"
    target_mask = domain_list == "target"

    # Source AUC (source normals + all anomalies)
    s_auc_mask = source_mask | (y_true == 1)
    t_auc_mask = target_mask | (y_true == 1)

    y_true_s = y_true[source_mask]
    y_pred_s = y_pred[source_mask]
    y_true_t = y_true[target_mask]
    y_pred_t = y_pred[target_mask]

    auc_s = metrics.roc_auc_score(y_true[s_auc_mask], y_pred[s_auc_mask])
    auc_t = metrics.roc_auc_score(y_true[t_auc_mask], y_pred[t_auc_mask])
    p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
    p_auc_s = metrics.roc_auc_score(y_true_s, y_pred_s, max_fpr=max_fpr)
    p_auc_t = metrics.roc_auc_score(y_true_t, y_pred_t, max_fpr=max_fpr)

    # Precision, recall, F1 per domain
    dec_s = [1 if x > decision_threshold else 0 for x in y_pred_s]
    tn_s, fp_s, fn_s, tp_s = metrics.confusion_matrix(y_true_s, dec_s).ravel()
    prec_s = tp_s / max(tp_s + fp_s, sys.float_info.epsilon)
    recall_s = tp_s / max(tp_s + fn_s, sys.float_info.epsilon)
    f1_s = 2.0 * prec_s * recall_s / max(prec_s + recall_s, sys.float_info.epsilon)

    dec_t = [1 if x > decision_threshold else 0 for x in y_pred_t]
    tn_t, fp_t, fn_t, tp_t = metrics.confusion_matrix(y_true_t, dec_t).ravel()
    prec_t = tp_t / max(tp_t + fp_t, sys.float_info.epsilon)
    recall_t = tp_t / max(tp_t + fn_t, sys.float_info.epsilon)
    f1_t = 2.0 * prec_t * recall_t / max(prec_t + recall_t, sys.float_info.epsilon)

    print(f"AUC (source) : {auc_s}")
    print(f"AUC (target) : {auc_t}")
    print(f"pAUC : {p_auc}")
    print(f"pAUC (source) : {p_auc_s}")
    print(f"pAUC (target) : {p_auc_t}")
    print(f"precision (source) : {prec_s}")
    print(f"recall (source) : {recall_s}")
    print(f"F1 score (source) : {f1_s}")
    print(f"precision (target) : {prec_t}")
    print(f"recall (target) : {recall_t}")
    print(f"F1 score (target) : {f1_t}")

    # Save result CSV (same format as baseline)
    csv_lines = [
        ["section", "AUC (source)", "AUC (target)", "pAUC",
         "pAUC (source)", "pAUC (target)",
         "precision (source)", "precision (target)",
         "recall (source)", "recall (target)",
         "F1 score (source)", "F1 score (target)"],
        [section_id, auc_s, auc_t, p_auc, p_auc_s, p_auc_t,
         prec_s, prec_t, recall_s, recall_t, f1_s, f1_t],
        ["arithmetic mean", auc_s, auc_t, p_auc, p_auc_s, p_auc_t,
         prec_s, prec_t, recall_s, recall_t, f1_s, f1_t],
        ["harmonic mean", auc_s, auc_t, p_auc, p_auc_s, p_auc_t,
         prec_s, prec_t, recall_s, recall_t, f1_s, f1_t],
    ]
    result_csv = result_dir / f"result_{dataset}_test_seed{seed}_roc.csv"
    save_csv(result_csv, csv_lines)
    print(f"results -> {result_csv}")

    return {
        "auc_s": auc_s, "auc_t": auc_t, "p_auc": p_auc,
        "p_auc_s": p_auc_s, "p_auc_t": p_auc_t,
        "f1_s": f1_s, "f1_t": f1_t,
    }


def main():
    parser = argparse.ArgumentParser(description="BEATs + kNN Anomaly Detection")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name, e.g. DCASE2026T2ToyCar")
    parser.add_argument("-d", "--dev", action="store_true",
                        help="Use development dataset")
    parser.add_argument("-e", "--eval", action="store_true",
                        help="Use evaluation dataset")
    parser.add_argument("--beats_ckpt", type=str,
                        default="./BEATs/BEATs_iter3_plus_AS2M.pt",
                        help="Path to BEATs checkpoint")
    parser.add_argument("--k", type=int, default=2,
                        help="Number of nearest neighbors (default: 2)")
    parser.add_argument("--seed", type=int, default=13711,
                        help="Random seed (for result file naming)")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Root data directory")
    parser.add_argument("--result_dir", type=str, default="./results",
                        help="Result output directory")
    parser.add_argument("--decision_threshold", type=float, default=0.9,
                        help="Quantile for decision threshold (default: 0.9)")
    parser.add_argument("--max_fpr", type=float, default=0.1,
                        help="Max FPR for pAUC (default: 0.1)")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU device ID")
    args = parser.parse_args()

    print(f"BEATs + kNN Anomaly Detection")
    print(f"Dataset: {args.dataset}")
    print(f"k = {args.k}")
    print(f"BEATs checkpoint: {args.beats_ckpt}")
    print()

    # Device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load BEATs
    print("\n============== LOADING BEATs MODEL ==============")
    if not os.path.exists(args.beats_ckpt):
        print(f"ERROR: BEATs checkpoint not found at {args.beats_ckpt}")
        print("Download it first:")
        print("  mkdir -p BEATs && cd BEATs")
        print("  wget https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M.pt")
        sys.exit(1)
    model = load_beats_model(args.beats_ckpt, device)
    print("BEATs model loaded successfully.")

    # Get data directories
    train_dir, test_dir = get_data_dirs(args.dataset, args.data_dir, dev=args.dev)
    print(f"\nTrain dir: {train_dir}")
    print(f"Test dir:  {test_dir}")

    # Get file lists
    train_files = get_audio_files(train_dir)
    test_files = get_audio_files(test_dir)
    print(f"Train files: {len(train_files)}")
    print(f"Test files:  {len(test_files)}")

    # ============== Extract embeddings ==============
    print("\n============== EXTRACTING TRAIN EMBEDDINGS ==============")
    train_embeddings, train_filenames = extract_all_embeddings(
        model, train_files, device, desc="Train"
    )
    print(f"Train embeddings shape: {train_embeddings.shape}")

    print("\n============== EXTRACTING TEST EMBEDDINGS ==============")
    test_embeddings, test_filenames = extract_all_embeddings(
        model, test_files, device, desc="Test"
    )
    print(f"Test embeddings shape: {test_embeddings.shape}")

    # ============== kNN scoring ==============
    print(f"\n============== kNN SCORING (k={args.k}) ==============")
    scores = compute_knn_scores(train_embeddings, test_embeddings, k=args.k)

    # Compute training scores for percentile-based threshold
    train_scores = compute_train_knn_scores(train_embeddings, k=args.k)
    print(f"Train score range: [{train_scores.min():.6f}, {train_scores.max():.6f}]")
    print(f"Train score 90th percentile: {np.percentile(train_scores, 90):.6f}")

    # Parse labels and domains from filenames
    y_true = []
    domain_list = []
    section_id = None
    for fn in test_filenames:
        sid, domain, split, label = parse_filename(fn)
        y_true.append(1 if label == "anomaly" else 0)
        domain_list.append(domain)
        if section_id is None:
            section_id = sid

    # ============== Evaluate ==============
    print(f"\n============== RESULTS ==============")
    result_dir = Path(args.result_dir) / ("dev_data" if args.dev else "eval_data") / "beats_knn"
    result_dir.mkdir(parents=True, exist_ok=True)

    evaluate_and_report(
        y_true=y_true,
        y_pred=scores,
        domain_list=domain_list,
        train_scores=train_scores,
        filenames=test_filenames,
        dataset=args.dataset,
        section_id=section_id,
        result_dir=result_dir,
        seed=args.seed,
        decision_threshold_quantile=args.decision_threshold,
        max_fpr=args.max_fpr,
    )

    # Save embeddings for potential future ensemble use
    embed_dir = Path("models/saved_model/beats_knn")
    embed_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        embed_dir / f"embeddings_{args.dataset}_seed{args.seed}.npz",
        train_embeddings=train_embeddings,
        test_embeddings=test_embeddings,
        test_filenames=test_filenames,
        test_scores=scores,
        test_labels=np.array(y_true),
        test_domains=np.array(domain_list),
    )
    print(f"\nEmbeddings saved to {embed_dir}")


if __name__ == "__main__":
    main()
