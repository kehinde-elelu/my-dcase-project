"""
Contrastive fine-tuning of BEATs on normal machine audio.

Creates machine-specific embeddings by training BEATs with SimCLR-style
contrastive learning on augmented views of normal training audio.
Only the last N transformer layers + a projection head are trained.

Usage:
    python finetune_beats.py --dataset DCASE2026T2ToyCar -d --epochs 50
    python finetune_beats.py --dataset DCASE2026T2ToyCar -d --epochs 50 --unfreeze_layers 4
"""

import argparse
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
from tqdm import tqdm


# ============================================================
# BEATs loading (reuse from beats_knn.py)
# ============================================================

def load_beats_model(checkpoint_path, device):
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


# ============================================================
# Dataset mapping (same as beats_knn.py)
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


# ============================================================
# Audio augmentations
# ============================================================

class AudioAugmentor:
    """Audio augmentations for contrastive learning."""

    def __init__(self, sr=16000):
        self.sr = sr

    def add_gaussian_noise(self, waveform, snr_range=(10, 30)):
        snr_db = np.random.uniform(*snr_range)
        signal_power = (waveform ** 2).mean()
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
        return waveform + noise

    def time_shift(self, waveform, max_shift_ratio=0.2):
        shift = int(waveform.shape[-1] * np.random.uniform(-max_shift_ratio, max_shift_ratio))
        return torch.roll(waveform, shifts=shift, dims=-1)

    def time_mask(self, waveform, max_mask_ratio=0.15):
        length = waveform.shape[-1]
        mask_len = int(length * np.random.uniform(0, max_mask_ratio))
        start = np.random.randint(0, max(length - mask_len, 1))
        waveform = waveform.clone()
        waveform[..., start:start + mask_len] = 0
        return waveform

    def amplitude_scale(self, waveform, scale_range=(0.7, 1.3)):
        scale = np.random.uniform(*scale_range)
        return waveform * scale

    def speed_perturb(self, waveform, speed_range=(0.9, 1.1)):
        speed = np.random.uniform(*speed_range)
        # Resample to change speed
        orig_len = waveform.shape[-1]
        new_sr = int(self.sr * speed)
        # Use simple interpolation for speed change
        if speed != 1.0:
            resampled = torchaudio.functional.resample(waveform, new_sr, self.sr)
            # Pad or trim to original length
            if resampled.shape[-1] > orig_len:
                resampled = resampled[..., :orig_len]
            elif resampled.shape[-1] < orig_len:
                pad_len = orig_len - resampled.shape[-1]
                resampled = F.pad(resampled, (0, pad_len))
            return resampled
        return waveform

    def __call__(self, waveform):
        """Apply a random subset of augmentations."""
        # Always apply noise + one other augmentation
        waveform = self.add_gaussian_noise(waveform)

        # Randomly pick 1-2 additional augmentations
        augs = [self.time_shift, self.time_mask, self.amplitude_scale, self.speed_perturb]
        n_augs = np.random.randint(1, 3)
        chosen = np.random.choice(len(augs), size=n_augs, replace=False)
        for idx in chosen:
            waveform = augs[idx](waveform)

        return waveform


# ============================================================
# Dataset
# ============================================================

class NormalAudioDataset(Dataset):
    """Dataset of normal audio files for contrastive fine-tuning."""

    def __init__(self, audio_dir, sr=16000, segment_length=None):
        self.files = sorted(Path(audio_dir).glob("*.wav"))
        self.sr = sr
        self.segment_length = segment_length  # None = use full audio
        self.augmentor = AudioAugmentor(sr=sr)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio, sr = librosa.load(self.files[idx], sr=None, mono=False)
        if audio.ndim == 2:
            audio = audio[0]  # channel 0 (near-mic)
        if sr != self.sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)

        waveform = torch.from_numpy(audio).float()

        # Random crop if segment_length is set
        if self.segment_length is not None and waveform.shape[-1] > self.segment_length:
            start = np.random.randint(0, waveform.shape[-1] - self.segment_length)
            waveform = waveform[start:start + self.segment_length]

        # Create two augmented views
        view1 = self.augmentor(waveform.clone())
        view2 = self.augmentor(waveform.clone())

        return view1, view2


def collate_fn(batch):
    """Collate variable-length audio into a batch with padding."""
    views1, views2 = zip(*batch)

    # Find max length in batch
    max_len = max(max(v.shape[-1] for v in views1), max(v.shape[-1] for v in views2))

    # Pad all to max_len
    padded1 = torch.stack([F.pad(v, (0, max_len - v.shape[-1])) for v in views1])
    padded2 = torch.stack([F.pad(v, (0, max_len - v.shape[-1])) for v in views2])

    return padded1, padded2


# ============================================================
# Projection head for contrastive learning
# ============================================================

class ProjectionHead(nn.Module):
    """MLP projection head (SimCLR-style)."""

    def __init__(self, input_dim=768, hidden_dim=512, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# NT-Xent contrastive loss
# ============================================================

def nt_xent_loss(z1, z2, temperature=0.1):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.
    z1, z2: (batch_size, proj_dim) — projected embeddings of two views.
    """
    batch_size = z1.shape[0]
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Concatenate: [z1_0, z1_1, ..., z2_0, z2_1, ...]
    z = torch.cat([z1, z2], dim=0)  # (2B, D)

    # Similarity matrix
    sim = torch.mm(z, z.t()) / temperature  # (2B, 2B)

    # Mask out self-similarity
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim.masked_fill_(mask, -1e9)

    # Positive pairs: (i, i+B) and (i+B, i)
    pos_idx_1 = torch.arange(batch_size, device=z.device) + batch_size
    pos_idx_2 = torch.arange(batch_size, device=z.device)
    pos_idx = torch.cat([pos_idx_1, pos_idx_2])  # (2B,)

    # Cross-entropy loss where positive pair is the target
    labels = pos_idx
    loss = F.cross_entropy(sim, labels)
    return loss


# ============================================================
# Fine-tuning wrapper
# ============================================================

class BEATsFineTuner(nn.Module):
    """Wraps BEATs + projection head for contrastive fine-tuning."""

    def __init__(self, beats_model, proj_dim=128, unfreeze_layers=4):
        super().__init__()
        self.beats = beats_model
        self.proj_head = ProjectionHead(
            input_dim=beats_model.cfg.encoder_embed_dim,
            hidden_dim=512,
            output_dim=proj_dim,
        )

        # Freeze everything first
        for param in self.beats.parameters():
            param.requires_grad = False

        # Unfreeze last N transformer layers
        total_layers = len(self.beats.encoder.layers)
        for i in range(total_layers - unfreeze_layers, total_layers):
            for param in self.beats.encoder.layers[i].parameters():
                param.requires_grad = True

        # Also unfreeze layer_norm
        for param in self.beats.layer_norm.parameters():
            param.requires_grad = True

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(f"Trainable parameters: {n_trainable:,} / {n_total:,} "
              f"({100*n_trainable/n_total:.1f}%)")

    def extract_embedding(self, waveform):
        """Extract mean-pooled embedding from BEATs."""
        padding_mask = torch.zeros(
            waveform.shape[0], waveform.shape[1],
            dtype=torch.bool, device=waveform.device,
        )
        features, _ = self.beats.extract_features(waveform, padding_mask=padding_mask)
        # Mean pool over time
        embedding = features.mean(dim=1)  # (B, 768)
        return embedding

    def forward(self, view1, view2):
        """Forward pass for contrastive learning."""
        emb1 = self.extract_embedding(view1)
        emb2 = self.extract_embedding(view2)

        z1 = self.proj_head(emb1)
        z2 = self.proj_head(emb2)

        return z1, z2, emb1, emb2


# ============================================================
# Training loop
# ============================================================

def train_one_epoch(model, dataloader, optimizer, device, temperature=0.1):
    model.train()
    total_loss = 0
    n_batches = 0

    for view1, view2 in dataloader:
        view1 = view1.to(device)
        view2 = view2.to(device)

        z1, z2, _, _ = model(view1, view2)
        loss = nt_xent_loss(z1, z2, temperature=temperature)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Fine-tune BEATs with contrastive learning")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name, e.g. DCASE2026T2ToyCar")
    parser.add_argument("-d", "--dev", action="store_true",
                        help="Use development dataset")
    parser.add_argument("--beats_ckpt", type=str,
                        default="./BEATs/BEATs_iter3_plus_AS2M.pt",
                        help="Path to BEATs checkpoint")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="NT-Xent temperature (default: 0.1)")
    parser.add_argument("--unfreeze_layers", type=int, default=4,
                        help="Number of transformer layers to unfreeze (default: 4)")
    parser.add_argument("--proj_dim", type=int, default=128,
                        help="Projection head output dim (default: 128)")
    parser.add_argument("--segment_sec", type=float, default=4.0,
                        help="Audio segment length in seconds for training (default: 4.0)")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU device ID")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Root data directory")
    parser.add_argument("--save_dir", type=str, default="./models/saved_model/beats_finetuned",
                        help="Directory to save fine-tuned model")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data directory
    challenge, machine = DATASET_TO_MACHINE[args.dataset]
    split = "dev_data" if args.dev else "eval_data"
    train_dir = Path(args.data_dir) / challenge / split / "raw" / machine / "train"
    print(f"Train dir: {train_dir}")
    print(f"Train files: {len(list(train_dir.glob('*.wav')))}")

    # Load BEATs
    print("\n============== LOADING BEATs MODEL ==============")
    beats_model = load_beats_model(args.beats_ckpt, device)

    # Create fine-tuner
    model = BEATsFineTuner(
        beats_model,
        proj_dim=args.proj_dim,
        unfreeze_layers=args.unfreeze_layers,
    ).to(device)

    # Dataset and dataloader
    segment_length = int(args.segment_sec * 16000)
    dataset = NormalAudioDataset(train_dir, sr=16000, segment_length=segment_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True,
    )
    print(f"Dataset: {len(dataset)} files, segment={args.segment_sec}s, batch={args.batch_size}")
    print(f"Batches per epoch: {len(dataloader)}")

    # Optimizer — higher LR for projection head, lower for BEATs layers
    beats_params = [p for p in model.beats.parameters() if p.requires_grad]
    proj_params = list(model.proj_head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": beats_params, "lr": args.lr * 0.1},
        {"params": proj_params, "lr": args.lr},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )

    # Training
    print(f"\n============== TRAINING ({args.epochs} epochs) ==============")
    best_loss = float("inf")
    save_dir = Path(args.save_dir) / args.dataset
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, dataloader, optimizer, device, args.temperature)
        scheduler.step()
        lr_beats = optimizer.param_groups[0]["lr"]
        lr_proj = optimizer.param_groups[1]["lr"]

        print(f"Epoch {epoch:3d}/{args.epochs}  loss={loss:.4f}  "
              f"lr_beats={lr_beats:.2e}  lr_proj={lr_proj:.2e}")

        if loss < best_loss:
            best_loss = loss
            # Save the fine-tuned BEATs state_dict (without projection head)
            save_path = save_dir / "beats_finetuned_best.pt"
            torch.save({
                "beats_state_dict": model.beats.state_dict(),
                "cfg": model.beats.cfg.__dict__,
                "epoch": epoch,
                "loss": loss,
                "args": vars(args),
            }, save_path)

    # Save final model too
    final_path = save_dir / "beats_finetuned_final.pt"
    torch.save({
        "beats_state_dict": model.beats.state_dict(),
        "cfg": model.beats.cfg.__dict__,
        "epoch": args.epochs,
        "loss": loss,
        "args": vars(args),
    }, final_path)

    print(f"\nBest loss: {best_loss:.4f}")
    print(f"Best model saved to: {save_dir / 'beats_finetuned_best.pt'}")
    print(f"Final model saved to: {final_path}")
    print("\nTo use the fine-tuned model with beats_knn.py:")
    print(f"  python beats_knn.py --dataset {args.dataset} -d "
          f"--beats_ckpt {save_dir / 'beats_finetuned_best.pt'} "
          f"--scoring ensemble --k 1")


if __name__ == "__main__":
    main()
