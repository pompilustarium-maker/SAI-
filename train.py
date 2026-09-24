"""
Production-Ready Multi-Task Training and Evaluation Pipeline for WikiArt.
Features:
  - Joint multi-head optimization for Genre and Historical Century.
  - Resolves severe class imbalance using balanced Class Weights & Focal Loss.
  - Two-stage curriculum learning:
      * Stage 1: Warm up classification heads with frozen backbone.
      * Stage 2: End-to-end fine-tuning of top backbone layers with CosineAnnealingLR.
  - Mixed Precision (torch.amp) for acceleration and memory conservation.
  - Metrics: Accuracy, Macro-F1, Weighted-F1, Confusion Matrix export.
"""

import os
import sys
import json
import argparse
import time
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm

from data.dataset import load_and_preprocess_wikiart, build_dataloaders
from models.multi_head_art_net import MultiHeadArtNet, MultiTaskCriterion


def parse_args():
    parser = argparse.ArgumentParser(description="Train Multi-Task WikiArt Network (Genre + Century)")
    
    # Paths
    parser.add_argument("--csv_path", type=str, default="/content/wikiart_dataset/classes.csv",
                        help="Path to WikiArt metadata classes.csv")
    parser.add_argument("--data_dir", type=str, default="/content/wikiart_dataset",
                        help="Root directory containing artwork images")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                        help="Directory to save model checkpoints and logs")

    # Architecture
    parser.add_argument("--backbone", type=str, default="convnext_tiny",
                        help="Backbone: convnext_tiny, convnext_small, swin_t, efficientnet_v2_s, resnet50")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Input image resolution (e.g. 224, 384)")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate in classification MLP heads")

    # Training schedule
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--epochs_stage1", type=int, default=5,
                        help="Stage 1: Epochs to train heads with frozen backbone")
    parser.add_argument("--epochs_stage2", type=int, default=20,
                        help="Stage 2: Epochs to fine-tune backbone + heads end-to-end")
    parser.add_argument("--lr_stage1", type=float, default=1e-3,
                        help="Learning rate for Stage 1 (heads only)")
    parser.add_argument("--lr_stage2_heads", type=float, default=1e-4,
                        help="Learning rate for heads in Stage 2")
    parser.add_argument("--lr_stage2_backbone", type=float, default=2e-5,
                        help="Learning rate for backbone in Stage 2")
    parser.add_argument("--weight_decay", type=float, default=1e-2,
                        help="Weight decay for AdamW")

    # Loss & Task balance
    parser.add_argument("--genre_weight", type=float, default=1.0,
                        help="Task loss weight lambda for Genre")
    parser.add_argument("--century_weight", type=float, default=0.8,
                        help="Task loss weight lambda for Century")
    parser.add_argument("--use_focal_loss", action="store_true",
                        help="Use Multi-Class Focal Loss instead of standard CrossEntropy")
    parser.add_argument("--min_genre_samples", type=int, default=50,
                        help="Minimum samples required to retain a genre category")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker subprocesses")
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable automatic mixed precision")

    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: MultiTaskCriterion,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    use_amp: bool = True
) -> Tuple[float, float, float]:
    """Runs a single training epoch with mixed precision."""
    model.train()
    total_loss, total_g_loss, total_c_loss = 0.0, 0.0, 0.0
    num_samples = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, genre_targets, century_targets, _ in pbar:
        images = images.to(device, non_blocking=True)
        genre_targets = genre_targets.to(device, non_blocking=True)
        century_targets = century_targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        if use_amp and device.type == "cuda":
            with torch.amp.autocast('cuda'):
                g_logits, c_logits = model(images)
                loss, loss_g, loss_c = criterion(g_logits, c_logits, genre_targets, century_targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            g_logits, c_logits = model(images)
            loss, loss_g, loss_c = criterion(g_logits, c_logits, genre_targets, century_targets)
            loss.backward()
            optimizer.step()

        batch_sz = images.size(0)
        total_loss += loss.item() * batch_sz
        total_g_loss += loss_g.item() * batch_sz
        total_c_loss += loss_c.item() * batch_sz
        num_samples += batch_sz

        pbar.set_postfix({
            'loss': f"{loss.item():.3f}",
            'g_loss': f"{loss_g.item():.3f}",
            'c_loss': f"{loss_c.item():.3f}"
        })

    return total_loss / num_samples, total_g_loss / num_samples, total_c_loss / num_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: MultiTaskCriterion,
    device: torch.device
) -> Dict[str, Any]:
    """Evaluates validation performance: Accuracy, Macro-F1, Weighted-F1 for both heads."""
    model.eval()
    total_loss, total_g_loss, total_c_loss = 0.0, 0.0, 0.0
    num_samples = 0

    all_genre_preds, all_genre_targets = [], []
    all_century_preds, all_century_targets = [], []

    for images, genre_targets, century_targets, _ in tqdm(loader, desc="Validating", leave=False):
        images = images.to(device, non_blocking=True)
        genre_targets = genre_targets.to(device, non_blocking=True)
        century_targets = century_targets.to(device, non_blocking=True)

        g_logits, c_logits = model(images)
        loss, loss_g, loss_c = criterion(g_logits, c_logits, genre_targets, century_targets)

        batch_sz = images.size(0)
        total_loss += loss.item() * batch_sz
        total_g_loss += loss_g.item() * batch_sz
        total_c_loss += loss_c.item() * batch_sz
        num_samples += batch_sz

        g_preds = torch.argmax(g_logits, dim=-1).cpu().numpy()
        c_preds = torch.argmax(c_logits, dim=-1).cpu().numpy()

        all_genre_preds.extend(g_preds)
        all_genre_targets.extend(genre_targets.cpu().numpy())
        all_century_preds.extend(c_preds)
        all_century_targets.extend(century_targets.cpu().numpy())

    # Genre Metrics
    g_acc = accuracy_score(all_genre_targets, all_genre_preds)
    g_f1_macro = f1_score(all_genre_targets, all_genre_preds, average='macro', zero_division=0)
    g_f1_weighted = f1_score(all_genre_targets, all_genre_preds, average='weighted', zero_division=0)

    # Century Metrics
    c_acc = accuracy_score(all_century_targets, all_century_preds)
    c_f1_macro = f1_score(all_century_targets, all_century_preds, average='macro', zero_division=0)
    c_f1_weighted = f1_score(all_century_targets, all_century_preds, average='weighted', zero_division=0)

    # Combined score (Harmonic Mean of task Macro-F1s)
    harmonic_f1 = 2 * (g_f1_macro * c_f1_macro) / max(g_f1_macro + c_f1_macro, 1e-6)

    return {
        'val_loss': total_loss / num_samples,
        'val_genre_loss': total_g_loss / num_samples,
        'val_century_loss': total_c_loss / num_samples,
        'genre_acc': g_acc,
        'genre_f1_macro': g_f1_macro,
        'genre_f1_weighted': g_f1_weighted,
        'century_acc': c_acc,
        'century_f1_macro': c_f1_macro,
        'century_f1_weighted': c_f1_weighted,
        'harmonic_f1': harmonic_f1,
        'genre_preds': np.array(all_genre_preds),
        'genre_targets': np.array(all_genre_targets),
        'century_preds': np.array(all_century_preds),
        'century_targets': np.array(all_century_targets),
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and torch.cuda.is_available()
    print(f"================================================================")
    print(f" WikiArt Multi-Task Deep Learning Pipeline (Genre + Century)   ")
    print(f" Hardware Device: {device} | Mixed Precision (AMP): {use_amp} ")
    print(f" Backbone Architecture: {args.backbone}                        ")
    print(f"================================================================")

    # 1. Load, clean, and encode data
    print(f"\n[1/4] Preparing dataset from {args.csv_path}...")
    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file not found at {args.csv_path}. Please check path.")
        sys.exit(1)

    train_df, val_df, genre_enc, century_enc, genre_weights, century_weights = load_and_preprocess_wikiart(
        csv_path=args.csv_path,
        min_genre_samples=args.min_genre_samples
    )

    num_genres = len(genre_enc.classes_)
    num_centuries = len(century_enc.classes_)
    print(f"✓ Train Samples: {len(train_df)} | Validation Samples: {len(val_df)}")
    print(f"✓ Task 1: {num_genres} Clean Genre classes")
    print(f"✓ Task 2: {num_centuries} Clean Century classes: {list(century_enc.classes_)}")

    # Save class label dictionaries
    labels_meta = {
        'genres': list(genre_enc.classes_),
        'centuries': list(century_enc.classes_),
        'backbone': args.backbone,
        'img_size': args.img_size,
    }
    with open(os.path.join(args.output_dir, "classes_metadata.json"), "w") as f:
        json.dump(labels_meta, f, indent=2)

    # 2. Build DataLoaders
    train_loader, val_loader = build_dataloaders(
        train_df=train_df,
        val_df=val_df,
        base_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # 3. Instantiate Multi-Head Art Network
    print(f"\n[2/4] Building Multi-Head Network with backbone '{args.backbone}'...")
    model = MultiHeadArtNet(
        num_genres=num_genres,
        num_centuries=num_centuries,
        backbone_name=args.backbone,
        pretrained=True,
        dropout=args.dropout
    ).to(device)

    # Loss criterion with task class weights
    genre_weights = genre_weights.to(device)
    century_weights = century_weights.to(device)
    criterion = MultiTaskCriterion(
        genre_weight=args.genre_weight,
        century_weight=args.century_weight,
        genre_class_weights=genre_weights,
        century_class_weights=century_weights,
        use_focal=args.use_focal_loss
    )

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_harmonic_f1 = -1.0
    best_checkpoint_path = os.path.join(args.output_dir, "best_multi_head_artnet.pt")

    # =========================================================================
    # STAGE 1: Head Warm-up (Backbone Frozen)
    # =========================================================================
    if args.epochs_stage1 > 0:
        print(f"\n[3/4] Starting STAGE 1: Training Classification Heads ({args.epochs_stage1} Epochs, Backbone Frozen)...")
        model.freeze_backbone()

        head_params = list(model.genre_head.parameters()) + list(model.century_head.parameters())
        optimizer_stage1 = AdamW(head_params, lr=args.lr_stage1, weight_decay=args.weight_decay)

        for epoch in range(1, args.epochs_stage1 + 1):
            t0 = time.time()
            train_loss, tr_g_loss, tr_c_loss = train_one_epoch(
                model, train_loader, criterion, optimizer_stage1, scaler, device, use_amp
            )
            metrics = evaluate(model, val_loader, criterion, device)
            dur = time.time() - t0

            print(
                f"[Stage 1 | Epoch {epoch:02d}/{args.epochs_stage1:02d}] ({dur:.1f}s) "
                f"TrLoss: {train_loss:.4f} | ValLoss: {metrics['val_loss']:.4f} | "
                f"Genre Acc: {metrics['genre_acc']*100:.2f}% (F1-M: {metrics['genre_f1_macro']*100:.2f}%) | "
                f"Century Acc: {metrics['century_acc']*100:.2f}% (F1-M: {metrics['century_f1_macro']*100:.2f}%)"
            )

    # =========================================================================
    # STAGE 2: End-to-End Fine-Tuning (Differential Learning Rates)
    # =========================================================================
    print(f"\n[4/4] Starting STAGE 2: End-to-End Fine-Tuning ({args.epochs_stage2} Epochs with Cosine Annealing)...")
    # Unfreeze top stages of backbone
    model.unfreeze_top_stages(num_stages=2)

    # Differential learning rate: lower for backbone, higher for classification heads
    optimizer_stage2 = AdamW([
        {'params': model.backbone.parameters(), 'lr': args.lr_stage2_backbone},
        {'params': model.genre_head.parameters(), 'lr': args.lr_stage2_heads},
        {'params': model.century_head.parameters(), 'lr': args.lr_stage2_heads},
    ], weight_decay=args.weight_decay)

    scheduler = CosineAnnealingLR(optimizer_stage2, T_max=args.epochs_stage2, eta_min=1e-6)

    for epoch in range(1, args.epochs_stage2 + 1):
        t0 = time.time()
        train_loss, tr_g_loss, tr_c_loss = train_one_epoch(
            model, train_loader, criterion, optimizer_stage2, scaler, device, use_amp
        )
        metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        dur = time.time() - t0

        print(
            f"[Stage 2 | Epoch {epoch:02d}/{args.epochs_stage2:02d}] ({dur:.1f}s) "
            f"TrLoss: {train_loss:.4f} | ValLoss: {metrics['val_loss']:.4f} | "
            f"Genre Acc: {metrics['genre_acc']*100:.2f}% (F1-M: {metrics['genre_f1_macro']*100:.2f}%) | "
            f"Century Acc: {metrics['century_acc']*100:.2f}% (F1-M: {metrics['century_f1_macro']*100:.2f}%) | "
            f"Harmonic F1: {metrics['harmonic_f1']*100:.2f}%"
        )

        # Checkpoint if new best Harmonic F1
        if metrics['harmonic_f1'] > best_harmonic_f1:
            best_harmonic_f1 = metrics['harmonic_f1']
            print(f"  ★ New Best Combined F1: {best_harmonic_f1*100:.2f}%. Saving checkpoint to {best_checkpoint_path}...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer_stage2.state_dict(),
                'metrics': {k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)},
                'classes_meta': labels_meta,
                'config': vars(args)
            }, best_checkpoint_path)

    print(f"\n================================================================")
    print(f" Training Complete! Best Harmonic F1 Score: {best_harmonic_f1*100:.2f}%")
    print(f" Best Checkpoint Saved to: {best_checkpoint_path}")
    print(f" Metadata & Classes: {os.path.join(args.output_dir, 'classes_metadata.json')}")
    print(f"================================================================")


if __name__ == "__main__":
    main()
