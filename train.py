"""
PalmNet training script.

Run from inside palm_net/:
    python train.py --dataset tongji --id_num 600
    python train.py --dataset iitd   --id_num 460
    python train.py --dataset polyu  --id_num 378

Or with explicit paths:
    python train.py --train_file ../CCNet/data/train_Tongji.txt \\
                    --test_file  ../CCNet/data/test_Tongji.txt  \\
                    --id_num 600
"""

import argparse
import copy
import os
import sys
import time
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PalmDataset
from loss    import SupConLoss
from model   import PalmNet
from utils   import get_file_names, save_loss_acc

PALM_NET_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PALM_NET_DIR.parent
DATA_DIR     = PROJECT_ROOT / "VT-PalmNet"

DATASET_FILES = {
    "tongji": ("train_Tongji.txt", "test_Tongji.txt"),
    "iitd":   ("train_IITD.txt",   "test_IITD.txt"),
    "polyu":  ("train_PolyU.txt",  "test_PolyU.txt"),
}

def extract_features(
    model: PalmNet, loader: DataLoader, device: torch.device, desc: str = "extracting"
) -> tuple[np.ndarray, np.ndarray]:
    feats, ids = [], []
    model.eval()
    with torch.no_grad():
        for datas, target in tqdm(loader, desc=f"  {desc}", leave=False, unit="batch"):
            codes = model.getFeatureCode(datas[0].to(device)).cpu().numpy()
            feats.append(codes)
            ids.append(target.numpy())
    return np.concatenate(feats), np.concatenate(ids)


def quick_rank1(
    model: PalmNet,
    train_file: str,
    test_file: str,
    device: torch.device,
) -> float:
    """Fast rank-1 accuracy using batched matrix ops. No file I/O, no external scripts."""
    train_ds = PalmDataset(train_file, train=False)
    test_ds  = PalmDataset(test_file,  train=False)
    feat_train, id_train = extract_features(model, DataLoader(train_ds, batch_size=512, num_workers=2), device, "gallery")
    feat_test,  id_test  = extract_features(model, DataLoader(test_ds,  batch_size=512, num_workers=2), device, "probe")
    # features are unit vectors → dot product = cosine similarity
    sim  = feat_test @ feat_train.T          # (n_test, n_train)
    pred = id_train[np.argmax(sim, axis=1)]
    rank1 = float(np.mean(pred == id_test) * 100)
    print(f"  [quick] Rank-1: {rank1:.3f}%")
    return rank1


def evaluate(
    model: PalmNet,
    train_file: str,
    test_file: str,
    rst_dir: Path,
    device: torch.device,
):
    print("\n── Evaluation ──────────────────────────────")
    verieer_dir = rst_dir / "veriEER"
    rank1_dir   = verieer_dir / "rank1_hard"
    verieer_dir.mkdir(parents=True, exist_ok=True)
    rank1_dir.mkdir(exist_ok=True)

    train_ds = PalmDataset(train_file, train=False)
    test_ds  = PalmDataset(test_file,  train=False)
    train_loader = DataLoader(train_ds, batch_size=512, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=512, num_workers=2)

    feat_train, id_train = extract_features(model, train_loader, device, "gallery feats")
    feat_test,  id_test  = extract_features(model, test_loader,  device, "probe  feats")

    print(f"  gallery: {feat_train.shape}   probe: {feat_test.shape}")

    n_test, n_train = len(feat_test), len(feat_train)

    # ── Verification EER (probe vs gallery) ──────────────────────────────
    scores, labels = [], []
    for i in tqdm(range(n_test), desc="  matching (veriEER)", leave=False):
        for j in range(n_train):
            d = np.arccos(np.clip(np.dot(feat_test[i], feat_train[j]), -1, 1)) / np.pi
            scores.append(d)
            labels.append(1 if id_test[i] == id_train[j] else -1)

    score_path = verieer_dir / "scores_VeriEER.txt"
    with open(score_path, "w") as f:
        for s, l in zip(scores, labels):
            f.write(f"{s} {l}\n")

    # ── Rank-1 accuracy ───────────────────────────────────────────────────
    scores_arr  = np.array(scores).reshape(n_test, n_train)
    train_paths = get_file_names(train_file)
    test_paths  = get_file_names(test_file)
    corr = 0
    for i in range(n_test):
        best_j = int(np.argmin(scores_arr[i]))
        if id_test[i] == id_train[best_j]:
            corr += 1
        else:
            try:
                im_t  = cv.imread(test_paths[i])
                im_tr = cv.imread(train_paths[best_j])
                if im_t is not None and im_tr is not None:
                    cv.imwrite(
                        str(rank1_dir / f"{scores_arr[i, best_j]:.4f}_{i}_{best_j}.png"),
                        np.concatenate([im_t, im_tr], axis=1),
                    )
            except Exception:
                pass

    rank1 = corr / n_test * 100
    print(f"  Rank-1 accuracy : {rank1:.3f}%")
    with open(verieer_dir / "rank1.txt", "w") as f:
        f.write(f"rank-1 acc: {rank1:.3f}%\n")

    # ── Test-test EER (all pairs within probe set) ────────────────────────
    s2, l2 = [], []
    for i in tqdm(range(n_test - 1), desc="  matching (testEER)", leave=False):
        for j in range(i + 1, n_test):
            d = np.arccos(np.clip(np.dot(feat_test[i], feat_test[j]), -1, 1)) / np.pi
            s2.append(d)
            l2.append(1 if id_test[i] == id_test[j] else -1)

    score2_path = verieer_dir / "scores_EER_test.txt"
    with open(score2_path, "w") as f:
        for s, l in zip(s2, l2):
            f.write(f"{s} {l}\n")

    # ── Local evaluation scripts ──────────────────────────────────────────
    _run_eval("getGI.py",  score_path,  "scores_VeriEER")
    _run_eval("getEER.py", score_path,  "scores_VeriEER")
    _run_eval("getGI.py",  score2_path, "scores_EER_test")
    _run_eval("getEER.py", score2_path, "scores_EER_test")

    sys.stdout.flush()
    print("────────────────────────────────────────────\n")
    return rank1


def _run_eval(script: str, score_path: Path, tag: str):
    """Run a local eval script from palm_net/."""
    os.system(f"cd {PALM_NET_DIR} && python {script} {score_path} {tag}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(
    model: PalmNet,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    con_criterion: SupConLoss,
    optimizer: optim.Optimizer,
    w_ce: float,
    w_con: float,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> tuple[float, float]:
    model.train()
    running_loss, running_correct, total = 0.0, 0, 0

    bar = tqdm(
        loader,
        desc=f"Epoch {epoch:4d}/{total_epochs}",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    )
    for datas, target in bar:
        img1   = datas[0].to(device)
        img2   = datas[1].to(device)
        target = target.to(device)

        optimizer.zero_grad()

        logits1, emb1 = model(img1, target)
        _,       emb2 = model(img2, target)

        fe       = torch.stack([emb1, emb2], dim=1)   # (B, 2, embed_dim)
        ce_loss  = criterion(logits1, target)
        con_loss = con_criterion(fe, target)
        loss     = w_ce * ce_loss + w_con * con_loss

        loss.backward()
        optimizer.step()

        bs               = target.size(0)
        running_loss    += loss.item() * bs
        running_correct += logits1.argmax(1).eq(target).sum().item()
        total           += bs

        bar.set_postfix(
            loss=f"{running_loss / total:.4f}",
            acc=f"{100.0 * running_correct / total:.1f}%",
        )

    return running_loss / total, 100.0 * running_correct / total

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PalmNet")

    p.add_argument("--dataset",    type=str, default="tongji",
                   choices=list(DATASET_FILES.keys()))
    p.add_argument("--train_file", type=str, default=None)
    p.add_argument("--test_file",  type=str, default=None)
    p.add_argument("--id_num",     type=int, default=600,
                   help="Tongji=600  IITD=460  PolyU=378")

    p.add_argument("--embed_dim",     type=int,   default=512)
    p.add_argument("--ppu_channels", type=int,   default=32,
                   help="PPU output channels per CB order. feat_dim=411×ppu_channels. "
                        "32→~57M params  16→~14M  8→~1.7M")
    p.add_argument("--fc_hidden",    type=int,   default=4096,
                   help="FC1 hidden width (largest layer). Reduce alongside ppu_channels.")
    p.add_argument("--arc_s",        type=float, default=30.0)
    p.add_argument("--arc_m",        type=float, default=0.5)

    p.add_argument("--epochs",     type=int,   default=3000)
    p.add_argument("--batch_size", type=int,   default=512)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--lr_step",    type=int,   default=500)
    p.add_argument("--lr_gamma",   type=float, default=0.8)
    p.add_argument("--temp",       type=float, default=0.07)
    
    p.add_argument("--w_ce",       type=float, default=0.8)
    p.add_argument("--w_con",      type=float, default=0.2)

    p.add_argument("--gpu",            type=str, default="0")
    p.add_argument("--checkpoint_dir", type=str, default="./results/checkpoint/")
    p.add_argument("--rst_dir",        type=str, default="./results/rst_test/")

    p.add_argument("--eval_interval",       type=int, default=1000,
                   help="Interval for full evaluation (EER + rank-1 + scripts)")
    p.add_argument("--quick_eval_interval", type=int, default=100,
                   help="Interval for fast rank-1 check used by early stopping")
    p.add_argument("--save_interval",       type=int, default=500)
    p.add_argument("--patience",            type=int, default=None,
                   help="Stop after this many quick evals with no rank-1 improvement. Disabled if not set.")

    return p.parse_args()

def main():
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")
    if args.train_file and args.test_file:
        train_file, test_file = args.train_file, args.test_file
    else:
        tr, te     = DATASET_FILES[args.dataset]
        train_file = str(DATA_DIR / tr)
        test_file  = str(DATA_DIR / te)

    ckpt_dir = Path(args.checkpoint_dir)
    rst_dir  = Path(args.rst_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    rst_dir.mkdir(parents=True, exist_ok=True)

    train_ds = PalmDataset(train_file, train=True)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=(device.type == "cuda"),
    )

    net      = PalmNet(args.id_num, embed_dim=args.embed_dim,
                       ppu_channels=args.ppu_channels, fc_hidden=args.fc_hidden,
                       s=args.arc_s, m=args.arc_m)
    best_net = copy.deepcopy(net)
    net.to(device)

    total_params = sum(p.numel() for p in net.parameters())

    criterion     = nn.CrossEntropyLoss()
    con_criterion = SupConLoss(temperature=args.temp, base_temperature=args.temp)
    optimizer     = optim.Adam(net.parameters(), lr=args.lr)
    scheduler     = lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    print(f"\n{'='*55}")
    print(f"  VisionTech PalmNet training")
    print(f"  Device     : {device}")
    print(f"  Dataset    : {args.dataset}  ({args.id_num} classes)")
    print(f"  Params     : {total_params:,}")
    print(f"  Embed dim  : {args.embed_dim}  |  PPU ch: {args.ppu_channels}  |  FC hidden: {args.fc_hidden}")
    print(f"  Epochs     : {args.epochs}  |  Batch: {args.batch_size}")
    print(f"  LR         : {args.lr}  step {args.lr_step}×{args.lr_gamma}")
    print(f"  Losses     : CE×{args.w_ce} + SupCon×{args.w_con}  (T={args.temp})")
    print(f"  Patience   : {args.patience if args.patience is not None else 'disabled'}  (quick eval every {args.quick_eval_interval} epochs)")
    print(f"  Started    : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*55}\n")

    train_losses, train_accs = [], []
    best_train_acc  = 0.0
    best_eval_acc   = 0.0
    patience_counter = 0
    stopped_early   = False

    epoch_bar = tqdm(range(args.epochs), desc="training", unit="epoch", dynamic_ncols=True)

    for epoch in epoch_bar:
        loss, acc = train_epoch(
            net, train_loader, criterion, con_criterion,
            optimizer, args.w_ce, args.w_con, device, epoch, args.epochs,
        )
        scheduler.step()

        train_losses.append(loss)
        train_accs.append(acc)
        if acc > best_train_acc:
            best_train_acc = acc

        epoch_bar.set_postfix(
            loss=f"{loss:.4f}",
            acc=f"{acc:.1f}%",
            best_eval=f"{best_eval_acc:.2f}%",
            lr=f"{scheduler.get_last_lr()[0]:.2e}",
        )

        if epoch % args.save_interval == 0 or epoch == args.epochs - 1:
            torch.save(net.state_dict(), ckpt_dir / "net_params.pth")
            torch.save(net.state_dict(), ckpt_dir / f"epoch_{epoch}_net_params.pth")
            save_loss_acc(train_losses, train_accs, best_train_acc, str(rst_dir))

        if epoch % args.quick_eval_interval == 0 and epoch != 0:
            eval_acc = quick_rank1(net, train_file, test_file, device)

            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                patience_counter = 0
                torch.save(net.state_dict(), ckpt_dir / "net_params_best.pth")
                best_net = copy.deepcopy(net)
                print(f"  New best rank-1: {best_eval_acc:.3f}%  → checkpoint saved")
            elif args.patience is not None:
                patience_counter += 1
                print(f"  No improvement ({eval_acc:.3f}% vs best {best_eval_acc:.3f}%). "
                      f"Patience: {patience_counter}/{args.patience}")
                if patience_counter >= args.patience:
                    print(f"\nEarly stopping triggered after {patience_counter} quick evals without improvement.")
                    stopped_early = True
                    break

        if epoch % args.eval_interval == 0 and epoch != 0:
            evaluate(net, train_file, test_file, rst_dir, device)

    if stopped_early:
        print(f"\nStopped early at epoch {epoch}.  Best eval rank-1: {best_eval_acc:.3f}%")
    else:
        print(f"\nTraining done.  Best train acc: {best_train_acc:.2f}%  Best eval rank-1: {best_eval_acc:.3f}%")
        print("\n── Final evaluation (last model) ──")
        evaluate(net, train_file, test_file, rst_dir, device)

    print("\n── Final evaluation (best model) ──")
    best_net.to(device)
    evaluate(best_net, train_file, test_file, rst_dir, device)

    print(f"\nBest weights → {ckpt_dir / 'net_params_best.pth'}")


if __name__ == "__main__":
    main()
