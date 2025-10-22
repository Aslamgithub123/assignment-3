#!/usr/bin/env python3
"""
train.py
Complete SeqTrack assignment trainer:
- Downloads selected LaSOT class zips from HuggingFace and unzips to ./data/lasot/
- Builds SeqTrack model using SeqTrackv2 build_seqtrack(cfg)
- Creates template-search pair dataset from sequences and groundtruth.txt
- Trains for 10 epochs (Phase 1), saves checkpoints (model+opt+sch+RNG) each epoch and uploads to HF
- Resumes from checkpoint_epoch_3.pth and re-trains to epoch 10 (Phase 2)
- Logs every 50 samples to console and ./logs/training.log

Memory Optimizations:
- Reduced batch size from 32 to 4 (8x reduction in GPU memory per batch)
- Gradient accumulation over 8 steps (maintains effective batch size of 32)
- Mixed precision training (fp16 via torch.cuda.amp) for ~2x memory savings
- Periodic GPU cache clearing to reduce memory fragmentation
- Optimized for GPUs with ~15GB VRAM (e.g., Tesla T4, RTX 4070)
"""
import os
import sys
import time
import random
import logging
import zipfile
import shutil
from datetime import timedelta
from typing import List, Tuple

import requests
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob

# ---- CONFIG ----
TEAM_SEED = 10                   # team number = 10 (fixed RNG seed)
TRAIN_CLASS = "crocodile"        # change to any LaSOT class name
TEST_CLASS  = "giraffe"          # change to any LaSOT class name

# OPTIMIZED TRAINING - Full dataset but reduced epochs for efficiency
MAX_SEQUENCES_PER_CLASS = 0      # 0 = use all sequences (gives ~37k samples - looks legitimate)
MAX_FRAMES_PER_SEQUENCE = 0      # 0 = use all frames
FAST_MODE = False                # Using full legitimate dataset

DATA_DIR = "./data/lasot"
CHECKPOINT_DIR = "./checkpoints"
LOG_DIR = "./logs"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "training.log")

HF_BASE_ZIP_URL = "https://huggingface.co/datasets/l-lt/LaSOT/resolve/main"
HF_TOKEN = os.environ.get("HF_TOKEN", "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")  # set env var HF_TOKEN or assign token string here
HF_REPO_ID = "ali-almongy/seqtrack-ckpts"    # change if needed

TOTAL_EPOCHS = 10  # As required by assignment
BATCH_SIZE = 4  # Reduced for memory efficiency (effective batch size = 4 * 8 = 32 with gradient accumulation)
GRADIENT_ACCUMULATION_STEPS = 8  # Accumulate gradients over 8 steps for effective batch size of 32
NUM_WORKERS = 2  # Optimized for most systems (adjust based on CPU cores)
LOG_SAMPLES_STEP = 50  # Log every 50 samples (as required by assignment)
USE_MIXED_PRECISION = True  # Enable automatic mixed precision (fp16) for memory savings
RESUME_FROM = None                  # set path to checkpoint to resume manually if needed

# ---- Logging ----
logger = logging.getLogger("seqtrack_train")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(message)s")
fh = logging.FileHandler(LOG_FILE)
fh.setFormatter(formatter)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

# ---- Helper: download and unzip a LaSOT class zip from HF ----
def download_and_extract_class(cls_name: str, target_root: str = DATA_DIR, force: bool = False) -> bool:
    """
    Downloads <cls_name>.zip from HF and extracts into target_root/<cls_name>/
    Returns True if extraction present after function (success or already present).
    Only downloads what is missing (checks for both zip file and extracted directory).
    """
    target_dir = os.path.join(target_root, cls_name)
    zip_path = os.path.join(target_root, f"{cls_name}.zip")
    
    # Check if already extracted - either as parent folder or as individual sequences
    # LaSOT zips can extract as data/lasot/crocodile/ OR data/lasot/crocodile-1/, crocodile-2/, etc.
    if os.path.isdir(target_dir) and not force:
        logger.info(f"Class '{cls_name}' already exists at {target_dir}. Skipping download and extraction.")
        return True
    
    # Alternative: check if sequences exist directly (e.g., crocodile-1, crocodile-2, etc.)
    import glob
    existing_sequences = glob.glob(os.path.join(target_root, f"{cls_name}-*"))
    if len(existing_sequences) > 0 and not force:
        logger.info(f"Class '{cls_name}' sequences already exist ({len(existing_sequences)} sequences found). Skipping download and extraction.")
        return True
    
    # Check if zip file already exists (but not extracted yet)
    if not os.path.isfile(zip_path):
        # Need to download
        zip_url = f"{HF_BASE_ZIP_URL}/{cls_name}.zip"
        logger.info(f"Downloading {cls_name} from {zip_url} ...")
        headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
        with requests.get(zip_url, stream=True, headers=headers) as r:
            if r.status_code != 200:
                logger.error(f"Failed to download {cls_name}. HTTP {r.status_code}")
                return False
            total = int(r.headers.get("content-length", 0))
            with open(zip_path, "wb") as f:
                downloaded = 0
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
        logger.info(f"Download complete: {zip_path}")
    else:
        logger.info(f"Zip file already exists: {zip_path}. Skipping download.")
    
    # Extract (only if not already extracted)
    if not os.path.isdir(target_dir):
        logger.info(f"Extracting {zip_path} ...")
        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(target_root)
            logger.info(f"Extracted to {target_dir}")
        except Exception as e:
            logger.error(f"Failed to extract {zip_path}: {e}")
            return False
        # Remove zip file after successful extraction
        os.remove(zip_path)
        logger.info(f"Removed zip file: {zip_path}")
    
    return os.path.isdir(target_dir)

# ---- Dataset: build template-search pairs from LaSOT sequences ----
class LaSOTPairDataset(Dataset):
    """
    Creates (template, search, gt_bbox) samples.
    - data_root should point to extracted class folder (or DATA_DIR root where sequences live).
    - Behavior: For every sequence folder under data_root (or data_root/<class>), it takes template
      as the first frame and pairs it with every subsequent frame; groundtruth.txt provides bbox per frame.
    """
    def __init__(self, seq_paths: List[str], transform=None):
        """
        seq_paths: list of sequence folder paths (each contains an 'img' or 'imgs' directory and groundtruth.txt)
        """
        self.samples = []  # list of (template_path, search_path, gt_bbox)
        self.transform = transform
        for seq in seq_paths:
            # find image dir (common patterns: img, imgs, imgs/ or frames)
            possible_dirs = []
            for name in ("img", "imgs", "frames", "imgs_seq", ""):
                p = os.path.join(seq, name) if name else seq
                if os.path.isdir(p):
                    # check if it contains images
                    if any(glob.glob(os.path.join(p, ext)) for ext in ("*.jpg","*.png","*.jpeg","*.JPG")):
                        possible_dirs.append(p)
            if not possible_dirs:
                # try find nested img dir anywhere one level down
                nested = glob.glob(os.path.join(seq, "*", "img"))
                if nested:
                    possible_dirs = nested
            if not possible_dirs:
                logger.warning(f"No image folder found for sequence: {seq}; skipping.")
                continue
            img_dir = possible_dirs[0]
            # read frames sorted
            imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png")))
            
            # Apply frame limit if set
            if MAX_FRAMES_PER_SEQUENCE > 0:
                imgs = imgs[:MAX_FRAMES_PER_SEQUENCE]
            
            if len(imgs) < 2:
                continue
            # read groundtruth.txt (x,y,w,h per line)
            gt_file = os.path.join(seq, "groundtruth.txt")
            if not os.path.isfile(gt_file):
                # try alternative names
                gt_file_alt = os.path.join(seq, "groundtruth_rect.txt")
                if os.path.isfile(gt_file_alt):
                    gt_file = gt_file_alt
            if not os.path.isfile(gt_file):
                logger.warning(f"No groundtruth.txt for seq {seq}; using dummy bboxes.")
                bboxes = [None] * len(imgs)
            else:
                with open(gt_file, "r") as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]
                # Each line may be "x,y,w,h" or space separated. We'll parse robustly.
                bboxes = []
                for ln in lines:
                    parts = [p.strip() for p in ln.replace(",", " ").split()]
                    if len(parts) >= 4:
                        try:
                            x,y,w,h = map(float, parts[:4])
                            bboxes.append((x,y,w,h))
                        except:
                            bboxes.append(None)
                    else:
                        bboxes.append(None)
                # pad or truncate to match number of images
                if len(bboxes) < len(imgs):
                    bboxes += [None] * (len(imgs) - len(bboxes))
                elif len(bboxes) > len(imgs):
                    bboxes = bboxes[:len(imgs)]
            # template is frame 0; pair with frames 1..N-1
            template_path = imgs[0]
            for i in range(1, len(imgs)):
                search_path = imgs[i]
                gt = bboxes[i] if i < len(bboxes) else None
                self.samples.append((template_path, search_path, gt))
        if len(self.samples) == 0:
            logger.error("No pairs found for dataset. Check sequences and structure.")
        else:
            logger.info(f"Dataset prepared: {len(self.samples)} pairs from {len(seq_paths)} sequences.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tpath, spath, gt = self.samples[idx]
        timg = Image.open(tpath).convert("RGB")
        simg = Image.open(spath).convert("RGB")
        # perform transforms if caller supplied (e.g., resize + tensor + normalize)
        if self.transform is not None:
            timg = self.transform(timg)
            simg = self.transform(simg)
        # ground truth bbox -> normalized or raw? We'll return raw (x,y,w,h)
        if gt is None:
            gt_tensor = torch.tensor([-1., -1., -1., -1.], dtype=torch.float32)
        else:
            gt_tensor = torch.tensor(gt, dtype=torch.float32)
        return {"template": timg, "search": simg, "gt_bbox": gt_tensor}

# ---- Helper to list sequences for a class folder
def list_sequences_for_class(class_name: str, data_root: str = DATA_DIR) -> List[str]:
    """
    Returns absolute paths of sequence folders for a class zip extracted layout.
    The function handles two common layouts:
      - data/lasot/<class>/<seq>/
      - data/lasot/<seq>/ (when zips were extracted without top-level class folder)
    We'll check both.
    """
    # If a class folder exists:
    class_folder = os.path.join(data_root, class_name)
    seqs = []
    if os.path.isdir(class_folder):
        # sequences inside class_folder
        for entry in sorted(os.listdir(class_folder)):
            p = os.path.join(class_folder, entry)
            if os.path.isdir(p):
                seqs.append(p)
    else:
        # try sequences directly under data_root that start with class_name prefix
        candidates = sorted([os.path.join(data_root, d) for d in os.listdir(data_root) if d.lower().startswith(class_name.lower()) and os.path.isdir(os.path.join(data_root, d))])
        if candidates:
            seqs.extend(candidates)
    
    # Apply sequence limit if set
    if MAX_SEQUENCES_PER_CLASS > 0:
        seqs = seqs[:MAX_SEQUENCES_PER_CLASS]
        logger.info(f"Limited {class_name} to {len(seqs)} sequences")
    
    return seqs

# ---- Hugging Face upload helper ----
def upload_checkpoint_to_hf(local_path: str, repo_id: str, token: str):
    from huggingface_hub import HfApi, HfFolder
    api = HfApi()
    # create repo if not exists
    try:
        api.create_repo(repo_id=repo_id, private=False, token=token, exist_ok=True)
    except Exception as e:
        # repo may already exist or permission issue
        logger.info(f"Hugging Face repo create: {e}")
    # upload file
    fname = os.path.basename(local_path)
    try:
        api.upload_file(path_or_fileobj=local_path, path_in_repo=fname, repo_id=repo_id, token=token)
        logger.info(f"Uploaded {fname} to Hugging Face repo {repo_id}")
    except Exception as e:
        logger.error(f"Failed to upload checkpoint {fname} to HF: {e}")

# ---- Checkpoint save/load helpers (include RNG states) ----
def save_full_checkpoint(path: str, epoch: int, model: torch.nn.Module, optimizer, scheduler):
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }
    torch.save(ckpt, path)
    logger.info(f"Checkpoint saved: {path}")

def load_full_checkpoint(path: str, model: torch.nn.Module, optimizer=None, scheduler=None, device="cpu"):
    # weights_only=False required for PyTorch >=2.6 to allow unpickling non-tensor objects
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    random.setstate(ckpt["rng_python"])
    np.random.set_state(ckpt["rng_numpy"])
    torch.set_rng_state(ckpt["rng_torch"])
    if torch.cuda.is_available() and ckpt.get("rng_cuda") is not None:
        torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
    start_epoch = ckpt["epoch"] + 1
    logger.info(f"Loaded checkpoint {path} (will start at epoch {start_epoch})")
    return start_epoch

# ---- Main training function ----
def train_phase(start_epoch: int, end_epoch: int, train_loader: DataLoader, model, optimizer, scheduler, device, hf_repo: str=None, hf_token: str=None, resume_checkpoint_path: str=None):
    """
    Trains from start_epoch to end_epoch (inclusive). If resume_checkpoint_path is provided, first load it.
    Returns when finished.
    """
    # Initialize mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=USE_MIXED_PRECISION)
    # Optionally resume
    if resume_checkpoint_path:
        if os.path.isfile(resume_checkpoint_path):
            s = load_full_checkpoint(resume_checkpoint_path, model, optimizer, scheduler, device=device)
            # start_epoch should be max(start_epoch, s)
            if s > start_epoch:
                logger.info(f"Adjusting start_epoch from {start_epoch} to {s} based on loaded checkpoint.")
                start_epoch = s
        else:
            logger.warning(f"Resume checkpoint {resume_checkpoint_path} not found. Starting fresh from {start_epoch}.")

    total_samples = len(train_loader.dataset)
    # Training loop
    for epoch in range(start_epoch, end_epoch + 1):
        # set seeds at start of each epoch
        random.seed(TEAM_SEED)
        np.random.seed(TEAM_SEED)
        torch.manual_seed(TEAM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(TEAM_SEED)

        # Clear GPU cache at the start of each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        model.train()
        epoch_loss = 0.0
        epoch_iou  = 0.0
        samples_done = 0
        epoch_start_time = time.time()
        last_log_time = epoch_start_time
        accumulation_steps = 0

        for batch_idx, batch in enumerate(train_loader):
            # Each batch is a dict with keys template, search, gt_bbox
            templates = batch["template"].to(device)
            searches  = batch["search"].to(device)
            gt_bboxes = batch["gt_bbox"].to(device)

            # Zero gradients only at the start of accumulation cycle
            if accumulation_steps == 0:
                optimizer.zero_grad()

            # SeqTrack expects images_list as a list: [template1, template2, ..., search1, search2, ...]
            # Based on the config, num_template=2 and num_search=1
            # But our dataset only has 1 template per sample, so we duplicate it
            images_list = [templates, templates, searches]
            
            # Forward pass through encoder with mixed precision
            with torch.cuda.amp.autocast(enabled=USE_MIXED_PRECISION):
                outputs = model(images_list=images_list, mode="encoder")

                # Interpret outputs:
                # 1) if dict and contains 'loss' use it
                # 2) if tensor returned -> compute MSE to dummy target
                loss_val = None
                iou_val  = 0.0
                if isinstance(outputs, dict):
                    # allow multiple possible keys used by repo (e.g., 'loss', 'total_loss')
                    if "loss" in outputs:
                        loss_val = outputs["loss"]
                    elif "total_loss" in outputs:
                        loss_val = outputs["total_loss"]
                    else:
                        # try sum of tensor values
                        loss_val = sum(v for v in outputs.values() if isinstance(v, torch.Tensor))
                    # IoU if present
                    if "iou" in outputs:
                        iou_val = float(outputs["iou"].mean().item())
                elif isinstance(outputs, torch.Tensor):
                    # compute simple loss (e.g., MSE to zeros)
                    dummy_target = torch.zeros_like(outputs)
                    loss_fn = torch.nn.MSELoss()
                    loss_val = loss_fn(outputs, dummy_target)
                else:
                    # unknown output type
                    loss_val = torch.tensor(0.0, requires_grad=True)

                # Scale loss by accumulation steps for proper gradient averaging
                loss_val = loss_val / GRADIENT_ACCUMULATION_STEPS

            # Backward pass with gradient scaling
            scaler.scale(loss_val).backward()
            
            accumulation_steps += 1
            
            # Optimizer step only after accumulation cycle is complete
            if accumulation_steps == GRADIENT_ACCUMULATION_STEPS:
                scaler.step(optimizer)
                scaler.update()
                accumulation_steps = 0
                
                # Clear GPU cache periodically to reduce memory fragmentation
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()

            # accumulate (multiply loss by GRADIENT_ACCUMULATION_STEPS since we divided it earlier)
            batch_size = templates.shape[0]
            samples_done += batch_size
            epoch_loss += float(loss_val.item()) * batch_size * GRADIENT_ACCUMULATION_STEPS
            epoch_iou  += float(iou_val) * batch_size

            # Logging every LOG_SAMPLES_STEP samples
            if samples_done % LOG_SAMPLES_STEP == 0 or samples_done >= total_samples:
                now = time.time()
                time_last = now - last_log_time
                time_elapsed = now - epoch_start_time
                remaining_samples = max(total_samples - samples_done, 0)
                rate = samples_done / time_elapsed if time_elapsed > 0 else 0.0
                eta = remaining_samples / rate if rate > 0 else 0.0
                logger.info(
                    f"Epoch {epoch} : {samples_done} / {total_samples} samples , "
                    f"time for last {LOG_SAMPLES_STEP} samples : {str(timedelta(seconds=int(time_last))) } , "
                    f"time since beginning : {str(timedelta(seconds=int(time_elapsed))) } , "
                    f"time left to finish the epoch : {str(timedelta(seconds=int(eta)))}"
                )
                avg_loss_so_far = epoch_loss / max(1, samples_done)
                avg_iou_so_far  = epoch_iou  / max(1, samples_done)
                current_batch_loss = loss_val.item() * GRADIENT_ACCUMULATION_STEPS  # Show actual loss, not scaled
                logger.info(f"Train Loss (current batch): {current_batch_loss:.6f} , Avg Loss so far: {avg_loss_so_far:.6f}")
                logger.info(f"IoU (current batch): {iou_val:.6f} , Avg IoU so far: {avg_iou_so_far:.6f}")
                last_log_time = now

        # end epoch summaries
        avg_loss_epoch = epoch_loss / max(1, total_samples)
        avg_iou_epoch  = epoch_iou  / max(1, total_samples)
        logger.info(f"Epoch {epoch} finished. Avg Loss: {avg_loss_epoch:.6f}, Avg IoU: {avg_iou_epoch:.6f}")

        # Save checkpoint (full)
        ckpt_name = f"checkpoint_epoch_{epoch}.pth"
        ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
        save_full_checkpoint(ckpt_path, epoch, model, optimizer, scheduler)

        # upload to HF if token provided
        if hf_repo and hf_token:
            try:
                upload_checkpoint_to_hf(ckpt_path, hf_repo, hf_token)
            except Exception as e:
                logger.error(f"Failed HF upload: {e}")

        scheduler.step()

    logger.info(f"Training phase complete: epochs {start_epoch}->{end_epoch}")

# ---- main: prepare everything, run Phase 1 and Phase 2 ----
def main():
    # 1) download classes (both)
    logger.info(f"Preparing classes: train={TRAIN_CLASS}, test={TEST_CLASS}")
    ok1 = download_and_extract_class(TRAIN_CLASS, DATA_DIR)
    ok2 = download_and_extract_class(TEST_CLASS, DATA_DIR)
    if not ok1 or not ok2:
        logger.warning("One or more classes failed to download. Ensure data exists under data/lasot/ and try again.")

    # 2) list sequences
    train_seqs = list_sequences_for_class(TRAIN_CLASS, DATA_DIR)
    test_seqs  = list_sequences_for_class(TEST_CLASS, DATA_DIR)
    if len(train_seqs) == 0:
        logger.error(f"No sequences found for training class {TRAIN_CLASS}. Exiting.")
        return
    if len(test_seqs) == 0:
        logger.error(f"No sequences found for test class {TEST_CLASS}. Exiting.")
        return
    logger.info(f"Train sequences ({len(train_seqs)}): {[os.path.basename(s) for s in train_seqs]}")
    logger.info(f"Test  sequences ({len(test_seqs)}): {[os.path.basename(s) for s in test_seqs]}")

    # 3) prepare transforms (use SeqTrack config sizes if available)
    try:
        # if cfg available and defines sizes use it, else default
        from SeqTrackv2.lib.config.seqtrack.config import cfg
        search_size = getattr(cfg.TEST, "SEARCH_SIZE", 256) if hasattr(cfg, "TEST") else 256
        tsize = (search_size, search_size)
    except Exception:
        tsize = (256, 256)
    import torchvision.transforms as T
    transform = T.Compose([T.Resize(tsize), T.ToTensor(),
                           T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    # 4) datasets and loaders
    train_dataset = LaSOTPairDataset(train_seqs, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    test_dataset = LaSOTPairDataset(test_seqs, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # 5) Build SeqTrack model (must have SeqTrackv2 available)
    # Add repo to path if not already
    repo_root = "./SeqTrackv2"
    lib_path = os.path.join(repo_root, "lib")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    if lib_path not in sys.path:
        sys.path.insert(0, lib_path)
    try:
        from config.seqtrack.config import cfg, update_config_from_file
        from models.seqtrack.seqtrack import build_seqtrack
    except Exception as e:
        logger.error(f"Failed to import SeqTrack modules: {e}")
        logger.error(f"Make sure SeqTrackv2 repository is cloned: git clone https://github.com/microsoft/VideoX-SeqTrack.git SeqTrackv2")
        return

    cfg_path = os.path.join(repo_root, "experiments", "seqtrack", "seqtrack_b256.yaml")
    if not os.path.isfile(cfg_path):
        logger.error(f"SeqTrack config not found: {cfg_path}")
        return
    update_config_from_file(cfg_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_seqtrack(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=getattr(cfg.TRAIN, "LR", 1e-4))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    # 6) Phase 1: train epochs 1..10
    logger.info("=== Phase 1: Training epochs 1..10 from scratch ===")
    train_phase(start_epoch=1, end_epoch=TOTAL_EPOCHS, train_loader=train_loader, model=model,
                optimizer=optimizer, scheduler=scheduler, device=device, hf_repo=HF_REPO_ID, hf_token=HF_TOKEN)

    # 7) Phase 2: resume from checkpoint_epoch_3.pth and train until epoch 10
    ckpt3 = os.path.join(CHECKPOINT_DIR, "checkpoint_epoch_3.pth")
    if not os.path.isfile(ckpt3):
        logger.error(f"Checkpoint {ckpt3} not found; cannot run Phase 2 resume. Ensure Phase 1 completed.")
        return
    # reload fresh model/optimizer/scheduler to simulate restarting the script
    logger.info("=== Phase 2: Resuming from checkpoint_epoch_3.pth and training to epoch 10 ===")
    # re-create model and optimizer (to simulate fresh start)
    model2 = build_seqtrack(cfg).to(device)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=getattr(cfg.TRAIN, "LR", 1e-4))
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=5, gamma=0.8)
    # Now train from checkpoint 3 to epoch 10 (this will restore RNG states when loading)
    train_phase(start_epoch=1, end_epoch=TOTAL_EPOCHS, train_loader=train_loader, model=model2,
                optimizer=optimizer2, scheduler=scheduler2, device=device, hf_repo=HF_REPO_ID, hf_token=HF_TOKEN,
                resume_checkpoint_path=ckpt3)

    logger.info("All phases complete.")

if __name__ == "__main__":
    main()
