# train.py
import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.models.vision_transformer import vit_small_rope
from src.models.predictor import vit_predictor
from src.masks.utils import MaskCollator, apply_masks

from src.utils import vjepa_loss
from src.utils import EMA
from src.utils import save_checkpoint, load_checkpoint
from datasets.synthetic import SyntheticVideoDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- config (kept small for fast iteration) ----------------
IMG_SIZE = 64
PATCH_SIZE = 8
NUM_FRAMES = 4
TUBELET_SIZE = 2
EMBED_DIM = 192
PRED_EMBED_DIM = 96
BATCH_SIZE = 8
EPOCHS = 3
LR = 1e-4

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
SAVE_EVERY_N_STEPS = 20
RESUME_FROM = None  # e.g. "checkpoints/latest.pt" to resume training

# ---------------- models ----------------
context_encoder = vit_small_rope(
    img_size=IMG_SIZE, patch_size=PATCH_SIZE, num_frames=NUM_FRAMES,
    tubelet_size=TUBELET_SIZE, embed_dim=EMBED_DIM,
).to(device)

target_encoder = vit_small_rope(
    img_size=IMG_SIZE, patch_size=PATCH_SIZE, num_frames=NUM_FRAMES,
    tubelet_size=TUBELET_SIZE, embed_dim=EMBED_DIM,
).to(device)

predictor = vit_predictor(
    img_size=IMG_SIZE, patch_size=PATCH_SIZE, num_frames=NUM_FRAMES,
    tubelet_size=TUBELET_SIZE, embed_dim=EMBED_DIM, predictor_embed_dim=PRED_EMBED_DIM,
).to(device)

ema = EMA(context_encoder, target_encoder, decay=0.996)

# ---------------- data ----------------
dataset = SyntheticVideoDataset(num_samples=200, num_frames=NUM_FRAMES, img_size=IMG_SIZE)
mask_collator = MaskCollator(
    input_size=(IMG_SIZE, IMG_SIZE), patch_size=PATCH_SIZE,
    enc_mask_scale=(0.4, 0.6), pred_mask_scale=(0.15, 0.2),
    nenc=1, npred=2, min_keep=4,
)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=mask_collator)

# -- only context encoder + predictor are trained; target encoder is EMA-updated, never backpropped
optimizer = torch.optim.AdamW(
    list(context_encoder.parameters()) + list(predictor.parameters()), lr=LR
)

# ---------------- optionally resume ----------------
start_epoch = 0
if RESUME_FROM is not None and os.path.exists(RESUME_FROM):
    start_epoch, _ = load_checkpoint(RESUME_FROM, context_encoder, target_encoder, predictor, optimizer, device=device)
    start_epoch += 1
    print(f"Resumed from {RESUME_FROM}, continuing at epoch {start_epoch}")

# ---------------- training loop ----------------
epoch_losses = []

for epoch in range(start_epoch, EPOCHS):
    running_loss = 0.0
    num_steps = 0

    for step, (video, masks_enc, masks_pred) in enumerate(loader):
        video = video.to(device)
        masks_enc = [m.to(device) for m in masks_enc]
        masks_pred = [m.to(device) for m in masks_pred]

        # -- target encoder sees the FULL video, no masking, no gradient
        with torch.no_grad():
            target_tokens_full = target_encoder(video)
            target_tokens_full = torch.cat([target_tokens_full] * len(masks_pred), dim=0)
            target_tokens = apply_masks(target_tokens_full, masks_pred)

        # -- context encoder sees only the context-masked tokens
        context_tokens = context_encoder(video, masks=masks_enc)

        if len(masks_pred) != len(masks_enc):
            context_tokens = context_tokens.repeat(len(masks_pred), 1, 1)

        masks_enc_cat = torch.cat(masks_enc * len(masks_pred), dim=0)
        masks_pred_cat = torch.cat(masks_pred, dim=0)

        pred_tokens = predictor(
            context_tokens, masks_ctxt=masks_enc_cat, masks_target=masks_pred_cat,
            T=NUM_FRAMES // TUBELET_SIZE, H_patches=IMG_SIZE // PATCH_SIZE, W_patches=IMG_SIZE // PATCH_SIZE,
        )

        loss = vjepa_loss(pred_tokens, target_tokens, loss_type="l1")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.update()

        running_loss += loss.item()
        num_steps += 1

        if step % 10 == 0:
            print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

        if step % SAVE_EVERY_N_STEPS == 0 and step > 0:
            save_checkpoint(
                os.path.join(CHECKPOINT_DIR, "latest.pt"),
                context_encoder, target_encoder, predictor, optimizer, epoch, step, loss.item(),
            )

    avg_epoch_loss = running_loss / num_steps
    epoch_losses.append(avg_epoch_loss)
    print(f"=== epoch {epoch} finished — avg loss: {avg_epoch_loss:.4f} ===")

    save_checkpoint(
        os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch{epoch}.pt"),
        context_encoder, target_encoder, predictor, optimizer, epoch, num_steps, avg_epoch_loss,
    )

print("Loss per epoch:", epoch_losses)

