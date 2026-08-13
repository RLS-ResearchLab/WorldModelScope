"""
Train a decoder to invert frozen DINOv2 patch tokens back into RGB images,
using the BridgeData V2 IterableDataset / DataLoader.

This is a standalone autoencoding task:
    frozen DINOv2 encoder(image) -> patch tokens -> decoder -> reconstruction
supervised against the SAME input image. It does NOT depend on your
world-model predictor and can be trained independently, on the same
BridgeData frames you already use for DINO-WM.

Once trained, freeze the decoder and use it purely for visualization:
    predicted_tokens = world_model_predictor(...)
    predicted_image  = decoder(predicted_tokens)

pip install lpips  (perceptual loss; optional but strongly recommended
for sharper reconstructions than pure L1/L2)
"""

import os
import argparse
from itertools import islice

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from dino_decoder import DinoDecoder
from dinov2_encoder import DINOv2Encoder
from bridge_data.dataset import BridgeDataset, make_dataloader

try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False


def clip_batch_to_frames(batch, img_size, frame_subsample=1):
    """
    batch["frames"]: (B, T, C, H, W) float32 in [0, 1] from BridgeDataset.
    Flattens time into the batch dim, optionally subsamples frames within
    each clip (adjacent frames in a clip are highly redundant), and resizes
    to (img_size, img_size) if the native resolution differs.
    """
    frames = batch["frames"]  # (B, T, C, H, W)
    if frame_subsample > 1:
        frames = frames[:, ::frame_subsample]

    B, T, C, H, W = frames.shape
    frames = frames.reshape(B * T, C, H, W)

    if H != img_size or W != img_size:
        frames = F.interpolate(frames, size=(img_size, img_size), mode="bilinear", align_corners=False)

    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="raw/bridge_dataset/1.0.0",
                         help="Path passed to BridgeDataset (same as your DINO-WM data root).")
    parser.add_argument("--img_size", type=int, default=224,
                         help="Resolution the reconstruction TARGET is resized to. "
                              "Note: DINOv2Encoder always resizes its own input to 224 "
                              "internally regardless of this value, so 224 is recommended "
                              "unless you also edit dinov2_encoder.py.")
    parser.add_argument("--dino_model", type=str, default="dinov2_vits14",
                         choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"])
    parser.add_argument("--dino_normalize", action="store_true", default=True,
                         help="Apply ImageNet normalization before the DINOv2 encoder. "
                              "MUST match whatever your DINO-WM predictor was trained with.")
    parser.add_argument("--no_dino_normalize", dest="dino_normalize", action="store_false")
    parser.add_argument("--clip_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--frame_subsample", type=int, default=4,
                         help="Keep every Nth frame within a clip to reduce near-duplicate frames.")
    parser.add_argument("--camera_key", type=str, default="image_0")
    parser.add_argument("--batch_size", type=int, default=8,
                         help="Number of CLIPS per batch; actual images/step = batch_size * clip_len / frame_subsample.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps_per_epoch", type=int, default=500,
                         help="BridgeDataset is an IterableDataset (streamed, no fixed length) -> define epoch length explicitly.")
    parser.add_argument("--val_steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lpips_weight", type=float, default=0.5)
    parser.add_argument("--out_dir", type=str, default="./decoder_runs")
    parser.add_argument("--save_every", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------------------------------------------------------
    # 1. Frozen DINOv2 encoder -- use the SAME weights/checkpoint you
    #    used when training DINO-WM, so the token space matches exactly.
    # ---------------------------------------------------------------
    dinov2 = DINOv2Encoder(model_name=args.dino_model, freeze=True, normalize=args.dino_normalize)
    dinov2.to(device)

    embed_dim = dinov2.embed_dim
    grid_size = 224 // 14  # DINOv2Encoder always resizes to 224 internally, so this is fixed

    # ---------------------------------------------------------------
    # 2. Decoder (trainable)
    # ---------------------------------------------------------------
    decoder = DinoDecoder(embed_dim=embed_dim, grid_size=grid_size, img_size=args.img_size).to(device)

    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    lpips_fn = None
    if HAS_LPIPS:
        lpips_fn = lpips.LPIPS(net="vgg").to(device)
        for p in lpips_fn.parameters():
            p.requires_grad = False
    else:
        print("lpips not installed -> training with pure L1 loss only. "
              "`pip install lpips` for sharper reconstructions.")

    # ---------------------------------------------------------------
    # 3. Data -- reuse your existing BridgeDataset / make_dataloader
    # ---------------------------------------------------------------
    train_ds = BridgeDataset(
        data_dir=args.data_dir,
        split="train",
        clip_len=args.clip_len,
        stride=args.stride,
        camera_key=args.camera_key,
    )
    val_ds = BridgeDataset(
        data_dir=args.data_dir,
        split="val",
        clip_len=args.clip_len,
        stride=args.stride,
        camera_key=args.camera_key,
        shuffle=False,
    )
    train_loader = make_dataloader(train_ds, batch_size=args.batch_size,
                                    num_workers=args.num_workers, pin_memory=True)
    val_loader = make_dataloader(val_ds, batch_size=args.batch_size,
                                  num_workers=args.num_workers, pin_memory=True)

    def run_batches(loader, n_steps):
        """IterableDataset streams indefinitely -> take a fixed number of batches per epoch."""
        return islice(loader, n_steps)

    def compute_loss(images):
        tokens = dinov2(images)  # frozen, no grad (DINOv2Encoder.forward is @torch.no_grad())
        recon = decoder(tokens)
        l1_loss = F.l1_loss(recon, images)
        loss = l1_loss
        if lpips_fn is not None:
            perceptual = lpips_fn(recon * 2 - 1, images * 2 - 1).mean()
            loss = loss + args.lpips_weight * perceptual
        return loss, recon

    # ---------------------------------------------------------------
    # 4. Training loop
    # ---------------------------------------------------------------
    for epoch in range(args.epochs):
        decoder.train()
        running_loss = 0.0
        n_batches = 0

        for batch in run_batches(train_loader, args.steps_per_epoch):
            images = clip_batch_to_frames(batch, args.img_size, args.frame_subsample).to(device, non_blocking=True)

            loss, _ = compute_loss(images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = running_loss / max(n_batches, 1)
        print(f"[epoch {epoch+1}/{args.epochs}] train_loss={avg_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e}")

        # ------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------
        decoder.eval()
        val_loss = 0.0
        n_val = 0
        last_val_images = None
        with torch.no_grad():
            for batch in run_batches(val_loader, args.val_steps):
                images = clip_batch_to_frames(batch, args.img_size, args.frame_subsample).to(device)
                loss, recon = compute_loss(images)
                val_loss += loss.item()
                n_val += 1
                last_val_images = (images, recon)
        if n_val > 0:
            print(f"           val_loss={val_loss / n_val:.4f}")

        # ------------------------------------------------------------
        # Checkpoint + preview
        # ------------------------------------------------------------
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            ckpt_path = os.path.join(args.out_dir, f"decoder_epoch{epoch+1}.pt")
            torch.save(
                {"decoder_state_dict": decoder.state_dict(),
                 "embed_dim": embed_dim,
                 "grid_size": grid_size,
                 "img_size": args.img_size,
                 "dino_model": args.dino_model},
                ckpt_path,
            )
            if last_val_images is not None:
                images, recon = last_val_images
                n_show = min(8, images.shape[0])
                grid = torch.cat([images[:n_show], recon[:n_show]], dim=0)
                save_image(grid, os.path.join(args.out_dir, f"preview_epoch{epoch+1}.png"), nrow=n_show)
            print(f"  saved checkpoint + preview to {args.out_dir}")

    print("Done. Freeze this decoder and use it to visualize world-model predictions.")


if __name__ == "__main__":
    main()