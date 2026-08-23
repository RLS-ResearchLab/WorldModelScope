import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from src.utils.config import load_config
from src.utils.checkpoints import load_checkpoint
from datasets.dataloader import build_dataloader
from models.world_models.factory import build_model
from models.decoder.decoder_builder import EncoderDecoderModel


@torch.no_grad()
def evaluate(dino_wm_config, dino_wm_ckpt, decoder_config, decoder_ckpt, output_dir, num_samples, device):
    dino_wm = build_model(dino_wm_config).to(device).eval()
    load_checkpoint(dino_wm_ckpt, model=dino_wm, device=device)

    decoder_model = EncoderDecoderModel(decoder_config).to(device).eval()
    load_checkpoint(decoder_ckpt, model=decoder_model, device=device)
    decoder = decoder_model.decoder

    val_loader = build_dataloader(dino_wm_config, split="val")
    img_size = decoder_config["decoder"]["img_size"]
    num_hist = dino_wm.num_hist

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {"mse_pred_vs_gt": 0.0, "mse_pred_vs_encdec": 0.0, "cos_sim_latent": 0.0, "n": 0}
    saved = 0

    for raw_batch in val_loader:
        frames = raw_batch["frames"].to(device)     # [B, T, C, H, W]
        actions = raw_batch["actions"].to(device)    # [B, T-1, A]
        if frames.shape[1] < num_hist + 1:
            continue

        obs = frames[:, :num_hist + 1]
        acts = actions[:, :num_hist]
        acts_padded = F.pad(acts, (0, 0, 0, 1))  # match Trainer's zero-pad convention

        latents = dino_wm.encode_observations(obs)          # [B, H+1, P, D]
        context, target = latents[:, :-1], latents[:, 1:]    # [B, H, P, D] each
        predicted = dino_wm.predict(context, acts_padded[:, :num_hist])  # [B, H, P, D]

        B, H, P, D = predicted.shape
        gt_frames = F.interpolate(
            obs[:, 1:].reshape(B * H, *obs.shape[2:]), size=(img_size, img_size),
            mode="bilinear", align_corners=False,
        )
        recon_from_pred = decoder(predicted.reshape(B * H, P, D))
        recon_from_gt = decoder(target.reshape(B * H, P, D))

        metrics["mse_pred_vs_gt"] += F.mse_loss(recon_from_pred, gt_frames).item() * B
        metrics["mse_pred_vs_encdec"] += F.mse_loss(recon_from_pred, recon_from_gt).item() * B
        metrics["cos_sim_latent"] += F.cosine_similarity(predicted, target, dim=-1).mean().item() * B
        metrics["n"] += B

        if saved < num_samples:
            grid = torch.cat([gt_frames[:1], recon_from_gt[:1], recon_from_pred[:1]], dim=0)
            save_image(grid, output_dir / f"sample_{saved:03d}.png", nrow=3)
            saved += 1
        if saved >= num_samples and metrics["n"] >= num_samples * 8:
            break

    n = max(metrics["n"], 1)
    print(f"mse(predicted decode, ground-truth frame): {metrics['mse_pred_vs_gt'] / n:.6f}")
    print(f"mse(predicted decode, encoder decode):      {metrics['mse_pred_vs_encdec'] / n:.6f}")
    print(f"cosine sim(predicted latent, gt latent):     {metrics['cos_sim_latent'] / n:.4f}")
    print(f"Saved {saved} comparison grids (gt | encoder-decoded | predictor-decoded) to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dino_wm_config", required=True)
    parser.add_argument("--dino_wm_checkpoint", required=True)
    parser.add_argument("--decoder_config", required=True)
    parser.add_argument("--decoder_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    evaluate(
        load_config(args.dino_wm_config), args.dino_wm_checkpoint,
        load_config(args.decoder_config), args.decoder_checkpoint,
        args.output_dir, args.num_samples, args.device,
    )
