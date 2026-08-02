# src/losses.py
import torch
import torch.nn.functional as F


def vjepa_loss(pred_tokens, target_tokens, loss_type="l1"):
    """
    pred_tokens:   [B, N_target, embed_dim]  -- predictor's output
    target_tokens: [B, N_target, embed_dim]  -- target encoder's real output at the same positions
    """
    # -- normalize target representations (standard JEPA practice: predict a normalized target,
    #    since raw encoder activations can have widely varying scale, making L1/L2 hard to balance)
    target_tokens = F.layer_norm(target_tokens, (target_tokens.size(-1),))

    if loss_type == "l1":
        loss = F.l1_loss(pred_tokens, target_tokens)
    elif loss_type == "smooth_l1":
        loss = F.smooth_l1_loss(pred_tokens, target_tokens)
    else:
        raise ValueError(f"unknown loss_type: {loss_type}")

    return loss


class EMA:
    """Maintains a target encoder as an exponential moving average of the context encoder's weights."""

    def __init__(self, context_encoder, target_encoder, decay=0.996):
        self.decay = decay
        self.context_encoder = context_encoder
        self.target_encoder = target_encoder

        # -- initialize target encoder as an exact copy, then freeze its gradients entirely
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self):
        for p_ctxt, p_target in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            p_target.data.mul_(self.decay).add_(p_ctxt.data, alpha=1 - self.decay)


def save_checkpoint(path, context_encoder, target_encoder, predictor, optimizer, epoch, step, loss_value):
    torch.save({
        "context_encoder": context_encoder.state_dict(),
        "target_encoder": target_encoder.state_dict(),
        "predictor": predictor.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "loss": loss_value,
    }, path)


def load_checkpoint(path, context_encoder, target_encoder, predictor, optimizer, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    context_encoder.load_state_dict(ckpt["context_encoder"])
    target_encoder.load_state_dict(ckpt["target_encoder"])
    predictor.load_state_dict(ckpt["predictor"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["epoch"], ckpt["step"]