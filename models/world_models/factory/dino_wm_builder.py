import torch

from models.world_models.dino_wm import DINOWM

from models.encoders.action_encoder import DINOWMActionEncoder
from models.encoders.feature_adapter import FeatureAdapter
from models.predictors.dino_predictor import ViTPredictor
from models.encoders.dinov2 import DINOv2Encoder
from models.encoders.EUPE_encoder import EUPEEncoder

ENCODER_REGISTRY = {
    "dinov2": DINOv2Encoder,
    "eupe": EUPEEncoder,
}


def build_encoder(encoder_cfg):
    encoder_cls = ENCODER_REGISTRY[encoder_cfg["type"]]
    kwargs = {k: v for k, v in encoder_cfg.items() if k != "type"}
    return encoder_cls(**kwargs)


@torch.no_grad()
def infer_num_patches(encoder, image_size, device):
    encoder = encoder.to(device).eval()
    dummy = torch.zeros(1, 3, image_size, image_size, device=device)
    features = encoder(dummy)

    if features.dim() == 2:
        return 1, features.shape[-1]
    elif features.dim() == 3:
        return features.shape[1], features.shape[-1]

    raise ValueError(f"Unexpected encoder output shape: {features.shape}")


def build_dino_wm(config):
    model_cfg = config["model"]
    device = config.get("training", {}).get(
        "device", "cuda" if torch.cuda.is_available() else "cpu"
    )

    encoder = build_encoder(model_cfg["encoder"])

    num_patches, native_dim = infer_num_patches(
        encoder, model_cfg["image_size"], device
    )

    emb_dim = model_cfg["predictor"]["dim"]

    feature_adapter = (
        FeatureAdapter(native_dim, emb_dim)
        if native_dim != emb_dim
        else None
    )

    action_encoder = DINOWMActionEncoder(
        action_dim=model_cfg["action_dim"],
        emb_dim=emb_dim,
    )

    num_hist = model_cfg["num_hist"]
   

    predictor = ViTPredictor(
        num_patches=num_patches + 1,
        num_frames=num_hist,
        dim=emb_dim,
        depth=model_cfg["predictor"]["depth"],
        heads=model_cfg["predictor"]["heads"],
        mlp_dim=model_cfg["predictor"]["mlp_dim"],
        dim_head=model_cfg["predictor"].get("dim_head", 64),
        dropout=model_cfg["predictor"].get("dropout", 0.0),
        emb_dropout=model_cfg["predictor"].get("emb_dropout", 0.0),
    )

    model = DINOWM(
        encoder=encoder,
        action_encoder=action_encoder,
        predictor=predictor,
        feature_adapter=feature_adapter,
        num_hist=num_hist,
        loss_type=model_cfg.get("loss_type", "mse"),
        normalize_targets=model_cfg.get("normalize_targets", False),
        encoder_trainable=model_cfg.get("encoder_trainable", False),
    )

    return model.to(device)