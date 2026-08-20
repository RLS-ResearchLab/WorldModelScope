import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOWM(nn.Module):
   

    def __init__(
    self,
    encoder,
    action_encoder,
    predictor,
    num_hist,
    loss_type="mse",
    normalize_targets=False,   # paper uses plain MSE, no normalization by default
    encoder_trainable=False,
    feature_adapter=None,     
):
        super().__init__()

        self.encoder = encoder
        self.action_encoder = action_encoder
        self.predictor = predictor
        self.feature_adapter = feature_adapter

        self.num_hist = num_hist #H
        
        self.loss_type = loss_type
        self.normalize_targets = normalize_targets
        self.encoder_trainable = encoder_trainable

        if not encoder_trainable:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
            self.encoder.eval()


    def encode_observations(self, observations):
        B, T = observations.shape[:2]
        images = observations.reshape(B * T, *observations.shape[2:])

        if self.encoder_trainable:
            features = self.encoder(images)
        else:
            with torch.no_grad():
                features = self.encoder(images)

        if features.dim() == 2:
            features = features.reshape(B, T, 1, -1)
        elif features.dim() == 3:
            features = features.reshape(B, T, features.shape[1], features.shape[2])
        else:
            raise ValueError(f"Unexpected encoder output shape: {features.shape}")

 
        if self.feature_adapter is not None:
            features = self.feature_adapter(features)

        return features
    
    def encode_actions(
        self,
        actions,
    ):


        """B, T = actions.shape[:2]

        actions_flat = actions.reshape(B * T,*actions.shape[2:],)
        action_features = (self.action_encoder(actions_flat))
    
        action_features = (action_features.reshape(B,T,*action_features.shape[1:],))"""
        
        return  self.action_encoder(actions)

    

    def predict(self, context, actions):
        """
        Predict the next latent for every frame in the context.

        context:
            (B, H, P, D)

        actions:
            (B, H, A)

        Returns:
            predictions:
                (B, H, P, D)

        Interpretation for H=3:

            input frame 0 + action 0 -> prediction of frame 1
            input frame 1 + action 1 -> prediction of frame 2
            input frame 2 + action 2 -> prediction of frame 3

        Because the transformer is causally masked, the output
        at frame t can use frames <= t.
        """

        B, H, P, D = context.shape

        if actions.shape[1] != H:
            raise ValueError(
                f"Expected {H} actions, "
                f"but received {actions.shape[1]}."
            )

        # Encode every action into the predictor dimension.
        #
        # (B, H, action_dim)
        #        ↓
        # (B, H, D)
        action_tokens = self.encode_actions(actions)

        # Convert:
        #
        # (B, H, D)
        #
        # into:
        #
        # (B, H, 1, D)
        #
        # because every action becomes one token.
        action_tokens = action_tokens.unsqueeze(2)

        # Append one action token to each frame.
        #
        # Before:
        #
        # frame = [patch1 ... patchP]
        #
        # After:
        #
        # frame = [patch1 ... patchP action]
        tokens = torch.cat(
            [context, action_tokens],
            dim=2,
        )

        # Flatten temporal + spatial dimensions.
        #
        # (B, H, P+1, D)
        #        ↓
        # (B, H*(P+1), D)
        tokens = tokens.reshape(
            B,
            H * (P + 1),
            D,
        )

        # Causal transformer.
        out = self.predictor(tokens)

        # Restore frame structure.
        #
        # (B, H*(P+1), D)
        #        ↓
        # (B, H, P+1, D)
        out = out.reshape(
            B,
            H,
            P + 1,
            D,
        )

        # Remove the action-token output.
        #
        # We only want predicted visual/DINO tokens.
        return out[:, :, :P, :]

    def compute_prediction_loss(self, predicted, target):
        if self.normalize_targets:
            target = F.normalize(target, dim=-1)
            predicted = F.normalize(predicted, dim=-1)

        if self.loss_type == "mse":
            loss = F.mse_loss(predicted, target)
        elif self.loss_type == "l1":
            loss = F.l1_loss(predicted, target)
        elif self.loss_type == "cosine":
            # 1 - cosine similarity, averaged over all tokens
            cos_sim = F.cosine_similarity(predicted, target, dim=-1)
            loss = (1 - cos_sim).mean()
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        return loss

   

    """def compute_loss(self, batch):

        observations = batch["observations"]
        actions = batch["actions"]
        actions = actions[:, :-1]

        # --------------------------------------------------
        # Expected BridgeV2 training sample
        # --------------------------------------------------
        #
        # H = 3
        #
        # observations:
        #
        # I0 I1 I2 I3
        #
        # actions:
        #
        # a0 a1 a2
        #
        # Therefore:
        #
        # observations.shape[1] = H + 1
        # actions.shape[1]      = H
        # --------------------------------------------------

        B, T, C, H_img, W_img = observations.shape

        if T != self.num_hist + 1:
            raise ValueError(
                f"Expected {self.num_hist + 1} observations "
                f"for H={self.num_hist}, "
                f"but received {T}."
            )

        if actions.shape[1] != self.num_hist:
            raise ValueError(
                f"Expected {self.num_hist} actions, "
                f"but received {actions.shape[1]}."
            )

        # --------------------------------------------------
        # Encode all observations.
        # --------------------------------------------------
        #
        # I0 I1 I2 I3
        #
        #       ↓ DINOv2
        #
        # z0 z1 z2 z3
        #
        z = self.encode_observations(observations)

        # z shape:
        #
        # (B, H+1, P, D)

        # --------------------------------------------------
        # The predictor only receives the first H states.
        # --------------------------------------------------
        #
        # z0 z1 z2
        #
        # together with:
        #
        # a0 a1 a2
        #
        context = z[:, :-1]

        # --------------------------------------------------
        # Teacher-forced predictions.
        # --------------------------------------------------
        #
        # Predictor receives:
        #
        # (z0,a0)
        # (z1,a1)
        # (z2,a2)
        #
        # and produces:
        #
        # z_hat1
        # z_hat2
        # z_hat3
        #
        predicted = self.predict(
            context=context,
            actions=actions,
        )

        # --------------------------------------------------
        # Ground-truth next states.
        # --------------------------------------------------
        #
        # z1 z2 z3
        #
        target = z[:, 1:]

        # Both tensors must now have exactly the same shape:
        #
        # predicted: (B, H, P, D)
        # target:    (B, H, P, D)

        if predicted.shape != target.shape:
            raise ValueError(
                f"Prediction shape {predicted.shape} "
                f"does not match target shape {target.shape}."
            )

        # --------------------------------------------------
        # Average loss over all H predictions.
        # --------------------------------------------------

        loss = self.compute_prediction_loss(
            predicted,
            target,
        )

        return loss, {
            "prediction_loss": loss.detach(),
        }
"""

    def compute_loss(self, batch):
        observations = batch["observations"]
        actions = batch["actions"]

        B, T, C, H_img, W_img = observations.shape

        # We need at least H+1 observations:
        # H frames for context + 1 future frame as target.
        if T < self.num_hist + 1:
            raise ValueError(
                f"Need at least {self.num_hist + 1} observations "
                f"for num_hist={self.num_hist}, "
                f"but received {T}."
            )

        # Actions should be aligned with transitions:
        # a_t takes frame t -> frame t+1.
        if actions.shape[1] != T:
            raise ValueError(
                f"Expected {T} actions after Trainer padding, "
                f"but received {actions.shape[1]}."
            )

        # ---------------------------------------------------------
        # Encode the entire long clip once.
        # ---------------------------------------------------------
        z = self.encode_observations(observations)
        # z: [B, T, P, D]

        total_loss = 0.0
        num_windows = T - self.num_hist

        # ---------------------------------------------------------
        # Sliding training windows.
        # ---------------------------------------------------------
        for start in range(num_windows):

            end = start + self.num_hist

            # Context frames:
            # [start, ..., end-1]
            context = z[:, start:end]

            # Actions:
            # action[start] ... action[end-1]
            window_actions = actions[:, start:end]

            # Targets:
            # frames [start+1, ..., end]
            target = z[:, start + 1:end + 1]

            predicted = self.predict(
                context=context,
                actions=window_actions,
            )

            if predicted.shape != target.shape:
                raise ValueError(
                    f"Prediction shape {predicted.shape} "
                    f"does not match target shape {target.shape}."
                )

            total_loss = total_loss + self.compute_prediction_loss(
                predicted,
                target,
            )

        loss = total_loss / num_windows

        return loss, {
            "prediction_loss": loss.detach(),
            "num_windows": num_windows,
        }
    @torch.no_grad()
    def validation_step(self, batch):
        self.eval()
        return self.compute_loss(batch)   


    @torch.no_grad()
    def rollout(self, observations, actions):
        """
        observations: (B, num_hist, C, H, W)     initial context
        actions:      (B, num_hist + horizon, A) actions[:, t] taken FROM frame t
        """
        self.eval()

        current_context = self.encode_observations(observations)   # (B, num_hist, P, D)
        current_actions = actions[:, : self.num_hist]

        horizon = actions.shape[1] - self.num_hist
        predictions = []

        for t in range(horizon):
            predicted = self.predict(current_context, current_actions)
            next_latent = predicted[:, -1:]
            predictions.append(next_latent)

            current_context = torch.cat([current_context[:, 1:], next_latent], dim=1)

            next_action = actions[:, self.num_hist + t : self.num_hist + t + 1]
            current_actions = torch.cat([current_actions[:, 1:], next_action], dim=1)

        return torch.cat(predictions, dim=1)