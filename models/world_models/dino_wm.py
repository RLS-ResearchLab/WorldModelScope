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
    num_pred=1,
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

        self.num_hist = num_hist
        self.num_pred = num_pred
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


        B, T = actions.shape[:2]

        actions_flat = actions.reshape(B * T,*actions.shape[2:],)
        action_features = (self.action_encoder(actions_flat))
    
        action_features = (action_features.reshape(B,T,*action_features.shape[1:],))
        
        return action_features

    

    def predict(self, context, actions):
     
        B, T, P, D = context.shape

        action_tokens = self.encode_actions(actions)        # (B, T, D)
        action_tokens = action_tokens.unsqueeze(2)           # (B, T, 1, D)

        # Append action token to that frame's patch tokens.
        tokens = torch.cat([context, action_tokens], dim=2)  # (B, T, P+1, D)
        tokens = tokens.reshape(B, T * (P + 1), D)

        out = self.predictor(tokens)                         # (B, T*(P+1), D)
        out = out.reshape(B, T, P + 1, D)

       
        return out[:, :, :P, :]

   

    def compute_loss(self, batch):
        observations = batch["observations"]  # (B, T, C, H, W)
        actions = batch["actions"]            # (B, T, A)

        T_total = self.num_hist + self.num_pred
        assert observations.shape[1] == T_total, (
            f"Expected {T_total} frames (num_hist+num_pred), got {observations.shape[1]}"
        )

        z = self.encode_observations(observations)          # (B, T, P, D)
        predicted = self.predict(context=z, actions=actions) # (B, T, P, D)

       
        predicted = predicted[:, :-1]
        target = z[:, 1:]

        
        predicted = predicted[:, self.num_hist - 1:]
        target = target[:, self.num_hist - 1:]

        loss = self.compute_prediction_loss(predicted, target)
        return loss, {"prediction_loss": loss.detach()}


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