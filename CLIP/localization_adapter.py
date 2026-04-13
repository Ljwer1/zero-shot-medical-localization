import torch
from torch import nn
from torch.nn import functional as F


class SpatialTextGuidedAdapter(nn.Module):
    def __init__(self, c_in=1024, bottleneck=768, text_dim=768):
        super().__init__()
        self.text_dim = text_dim
        self.fc1 = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LeakyReLU(inplace=False),
        )
        self.text_proj = nn.Linear(text_dim, bottleneck, bias=False)
        self.token_norm = nn.LayerNorm(bottleneck)
        self.text_norm = nn.LayerNorm(bottleneck)
        self.gate = nn.Linear(bottleneck * 3, bottleneck, bias=False)
        self.context_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        self.spatial_depthwise = nn.Conv2d(
            bottleneck,
            bottleneck,
            kernel_size=3,
            padding=1,
            groups=bottleneck,
            bias=False,
        )
        self.spatial_pointwise = nn.Conv2d(
            bottleneck,
            bottleneck,
            kernel_size=1,
            bias=False,
        )
        self.spatial_gate = nn.Linear(bottleneck * 2, bottleneck, bias=False)
        self.spatial_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.mix_norm = nn.LayerNorm(bottleneck)

        self.fc2 = nn.Sequential(
            nn.Linear(bottleneck, c_in, bias=False),
            nn.LeakyReLU(inplace=False),
        )

    def _prepare_text_context(self, text_context, target_dtype, device):
        if text_context is None:
            return None

        if text_context.dim() == 3:
            if text_context.shape[0] == 1:
                text_context = text_context.squeeze(0)
            else:
                text_context = text_context.mean(dim=0)

        if text_context.dim() != 2:
            raise ValueError(
                f"text_context should be 2D or squeezable to 2D, got shape {tuple(text_context.shape)}"
            )

        if text_context.shape[-1] == self.text_dim:
            prepared = text_context
        elif text_context.shape[0] == self.text_dim:
            prepared = text_context.transpose(0, 1)
        else:
            raise ValueError(
                f"text_context last dimension should match {self.text_dim}, got shape {tuple(text_context.shape)}"
            )

        return prepared.to(device=device, dtype=target_dtype)

    def _apply_spatial_mixing(self, hidden):
        if hidden.shape[0] <= 1:
            return hidden

        cls_token = hidden[:1]
        patch_tokens = hidden[1:]
        patch_count, batch_size, channels = patch_tokens.shape
        grid_size = int(round(patch_count ** 0.5))
        if grid_size * grid_size != patch_count:
            return hidden

        spatial_tokens = patch_tokens.permute(1, 2, 0).reshape(batch_size, channels, grid_size, grid_size)
        spatial_tokens = self.spatial_depthwise(spatial_tokens)
        spatial_tokens = self.spatial_pointwise(spatial_tokens)
        spatial_tokens = spatial_tokens.reshape(batch_size, channels, patch_count).permute(2, 0, 1)

        spatial_gate = torch.sigmoid(self.spatial_gate(torch.cat([patch_tokens, spatial_tokens], dim=-1)))
        patch_tokens = patch_tokens + self.spatial_scale.tanh() * spatial_gate * spatial_tokens
        patch_tokens = self.mix_norm(patch_tokens)
        return torch.cat([cls_token, patch_tokens], dim=0)

    def forward(self, x, text_context):
        hidden = self.fc1(x)
        hidden = self._apply_spatial_mixing(hidden)

        prepared_text = self._prepare_text_context(
            text_context,
            target_dtype=hidden.dtype,
            device=hidden.device,
        )
        text_tokens = self.text_proj(prepared_text)
        token_query = F.normalize(self.token_norm(hidden), dim=-1)
        text_key = F.normalize(self.text_norm(text_tokens), dim=-1)

        attention = torch.einsum("lbd,nd->lbn", token_query, text_key)
        attention = torch.softmax(attention, dim=-1)
        text_context_tokens = torch.einsum("lbn,nd->lbd", attention, text_tokens)

        gate_input = torch.cat(
            [hidden, text_context_tokens, hidden - text_context_tokens],
            dim=-1,
        )
        gate = torch.sigmoid(self.gate(gate_input))
        hidden = hidden + self.context_scale.tanh() * gate * text_context_tokens

        out = self.fc2(hidden)
        return hidden, out


class LocalizationCLIP(nn.Module):
    def __init__(self, clip_model, features):
        super().__init__()
        if len(features) != 2:
            raise ValueError(f"LocalizationCLIP expects exactly 2 adapter layers, got {features}")

        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual
        self.features = features
        self.layer_to_feature_idx = {layer: idx for idx, layer in enumerate(features)}
        self.seg_adapters = nn.ModuleList(
            [SpatialTextGuidedAdapter(c_in=1024, bottleneck=768, text_dim=768) for _ in range(len(features))]
        )
        residual_init = torch.tensor([0.85, 0.15], dtype=torch.float32)
        self.residual_weight_logits = nn.Parameter(residual_init.log().repeat(len(features), 1))
        self.seg_fusion_logits = nn.Parameter(torch.zeros(len(features), dtype=torch.float32))

    def get_residual_weights(self):
        return torch.softmax(self.residual_weight_logits, dim=-1)

    def get_fusion_weights(self):
        return len(self.features) * torch.softmax(self.seg_fusion_logits, dim=0)

    def get_fusion_regularization(self):
        fusion_distribution = torch.softmax(self.seg_fusion_logits, dim=0)
        uniform = torch.full_like(fusion_distribution, 1.0 / fusion_distribution.numel())
        return torch.sum(
            fusion_distribution * (torch.log(fusion_distribution + 1e-8) - torch.log(uniform))
        )

    def localization_state_dict(self):
        return {
            "seg_adapters": self.seg_adapters.state_dict(),
            "residual_weight_logits": self.residual_weight_logits.detach().cpu(),
            "seg_fusion_logits": self.seg_fusion_logits.detach().cpu(),
            "features": list(self.features),
        }

    def load_localization_state_dict(self, checkpoint):
        self.seg_adapters.load_state_dict(checkpoint["seg_adapters"])

        residual_weight_logits = checkpoint.get("residual_weight_logits")
        if residual_weight_logits is not None and tuple(residual_weight_logits.shape) == tuple(self.residual_weight_logits.shape):
            self.residual_weight_logits.data.copy_(
                residual_weight_logits.to(device=self.residual_weight_logits.device, dtype=self.residual_weight_logits.dtype)
            )

        seg_fusion_logits = checkpoint.get("seg_fusion_logits")
        if seg_fusion_logits is not None and tuple(seg_fusion_logits.shape) == tuple(self.seg_fusion_logits.shape):
            self.seg_fusion_logits.data.copy_(
                seg_fusion_logits.to(device=self.seg_fusion_logits.device, dtype=self.seg_fusion_logits.dtype)
            )

    def forward(self, x, text_context):
        x = self.image_encoder.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = torch.cat(
            [
                self.image_encoder.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )
        x = x + self.image_encoder.positional_embedding.to(x.dtype)
        x = self.image_encoder.patch_dropout(x)
        x = self.image_encoder.ln_pre(x)
        x = x.permute(1, 0, 2)

        seg_patch_tokens = []
        residual_weights = self.get_residual_weights()

        for layer_id, resblock in enumerate(self.image_encoder.transformer.resblocks, start=1):
            x, _ = resblock(x, attn_mask=None)
            feature_idx = self.layer_to_feature_idx.get(layer_id)
            if feature_idx is None:
                continue

            seg_hidden, seg_out = self.seg_adapters[feature_idx](x, text_context=text_context)
            branch_weights = residual_weights[feature_idx].to(dtype=x.dtype, device=x.device)
            x = branch_weights[0] * x + branch_weights[1] * seg_out
            seg_patch_tokens.append(seg_hidden.permute(1, 0, 2))

        x = x.permute(1, 0, 2)
        pooled, _ = self.image_encoder._global_pool(x)
        pooled = self.image_encoder.ln_post(pooled)
        if self.image_encoder.proj is not None:
            pooled = pooled @ self.image_encoder.proj

        return pooled, seg_patch_tokens
