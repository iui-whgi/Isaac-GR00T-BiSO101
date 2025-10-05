# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.action_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)

class MambaBlock(nn.Module):
    """Flexible Mamba SSM block that adapts to input dimensions."""
    
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        
        # Input projection - now uses d_model instead of hardcoded value
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Convolution for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            padding=d_conv - 1,
            groups=self.d_inner,
        )
        
        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        # Output projection - now uses d_model instead of hardcoded value
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Input projection
        xz = self.in_proj(x)  # [batch, seq_len, 2*d_inner]
        x, z = xz.chunk(2, dim=-1)  # Each: [batch, seq_len, d_inner]
        
        # Convolution
        x = x.transpose(1, 2)  # [batch, d_inner, seq_len]
        x = self.conv1d(x)[:, :, :seq_len]  # [batch, d_inner, seq_len]
        x = x.transpose(1, 2)  # [batch, seq_len, d_inner]
        
        # Simplified SSM with gating
        x = F.silu(x)
        x = x * F.sigmoid(z)
        
        # Output projection
        x = self.out_proj(x)
        
        return x


class MambaActionModel(nn.Module):
    """Mamba-based action model for GR00T."""
    
    def __init__(self, d_model: int = 1536, n_layers: int = 6, 
                 d_state: int = 16, d_conv: int = 4, expand: int = 2, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Mamba layers with correct input dimension
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(n_layers)
        ])
        
        # Layer normalization with correct dimension
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, hidden_states=None, x=None, **kwargs):
        # Use hidden_states if provided, otherwise use x
        if hidden_states is not None:
            x = hidden_states
        elif x is None:
            raise ValueError("Either hidden_states or x must be provided")
        
        # Apply Mamba layers
        for layer in self.layers:
            residual = x
            x = layer(x)
            x = x + residual  # Residual connection
        
        # Final normalization
        x = self.norm(x)
        
        return x


class MambaSelfAttention(nn.Module):
    """Mamba-based self-attention for vision-language processing."""
    
    def __init__(self, d_model: int = 2048, n_layers: int = 4,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Mamba layers with correct input dimension
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(n_layers)
        ])
        
        # Layer normalization with correct dimension
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, **kwargs):
        # Apply Mamba layers
        for layer in self.layers:
            residual = x
            x = layer(x)
            x = x + residual  # Residual connection
        
        # Final normalization
        x = self.norm(x)
        
        return x


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))
    
    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)
    
    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments
        
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)
    
    def forward(self, actions, timesteps, cat_ids):
        B, T, _ = actions.shape
        
        # Expand timesteps if needed
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError("Expected timesteps to have shape (B,)")
        
        # Action embedding
        a_emb = self.W1(actions, cat_ids)
        
        # Time embedding
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)
        
        # Combine and process
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))
        x = self.W3(x, cat_ids)
        
        return x


@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    add_pos_embed: bool = True
    model_dtype: str = "float32"
    diffusion_model_cfg: dict = None
    input_embedding_dim: int = 1536
    backbone_embedding_dim: int = 1536
    hidden_size: int = 1024
    max_seq_len: int = 1024
    action_dim: int = None
    action_horizon: int = None
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000
    num_inference_timesteps: int = None
    max_num_embodiments: int = 32
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    load_pretrained_det_decode_layer_path: str = None
    detection_coeff: float = 1.0
    freeze_decode_layer: bool = False
    expand_batch: int = None
    use_vlln: bool = True
    vl_self_attention_cfg: dict = None
    num_target_vision_tokens: int = 32
    max_state_dim: int = None  # Add this field
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class FlowmatchingActionHead(nn.Module):
    config_class = FlowmatchingActionHeadConfig
    supports_gradient_checkpointing = True
    
    def __init__(self, config: FlowmatchingActionHeadConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim
        
        # Use MambaActionModel with correct dimensions
        self.model = MambaActionModel(
            d_model=self.input_embedding_dim,  # Use input_embedding_dim (1536)
            n_layers=6,
            d_state=16,
            d_conv=4,
            expand=2
        )
        
        # Add projection layer to map from input_embedding_dim to hidden_size
        self.output_projection = nn.Linear(self.input_embedding_dim, self.hidden_size)
        
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps
        
        # State encoder
        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        
        # Action encoder
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        
        # Action decoder - keep original dimensions for compatibility with checkpoint
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,  # Back to hidden_size for checkpoint compatibility
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )
        
        # Future tokens
        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)
        
        # Vision-Language normalization
        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim)
            if config.use_vlln else nn.Identity()
        )
        
        # Self-attention with correct dimension
        self.vl_self_attention = (
            MambaSelfAttention(
                d_model=config.backbone_embedding_dim,  # Use backbone_embedding_dim
                n_layers=4,
                d_state=16,
                d_conv=4,
                expand=2
            )
            if config.use_vlln else nn.Identity()
        )
        
        # Position embedding
        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        
        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config
        
        self.set_trainable_parameters(config.tune_projector, config.tune_diffusion_model)
    
    def set_trainable_parameters(self, tune_projector: bool, tune_diffusion_model: bool):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        
        for p in self.parameters():
            p.requires_grad = True
        
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
    
    def set_frozen_modules_to_eval_mode(self):
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()
    
    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s
    
    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)
    
    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output
    
    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()
        
        backbone_output = self.process_backbone_output(backbone_output)
        
        # Handle batch expansion if configured
        if self.config.expand_batch is not None:
            for k, v in backbone_output.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch] + [1] * (ndim - 1)
                backbone_output[k] = v.repeat(*factors)
            
            for k, v in action_input.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch] + [1] * (ndim - 1)
                action_input[k] = v.repeat(*factors)
        
        # Get vision and language embeddings
        vl_embs = backbone_output.backbone_features
        device = vl_embs.device
        
        # Get embodiment ID
        embodiment_id = action_input.embodiment_id
        
        # Embed state
        state_features = self.state_encoder(action_input.state, embodiment_id)
        
        # Embed noised action trajectory
        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast
        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise
        
        # Convert continuous t to discrete
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)
        
        # Add position embedding if configured
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs
        
        # Concatenate all embeddings
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)
        
        vl_attn_mask = backbone_output.backbone_attention_mask
        
        # Run model forward
        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=vl_attn_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,
        )
        
        # Project from input_embedding_dim to hidden_size
        model_output = self.output_projection(model_output)
        
        # Decode actions
        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1]:]
        
        # Calculate loss
        action_mask = action_input.action_mask
        loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = loss.sum() / action_mask.sum()
        
        output_dict = {"loss": loss}
        return BatchFeature(data=output_dict)
    
    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        backbone_output = self.process_backbone_output(backbone_output)
        
        vl_embs = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id
        
        # Embed state
        state_features = self.state_encoder(action_input.state, embodiment_id)
        
        # Initialize actions as noise
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )
        
        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps
        
        # Run denoising steps
        for t in range(num_steps):
            t_cont = t / float(num_steps)
            t_discretized = int(t_cont * self.num_timestep_buckets)
            
            timesteps_tensor = torch.full(
                size=(batch_size,),
                fill_value=t_discretized,
                device=device
            )
            
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs
            
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)
            
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
            )
            
            # Project from input_embedding_dim to hidden_size
            model_output = self.output_projection(model_output)
            
            pred = self.action_decoder(model_output, embodiment_id)
            pred_velocity = pred[:, -self.action_horizon:]
            
            # Update actions using euler integration
            actions = actions + dt * pred_velocity
        
        return BatchFeature(data={"action_pred": actions})
    
    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype