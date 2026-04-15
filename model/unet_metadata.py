from typing import List, Optional, Tuple, Union
from torch import einsum
import os
import json
from einops import rearrange
import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import logging
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from .unet_blocks import (
    CrossAttnDownBlock3D,
    CrossAttnUpBlock3D,
    DownBlock3D,
    UNetMidBlock3DCrossAttn,
    UpBlock3D,
    get_down_block,
    get_up_block)
from .resnet import InflatedConv3d
from .mask_decoder import get_mask_decoder

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
            
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class PreNormattention(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs) + x

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer_v2(nn.Module):
    def __init__(self, heads=8, dim=2048, dim_head_k=256, dim_head_v=256, dropout_atte = 0.05, mlp_dim=2048, dropout_ffn = 0.05, depth=1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.depth = depth
        for _ in range(depth):
            self.layers.append(nn.ModuleList([  # PreNormattention(2048, Attention(2048, heads = 8, dim_head = 256, dropout = 0.2))
                # PreNormattention(heads, dim, dim_head_k, dim_head_v, dropout=dropout_atte),
                PreNormattention(dim, Attention(dim, heads = heads, dim_head = dim_head_k, dropout = dropout_atte)),
                FeedForward(dim, mlp_dim, dropout = dropout_ffn),
            ]))
    def forward(self, x):
        # if self.depth
        for attn, ff in self.layers[:1]:
            x = attn(x)
            x = ff(x) + x
        if self.depth > 1:
            for attn, ff in self.layers[1:]:
                x = attn(x)
                x = ff(x) + x
        return x

class Metadata_Embedding(nn.Module):
    def __init__(self, concat_dim, needout = False):
        super().__init__()
                
        self.motion_embedding = nn.Sequential(
            nn.Conv2d(4, concat_dim * 4, 3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d((128, 224)),
            nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(concat_dim * 4, concat_dim, 3, stride=2, padding=1)
        )
        self.motion_embedding_after = Transformer_v2(heads=2, dim=concat_dim, dim_head_k=concat_dim, dim_head_v=concat_dim, dropout_atte=0.05, mlp_dim=concat_dim, dropout_ffn=0.05, depth=1)
        
        self.mv_mlp1 = Mlp(in_features=32*56*concat_dim, out_features=768)
        self.ft_mlp1 = Mlp(in_features=128, out_features=256)
        self.ft_mlp2 = Mlp(in_features=256, out_features=768)
        self.needout = needout
        # Apply zero initialization to all layers
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, motion, ft):
        batch = motion.shape[0]
        motion = rearrange(motion, 'b f h w c -> (b f) c h w')
        motion = self.motion_embedding(motion)
        h,w = motion.shape[2:4]
        motion = self.motion_embedding_after(rearrange(motion, '(b f) c h w -> (b h w) f c', b=batch))
        motion_rep = rearrange(motion, '(b h w) f c -> b c f h w', b=batch, h=h)
        frametype_rep = ft.unsqueeze(-1).unsqueeze(-1)
        frametype_rep = frametype_rep.expand(-1, -1, -1, h, w)
        frametype_rep = rearrange(frametype_rep, 'b f c h w -> b c f h w')

        motion = rearrange(motion, '(b h w) f c -> b f (c h w)', b=batch, h=h)
        motion = self.mv_mlp1(motion)
        frametype = self.ft_mlp1(ft)
        frametype = self.ft_mlp2(frametype)

        return frametype + motion, torch.cat([motion_rep,frametype_rep],dim=1)


class MGDMModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "DownBlock3D",
        ),
        mid_block_type: str = "UNetMidBlock3DCrossAttn",
        up_block_types: Tuple[str] = (
            "UpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D"
        ),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 768,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        temp_attn: bool = False,
        ref_frame_type_2: Optional[str] = None,
        cross_frame_type: str = None,
        mask_decoder_type: str = None,
        lan_cross_attn: bool = False,
        use_metadata_as_prompt: bool = False,
    ):
        super().__init__()

        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4

        # input
        if in_channels == 4:
            self.conv_in = InflatedConv3d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))
            self.condition_conv_in = InflatedConv3d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))
            nn.init.zeros_(self.condition_conv_in.weight.data)
            nn.init.zeros_(self.condition_conv_in.bias.data)
        elif in_channels == 8:
            self.conv_in = InflatedConv3d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))
            self.condition_conv_in = None

        # time
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        # class embedding
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        else:
            self.class_embedding = None

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        if isinstance(only_cross_attention, bool):
            only_cross_attention = [only_cross_attention] * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        if use_metadata_as_prompt:
            self.metadata_embedding  = Metadata_Embedding(4, needout = True)
        else:
            self.metadata_embedding = None
        
        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[i],
                downsample_padding=downsample_padding,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                temp_attn=temp_attn,
                ref_frame_type_2=ref_frame_type_2,
                cross_frame_type=cross_frame_type,
                lan_cross_attn=lan_cross_attn,
            )
            self.down_blocks.append(down_block)

        # mid
        if mid_block_type == "UNetMidBlock3DCrossAttn":
            self.mid_block = UNetMidBlock3DCrossAttn(
                in_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[-1],
                resnet_groups=norm_num_groups,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                upcast_attention=upcast_attention,
                temp_attn=temp_attn,
                ref_frame_type_2=ref_frame_type_2,
                cross_frame_type=cross_frame_type,
            )
        else:
            raise ValueError(f"unknown mid_block_type : {mid_block_type}")

        # count how many layers upsample the videos
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_attention_head_dim = list(reversed(attention_head_dim))
        only_cross_attention = list(reversed(only_cross_attention))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=reversed_attention_head_dim[i],
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                temp_attn=temp_attn,
                ref_frame_type_2=ref_frame_type_2,
                cross_frame_type=cross_frame_type,
                lan_cross_attn=lan_cross_attn,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = InflatedConv3d(block_out_channels[0], out_channels, kernel_size=3, padding=1)
        
        if mask_decoder_type:
            self.mask_decoder = get_mask_decoder(
                mask_decoder_type,
                in_channels=reversed_block_out_channels[1:],
                output_channels=64,
            )

    def forward(
        self,
        sample: torch.FloatTensor,
        motion: torch.FloatTensor,
        frametype: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        img_condition: torch.FloatTensor,
        class_labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):  
            
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb


        # pre-process
        if self.condition_conv_in is not None:
            sample = self.conv_in(sample)
            # image condition
            img_condition = self.condition_conv_in(img_condition)
            sample = sample + img_condition
        else:
            sample = torch.cat([sample, img_condition], dim=1)
            sample = self.conv_in(sample)

        if self.metadata_embedding is not None:
            encoder_hidden_states, metadata_rep = self.metadata_embedding(motion, frametype)

        # down
        down_block_res_samples = (sample,)
        for i, downsample_block in enumerate(self.down_blocks):
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # mid
        sample = self.mid_block(
            sample, emb, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
        )

        # up
        cross_attn_up_features = []
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )
                cross_attn_up_features.append(sample)
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )
            
        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        # decode mask output
        if self.mask_decoder is not None:
            cross_attn_up_features = [feature.to(self.mask_decoder.final_conv.weight.dtype) for feature in cross_attn_up_features]
            mask = self.mask_decoder(cross_attn_up_features, img_condition, metadata_rep)
            mask.to(self.conv_in.weight.dtype)
        else:
            mask = None
        if not return_dict:
            return (sample,)
        return sample, mask
