from einops import rearrange
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn import init as init
from .resnet import InflatedConv3d

def inflated_interpolate_3d(x, size=None, scale_factor=None, mode='nearest'):
    video_length = x.shape[2]
    x = rearrange(x, "b c f h w -> (b f) c h w")
    x = F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode)
    x = rearrange(x, "(b f) c h w -> b c f h w", f=video_length)
    return x

def get_mask_decoder(module_type, in_channels, output_channels=1):
    if module_type == "Metadata_MaskDecoder":
        return Metadata_MaskDecoder(in_channels=in_channels, output_channels=output_channels)
    return None

class Metadata_MaskDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        output_channels,
        reduced_channel=512,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.output_channels = output_channels
        self.projects = nn.ModuleList()
        
        for i in range(len(in_channels)):
            if i == 0:
                channel = in_channels[i]
            else:
                channel = reduced_channel + in_channels[i]
            self.projects.append(MultiScaleProj(channel, reduced_channel))

        self.mask_reconstuction = nn.Sequential(
            InflatedConv3d(reduced_channel + 136, self.output_channels, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(ResidualBlock3DNoBN, 3, num_feat=self.output_channels)
        )
        self.final_conv = InflatedConv3d(self.output_channels, self.output_channels, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, features, curropts, motion_rep):
        x = features[0]
        for i in range(len(self.in_channels)):
            x = self.projects[i](x)
            if i < len(self.in_channels) - 2:
                x = inflated_interpolate_3d(x, scale_factor=2, mode='nearest')
            if i < len(self.in_channels) - 1:
                x = torch.cat([x, features[i + 1]], dim=1)

        x = torch.cat([x, curropts, motion_rep], dim=1)
        x = self.mask_reconstuction(x)
        x = self.final_conv(x)

        if self.output_channels == 64:
            x = rearrange(x, "b c t h w -> (b t) c h w")
            x = F.pixel_shuffle(x, upscale_factor=8)
            x = rearrange(x, "(b t) c h w -> b c t h w", b=features[0].shape[0])
        x = torch.sigmoid(x)
        return x


class MultiScaleProj(nn.Module):
    def __init__(self, C_in, C_out):
        super().__init__()
        self.conv1 = InflatedConv3d(C_in, C_out, 3, 1, 1, bias=False)
        self.conv2 = InflatedConv3d(C_out, C_out, 3, 1, 1, bias=False)
        self.conv_temp = nn.Conv1d(C_out, C_out, 3, 1, 1, bias=False)
        self.norm1 = nn.GroupNorm(32, C_out)
        self.norm2 = nn.GroupNorm(32, C_out)
        self.norm_temp = nn.GroupNorm(32, C_out)

    def forward(self,x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x)

        # temporal convolution
        h, w = x.shape[-2:]
        x = rearrange(x, "b c f h w -> (b h w) c f")
        x = self.conv_temp(x)
        x = self.norm_temp(x)
        x = F.relu(x)
        x = rearrange(x, "(b h w) c f -> b c f h w", h=h, w=w)
        
        return x

def make_layer(basic_block, num_basic_block, **kwarg):
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)

class ResidualBlock3DNoBN(nn.Module):
    """3D Residual block without Batch Normalization (BN).

    Args:
        num_feat (int): Channel number of intermediate features. Default: 64.
        res_scale (float): Residual scale. Default: 1.
    """

    def __init__(self, num_feat=64, res_scale=1):
        super(ResidualBlock3DNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv3d(num_feat, num_feat, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv3d(num_feat, num_feat, kernel_size=3, stride=1, padding=1, bias=True)
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.normal_(self.conv1.weight, 0, 0.1)
        nn.init.normal_(self.conv2.weight, 0, 0.1)
        if self.conv1.bias is not None:
            nn.init.constant_(self.conv1.bias, 0)
        if self.conv2.bias is not None:
            nn.init.constant_(self.conv2.bias, 0)

    def forward(self, x):
        identity = x
        out = self.conv2(self.LeakyReLU(self.conv1(x)))
        return identity + out * self.res_scale