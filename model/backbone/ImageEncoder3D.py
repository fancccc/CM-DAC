# -*- coding: utf-8 -*-
'''
@file: ImageEncoder3D.py
@author: fanc
@time: 2025/1/8 13:39
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from model.backbone.ResNet3D import generate_model
from typing import Tuple
class ImageEncoder3D(nn.Module):
    def __init__(
            self,
            trunk: nn.Module,  # Backbone for 3D (e.g., 3D CNN)
            neck: nn.Module,  # Feature Pyramid Network for 3D
            scalp: int = 0,  # Control how many layers to discard
    ):
        super().__init__()
        self.trunk = trunk  # 3D backbone
        self.neck = neck  # 3D neck (FPN)
        self.scalp = scalp

        # Ensure that the channel dimensions of trunk and neck match
        assert self.trunk.channel_list == self.neck.backbone_channel_list, \
            f"Channel dims of trunk and neck do not match. Trunk: {self.trunk.channel_list}, neck: {self.neck.backbone_channel_list}"

    def forward(self, sample: torch.Tensor):
        # Forward through backbone
        x = self.trunk(sample)
        x = self.neck(x)
        # if self.scalp > 0:
        #     # Discard the lowest resolution features (3D)
        #     features, pos = features[: -self.scalp], pos[: -self.scalp]

        # src = features[-1]
        # output = {
        #     "vision_features": src,
        #     "vision_pos_enc": pos,
        #     "backbone_fpn": features,
        # }
        return x


class FpnNeck3D(nn.Module):
    """
    A modified version of Feature Pyramid Network (FPN) for 3D images.
    (We remove the output conv and apply bilinear interpolation similar to ViT's position embedding interpolation)
    """

    def __init__(
            self,
            # position_encoding: nn.Module,
            d_model: int,
            out_channels: int,
            backbone_channel_list: List[int],
            # kernel_size: int = 3,
            stride: int = 8,
            # padding: int = 1,
            # fpn_interp_model: str = "trilinear",  # Use trilinear interpolation for 3D
            # fuse_type: str = "sum",
            # fpn_top_down_levels: Optional[List[int]] = None,
            pool_sizes=[1, 5, 9, 13]
    ):
        """Initialize the neck for 3D
        :param trunk: the backbone for 3D (e.g., 3D CNN)
        :param position_encoding: the positional encoding for 3D space
        :param d_model: the model dimension
        :param neck_norm: the normalization to use
        """
        super().__init__()
        # self.position_encoding = position_encoding
        self.stride = stride
        self.pool_sizes = pool_sizes
        self.backbone_channel_list = backbone_channel_list
        self.pools = nn.ModuleList([nn.MaxPool3d(kernel_size=size, stride=1, padding=size // 2) for size in pool_sizes])
        self.conv = nn.Conv3d(backbone_channel_list[-1] * len(pool_sizes), out_channels, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv3d(1024, 256, kernel_size=1, stride=1,
                               padding=0)
        self.conv2 = nn.Conv3d(256*2, 128, kernel_size=1, stride=1,
                               padding=0)

        self.decetion = nn.Conv3d(128*2, 4, kernel_size=1, stride=1, padding=0)
        # self.conv3 = nn.Conv3d(backbone_channel_list[-1] * len(pool_sizes), out_channels, kernel_size=1, stride=1,
        #                        padding=0)
        # self.conv4 = nn.Conv3d(backbone_channel_list[-1] * len(pool_sizes), out_channels, kernel_size=1, stride=1,
        #                        padding=0)
        # self.conv_transpose1 = nn.ConvTranspose3d(in_channels=backbone_channel_list[-1],
        #                                           out_channels=backbone_channel_list[-2],
        #                                           kernel_size=4, stride=2, padding=1)

    def forward(self, xs: List[torch.Tensor]):
        pool_outputs = [pool(xs[-1]) for pool in self.pools]
        x = torch.cat(pool_outputs, dim=1)
        # for i in xs:
        #     print(i.shape)
        # print(x.shape)
        x = self.conv(x)
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat([x, xs[-2]], dim=1) # b, 256*2, w/16, h/16, d/16
        # print(x.shape)
        x = self.conv2(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        # print(x.shape, xs[-3].shape)
        x = torch.cat([x, xs[-3]], dim=1) # b, 128*2, w/8, h/8, d/8
        # print(x.shape)
        x = self.decetion(x)
        # print(x.shape)
        bs, _, nx, ny, nz = x.shape
        x = x.view(bs, 1, 4, nx, ny, nz).permute(0, 1, 3, 4, 5, 2).contiguous() # b, na, ny, nx, nz, 4
        # print(x.shape)
        # if not self.training:
        #inference
        self.device = x.device
        self.grid = self._make_grid(nx, ny, nz)
        xyz, conf = x.sigmoid().split((3, 1), 5)
        # print(xyz.shape, conf.shape, xyz[0, 0, 0, 0, 0, :], self.grid[0, 0, 0, 0, 0, :])
        xyz = (xyz * 2 + self.grid) * self.stride  # xyz
        # # wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
        y = torch.cat((xyz, conf), -1)
        y = y.view(bs, 1 * nx * ny * nz, 4)
        # print(y.shape)
        return x, y
        # return x if self.training else y
        # return x
            # print(xyz.shape, conf.shape, xyz[0, 0, 0, 0, 0, :])

    def _make_grid(self, nx=20, ny=20, nz=20, i=0):
        """Generates a mesh grid for anchor boxes with optional compatibility for torch versions < 1.10."""
        d = self.device
        t = torch.float32
        shape = 1, 1, ny, nx, nz, 3  # grid shape
        y, x, z = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t), torch.arange(nz, device=d, dtype=t)
        yv, xv, zv = torch.meshgrid(y, x, z, indexing="ij")
        grid = torch.stack((xv, yv, zv), -1).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        # anchor_grid = (self.anchors * self.stride).view((1, self.na, 1, 1, 2)).expand(shape)
        # return grid, anchor_grid
        return grid

class ImageEncoder2D(nn.Module):
    def __init__(
            self,
            trunk: nn.Module,  # Backbone for 3D (e.g., 3D CNN)
            neck: nn.Module,  # Feature Pyramid Network for 3D
            scalp: int = 0,  # Control how many layers to discard
    ):
        super().__init__()
        self.trunk = trunk  # 3D backbone
        self.neck = neck  # 3D neck (FPN)
        self.scalp = scalp

        # Ensure that the channel dimensions of trunk and neck match
        assert self.trunk.channel_list == self.neck.backbone_channel_list, \
            f"Channel dims of trunk and neck do not match. Trunk: {self.trunk.channel_list}, neck: {self.neck.backbone_channel_list}"

    def forward(self, sample: torch.Tensor):
        # Forward through backbone
        x = self.trunk(sample)
        x = self.neck(x)
        return x


class FpnNeck2D(nn.Module):
    """
    A modified version of Feature Pyramid Network (FPN) for 3D images.
    (We remove the output conv and apply bilinear interpolation similar to ViT's position embedding interpolation)
    """

    def __init__(
            self,
            # position_encoding: nn.Module,
            # d_model: int,
            out_channels: int,
            backbone_channel_list: List[int],
            stride: int = 8,
            pool_sizes=[1, 5, 9, 13]
    ):
        """Initialize the neck for 3D
        :param trunk: the backbone for 3D (e.g., 3D CNN)
        :param position_encoding: the positional encoding for 3D space
        :param d_model: the model dimension
        :param neck_norm: the normalization to use
        """
        super().__init__()
        # self.position_encoding = position_encoding
        self.stride = stride
        self.pool_sizes = pool_sizes
        self.backbone_channel_list = backbone_channel_list
        self.pools = nn.ModuleList([nn.MaxPool2d(kernel_size=size, stride=1, padding=size // 2) for size in pool_sizes])
        self.conv = nn.Conv2d(backbone_channel_list[-1] * len(pool_sizes), out_channels, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1,
                               padding=0)
        self.conv2 = nn.Conv2d(256*2, 128, kernel_size=1, stride=1,
                               padding=0)

        self.decetion = nn.Conv2d(128*2, 3, kernel_size=1, stride=1, padding=0)
        # self.conv3 = nn.Conv3d(backbone_channel_list[-1] * len(pool_sizes), out_channels, kernel_size=1, stride=1,
        #                        padding=0)
        # self.conv4 = nn.Conv3d(backbone_channel_list[-1] * len(pool_sizes), out_channels, kernel_size=1, stride=1,
        #                        padding=0)
        # self.conv_transpose1 = nn.ConvTranspose3d(in_channels=backbone_channel_list[-1],
        #                                           out_channels=backbone_channel_list[-2],
        #                                           kernel_size=4, stride=2, padding=1)

    def forward(self, xs: List[torch.Tensor]):
        pool_outputs = [pool(xs[-1]) for pool in self.pools]
        x = torch.cat(pool_outputs, dim=1)
        # for i in xs:
        #     print(i.shape)
        # print(x.shape)
        x = self.conv(x)
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat([x, xs[-2]], dim=1) # b, 256*2, w/16, h/16, d/16
        # print(x.shape)
        x = self.conv2(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        # print(x.shape, xs[-3].shape)
        x = torch.cat([x, xs[-3]], dim=1) # b, 128*2, w/8, h/8, d/8
        # print(x.shape)
        x = self.decetion(x) # b, 3, h, w
        # print(x.shape)
        bs, _, ny, nx = x.shape
        x = x.view(bs, 1, 3, ny, nx).permute(0, 1, 3, 4, 2).contiguous() # b, na, nx, ny, 1
        # print(x.shape)
        # if not self.training:
        #inference
        self.device = x.device
        self.grid = self._make_grid(ny, nx)
        yx, conf = x.sigmoid().split((2, 1), -1)
        # conf = x.sigmoid()
        # print(xyz.shape, conf.shape, xyz[0, 0, 0, 0, 0, :], self.grid[0, 0, 0, 0, 0, :])
        # xy = (xy * 2 + self.grid) * self.stride  # xy
        # yx = torch.ones_like(conf, device=x.device) * 0.5

        # yx = (yx + self.grid + 0.5) * self.stride
        yx = (yx + self.grid) * self.stride
        # xy = torch.cat((xy[..., 1:], xy[..., :1]), dim=-1)
        # print(yx, self.grid)
        # # wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
        # print(xy.shape, conf.shape)
        p = torch.cat((yx, conf), -1)
        p = p.view(bs, 1 * nx * ny, 3)
        # print(y.shape)
        return x, p
        # return x if self.training else y
        # return x
            # print(xyz.shape, conf.shape, xyz[0, 0, 0, 0, 0, :])

    def _make_grid(self, nx=20, ny=20, i=0):
        """Generates a mesh grid for anchor boxes with optional compatibility for torch versions < 1.10."""
        d = self.device
        t = torch.float32
        shape = 1, 1, nx, ny, 2  # grid shape
        x, y = torch.arange(nx, device=d, dtype=t), torch.arange(ny, device=d, dtype=t)
        xv, yv = torch.meshgrid(x, y, indexing="ij")
        grid = torch.stack((xv, yv), -1).expand(shape)  # add grid offset, i.e. y = 2.0 * x - 0.5
        # anchor_grid = (self.anchors * self.stride).view((1, self.na, 1, 1, 2)).expand(shape)
        # return grid, anchor_grid
        return grid


from torchvision.models import resnet18


class ResNet2D(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.model = resnet18(pretrained=False)
        self.channel_list = [64, 128, 256, 512]  # 各层输出通道数
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.conv1 = self.model.conv1
        # self.conv1.in_channels = in_channels

    def forward(self, x):
        # 提取多层级特征
        x = self.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        f1 = self.model.layer1(x)  # 64
        f2 = self.model.layer2(f1)  # 128
        f3 = self.model.layer3(f2)  # 256
        f4 = self.model.layer4(f3)  # 512
        return [f1, f2, f3, f4]
class PatchEmbed3D(nn.Module):
    """
    3D Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, ...] = (7, 7, 7),
        stride: Tuple[int, ...] = (4, 4, 4),
        padding: Tuple[int, ...] = (3, 3, 3),
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int):  embed_dim (int): Patch embedding dimension.
        """
        super().__init__()
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C D H W -> B D H W C
        x = x.permute(0, 2, 3, 4, 1)
        return x

if __name__ == "__main__":
    # img = torch.randn(1, 1, 512, 512, 416)
    # backbone_channel_list = [64, 128, 256, 512]
    #
    # # patch_embed = PatchEmbed3D(kernel_size=(7, 7, 7), stride=(4, 4, 4), padding=(3, 3, 3), in_chans=256, embed_dim=768)
    # resnet = generate_model(model_depth=18)
    # # out = resnet(img)
    # # for i in out:
    # #     print(i.shape)
    #
    # fpn = FpnNeck3D(d_model=256, out_channels=1024, backbone_channel_list=backbone_channel_list)
    # # fpn.eval()
    # # features, pos = fpn(out)
    # # print(len(features), len(pos))
    # # for i in range(len(features)):
    # #     print(features[i].shape, pos[i].shape)
    #
    #
    # model = ImageEncoder3D(resnet, fpn)
    # output = model(img)
    # print(output[0].shape, output[1].shape)

    img = torch.randn(1, 1, 512, 512)
    backbone_channel_list = [64, 128, 256, 512]
    resnet = ResNet2D()
    fpn = FpnNeck2D(backbone_channel_list=backbone_channel_list, out_channels=1024)
    model = ImageEncoder2D(resnet, fpn)
    output = model(img)
    print(output[0].shape, output[1].shape, output[1])
