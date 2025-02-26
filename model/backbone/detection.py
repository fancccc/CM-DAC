# -*- coding: utf-8 -*-
'''
@file: detection.py
@author: fanc
@time: 2025/1/8 16:00
'''
# from .ImageEncoder3D import ImageEncoder3D, PatchEmbed3D, generate_model, FpnNeck3D
from .ImageEncoder3D import *
from .RPN import RPN3D, RPN3DPostProcessing
import torch.nn as nn
import torch
# from .loss import RPN3DLoss
class Detection(nn.Module):
    def __init__(self, backbone_channel_list=[64, 128, 256, 512], stride=8, img_size=(512, 512, 416)):
        super(Detection, self).__init__()
        # patch_embed = PatchEmbed3D(kernel_size=(7, 7, 7), stride=(4, 4, 4), padding=(3, 3, 3), in_chans=256,
        #                            embed_dim=768)
        resnet = generate_model(model_depth=18)
        fpn = FpnNeck3D(d_model=256, out_channels=1024, backbone_channel_list=backbone_channel_list, stride=stride)
        self.detection = ImageEncoder3D(resnet, fpn)

    def forward(self, x):
        return self.detection(x)

class Detection2d(nn.Module):
    def __init__(self, backbone_channel_list=[64, 128, 256, 512], in_channels=3):
        super(Detection2d, self).__init__()
        resnet = ResNet2D(in_channels=in_channels)
        fpn = FpnNeck2D(backbone_channel_list=backbone_channel_list, out_channels=1024)
        self.detection = ImageEncoder2D(resnet, fpn)

    def forward(self, x):
        return self.detection(x)

if __name__ == '__main__':
    img = torch.randn(1, 1, 512, 512, 32)
    # backbone_channel_list = [64, 128, 256, 512]
    # patch_embed = PatchEmbed3D(kernel_size=(7, 7, 7), stride=(4, 4, 4), padding=(3, 3, 3), in_chans=256, embed_dim=768)
    # resnet = generate_model(model_depth=18)
    # fpn = FpnNeck3D(patch_embed, d_model=256, backbone_channel_list=backbone_channel_list)
    # img_encoder = ImageEncoder3D(resnet, fpn)
    # rpn = RPN3D(in_channels=256)
    detection = Detection()
    coords, scores = detection(img)
    # print(detection.get_criterion())
    # print(coords[0].shape, scores[0].shape)
    # postprocessing = RPN3DPostProcessing()
    # print(postprocessing(coords, scores))
    print(coords[0][0, :, 0, 0, 0], scores[0][0, :, 0, 0, 0])
