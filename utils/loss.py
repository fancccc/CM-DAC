# -*- coding: utf-8 -*-
'''
@file: loss.py
@author: fanc
@time: 2025/2/21 下午4:55
'''
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
class ClipLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07))

    def get_ground_truth(self, device, dtype, batch_size) -> torch.Tensor:
        labels = torch.arange(batch_size, device=device, dtype=dtype)
        if torch.distributed.is_initialized():
            labels += torch.distributed.get_rank() * batch_size
        return labels

    def forward(self, logits, output_dict=False):
        # 确保输入结构正确
        logits_per_image, logits_per_text = logits
        device = logits_per_image.device
        batch_size = logits_per_image.size(0)

        # 应用温度参数
        # logit_scale = self.logit_scale.exp()
        # logits_per_image = logits_per_image
        # logits_per_text = logits_per_text

        # 生成适应分布式的标签
        labels = self.get_ground_truth(
            device=device,
            dtype=torch.long,
            batch_size=batch_size
        )

        # 对称损失计算
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        total_loss = (loss_i + loss_t) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss