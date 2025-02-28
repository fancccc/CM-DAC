# -*- coding: utf-8 -*-
'''
@file: loss.py
@author: author
@time: 2025/1/8 16:40
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.metrics import bbox_iou
import numpy as np

def smooth_BCE(eps=0.1):
    """Returns label smoothing BCE targets for reducing overfitting."""
    return 1.0 - 0.5 * eps, 0.5 * eps

class BCEBlurWithLogitsLoss(nn.Module):
    """Modified BCEWithLogitsLoss to reduce missing label effects."""

    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction="none")
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)
        dx = pred - true
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()

class FocalLoss(nn.Module):
    """Applies focal loss to address class imbalance."""

    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor
        return loss.mean()

class ComputeLoss:
    """Computes the total loss for predictions, including classification and radius-based overlap losses."""

    def __init__(self, stride=8, img_size=(512, 512, 416), cls_weight=0.8, reg_weight=0.2, neg_samples_rate=3):
        self.stride = stride
        self.img_size = img_size
        # self.cls_loss_fcn = FocalLoss(gamma=gamma, alpha=alpha)
        # self.radius = radius
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.neg_samples_rate = neg_samples_rate
        # self.max_neg_samples = max_neg_samples

    def focal_loss(self, pred, target, positive_mask, negative_mask, alpha=0.25, gamma=2.0):
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        p_t = torch.exp(-loss)
        loss = alpha * (1 - p_t) ** gamma * loss
        # print(loss)
        mask = positive_mask | negative_mask

        # print(positive_mask.sum(), negative_mask.sum(), mask.sum())
        loss = loss * mask.float()
        # print('focal', loss.shape)
        return loss.sum()

    def offset_loss(self, pxyz, txyz, positive_mask):
        """计算基于偏移量的回归损失"""
        positive_mask = positive_mask.squeeze(-1)  # 去除最后一个维度，得到 [batch, height, width, depth]
        pxyz = pxyz[positive_mask]  # 应用正样本掩码
        txyz = txyz[positive_mask]
        return torch.norm(pxyz - txyz, p=2, dim=-1).sum()

    def __call__(self, pred, targets):
        p_offsets, p_cls = pred.split((3, 1), 5)
        device = pred.device
        bs, _, w, h, d, _ = pred.shape
        # Classification targets
        t_cls = torch.zeros_like(p_cls, device=device)
        t_offsets = torch.zeros((bs, 1, w, h, d, 3), device=device)
        pos_mask = torch.zeros_like(t_cls, dtype=torch.bool)
        for b in range(bs):
            for t in targets[b]:
                if t[0] < 0:  # Skip fill values
                    continue
                # Coordinate transformation
                abs_x = t[0]
                abs_y = t[1]
                abs_z = t[2]
                grid_x = min(int(abs_x // self.stride), w - 1)
                grid_y = min(int(abs_y // self.stride), h - 1)
                grid_z = min(int(abs_z // self.stride), d - 1)

                # Set classification target
                t_cls[b, 0, grid_x, grid_y, grid_z] = 1.0
                pos_mask[b, 0, grid_x, grid_y, grid_z] = True

                # # Calculate regression target (offsets)
                cell_center_x = grid_x * self.stride
                cell_center_y = grid_y * self.stride
                cell_center_z = grid_z * self.stride
                offset_x = (abs_x - cell_center_x) / self.stride
                offset_y = (abs_y - cell_center_y) / self.stride
                offset_z = (abs_z - cell_center_z) / self.stride
                t_offsets[b, 0, grid_x, grid_y, grid_z] = torch.tensor([offset_x, offset_y, offset_z])
        neg_mask = ~pos_mask
        # Negative sample selection
        # Compute losses
        bce_loss = F.binary_cross_entropy_with_logits(p_cls, t_cls, reduction='none')
        pos_avg_loss = bce_loss[pos_mask].mean() * bs
        neg_avg_loss = bce_loss[neg_mask].mean() * bs

        hard_neg_scores = p_cls[neg_mask]
        _, hard_neg_indices = torch.topk(hard_neg_scores, pos_mask.sum())
        hard_neg_mask = torch.zeros_like(neg_mask)
        hard_neg_mask.view(-1)[hard_neg_indices] = 1
        hard_neg_avg_loss = bce_loss[hard_neg_mask].mean() * bs

        cls_loss = pos_avg_loss + neg_avg_loss + hard_neg_avg_loss
        reg_loss = self.offset_loss(p_offsets, t_offsets, pos_mask)

        total_loss = self.cls_weight * cls_loss + self.reg_weight * reg_loss
        return total_loss / bs

    def select_negatives(self, pos_mask):
        # 去除多余的维度，确保 pos_mask 维度为 (batch_size, height, width)
        pos_mask = pos_mask.squeeze(1) if pos_mask.dim() == 4 else pos_mask
        # print(pos_mask.sum())
        neg_mask = ~pos_mask  # 计算负样本的掩码
        return neg_mask.unsqueeze(1)  # 保持维度一致，返回一个 (batch_size, 1, height, width) 的 tensor


class ComputeLoss2d(nn.Module):
    def __init__(self, stride=8, gamma=2.0, alpha=0.25, cls_weight=0.8, reg_weight=0.2, neg_samples_rate=3):
        super(ComputeLoss2d, self).__init__()
        self.stride = stride
        self.cls_loss_fcn = FocalLoss(alpha=alpha, gamma=gamma)
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.neg_samples_rate = neg_samples_rate


    def focal_loss(self, pred, target, mask=None, alpha=0.25, gamma=2.0):
        # Focal Loss to handle class imbalance
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = torch.exp(-loss)
        loss = alpha * (1 - p_t) ** gamma * loss
        if mask is not None:
            # print('neg samples:', mask.sum())
            return (loss * mask.float()).sum()
        else:
            return loss.sum()

    def offset_loss(self, p_offsets, t_offsets, pos_mask):
        # L1 Loss for target center coordinates (offsets)
        pos_mask = pos_mask.squeeze(-1)
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=p_offsets.device)
        return F.l1_loss(p_offsets[pos_mask], t_offsets[pos_mask], reduction='sum')

    def __call__(self, pred, targets):
        device = pred.device
        bs, _, h, w, _ = pred.shape
        p_offsets, p_cls = pred.split((2, 1), dim=-1)
        # p_cls = pred
        # Initialize target tensors
        t_cls = torch.zeros((bs, 1, h, w, 1), device=device)
        t_offsets = torch.zeros((bs, 1, h, w, 2), device=device)
        pos_mask = torch.zeros_like(t_cls, dtype=torch.bool)

        for b in range(bs):
            for t in targets[b]:
                if t[0] < 0:  # Skip fill values
                    continue
                # Coordinate transformation
                abs_x = t[0] * self.stride * w
                abs_y = t[1] * self.stride * h
                grid_x = min(int(abs_x // self.stride), w - 1)
                grid_y = min(int(abs_y // self.stride), h - 1)

                # Set classification target
                t_cls[b, 0, grid_y, grid_x] = 1.0
                pos_mask[b, 0, grid_y, grid_x] = True

                # # Calculate regression target (offsets)
                cell_center_x = grid_x * self.stride
                cell_center_y = grid_y * self.stride
                offset_x = (abs_x - cell_center_x) / self.stride
                offset_y = (abs_y - cell_center_y) / self.stride
                t_offsets[b, 0, grid_y, grid_x] = torch.tensor([offset_x, offset_y])

        # Negative sample selection
        self.max_neg_samples = self.neg_samples_rate * pos_mask.sum().item()
        neg_mask = self.select_negatives(pos_mask)
        # print(neg_mask.sum(), pos_mask.sum())

        # Compute losses
        # cls_loss = self.focal_loss(p_cls, t_cls, pos_mask | neg_mask)
        cls_loss = self.focal_loss(p_cls, t_cls, mask=neg_mask)
        reg_loss = self.offset_loss(p_offsets, t_offsets, pos_mask)

        total_loss = self.cls_weight * cls_loss + self.reg_weight * reg_loss
        return total_loss / bs

    def select_negatives(self, pos_mask):
        # 去除多余的维度，确保 pos_mask 维度为 (batch_size, height, width)
        pos_mask = pos_mask.squeeze(1) if pos_mask.dim() == 4 else pos_mask
        # print(pos_mask.sum())
        neg_mask = ~pos_mask  # 计算负样本的掩码

        num_pos = pos_mask.sum()  # 计算正样本的数量
        num_neg = min(neg_mask.sum(), self.max_neg_samples + num_pos)  # 计算负样本数量

        if num_neg == 0:
            return torch.zeros_like(neg_mask)

        # 创建一个全 1 的 tensor
        probs = torch.ones_like(neg_mask, dtype=torch.float32)

        # 将正样本的位置的概率设为 0
        probs[pos_mask] = 0.0  # 广播支持，确保 pos_mask 的维度与 probs 匹配

        # 从负样本中采样
        neg_indices = torch.multinomial(probs.view(-1), num_neg)

        # 创建一个负样本掩码，指定负样本的索引
        neg_mask = torch.zeros_like(neg_mask)
        neg_mask.view(-1)[neg_indices] = 1
        return neg_mask.unsqueeze(1)  # 保持维度一致，返回一个 (batch_size, 1, height, width) 的 tensor

class ComputeLoss2dpn(nn.Module):
    def __init__(self, stride=8, gamma=2.0, alpha=0.25, cls_weight=0.8, reg_weight=0.2, neg_samples_rate=3):
        super(ComputeLoss2dpn, self).__init__()
        self.stride = stride
        # self.cls_loss_fcn = FocalLoss(alpha=alpha, gamma=gamma)
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.neg_samples_rate = neg_samples_rate
        self.alpha = alpha
        self.gamma = gamma


    def focal_loss(self, pred, target, mask=None, alpha=0.25, gamma=2.0):
        # Focal Loss to handle class imbalance
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = torch.exp(-loss)
        loss = alpha * (1 - p_t) ** gamma * loss
        if mask is not None:
            # print('neg samples:', mask.sum())
            return (loss * mask.float()).sum()
        else:
            return loss.sum()

    def offset_loss(self, p_offsets, t_offsets, pos_mask):
        # L1 Loss for target center coordinates (offsets)
        pos_mask = pos_mask.squeeze(-1)
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=p_offsets.device)
        return F.l1_loss(p_offsets[pos_mask], t_offsets[pos_mask], reduction='mean')

    def __call__(self, pred, targets):
        device = pred.device
        bs, _, h, w, _ = pred.shape
        p_offsets, p_cls = pred.split((2, 1), dim=-1)
        # p_cls = pred
        # Initialize target tensors
        t_cls = torch.zeros((bs, 1, h, w, 1), device=device)

        t_offsets = torch.zeros((bs, 1, h, w, 2), device=device)
        pos_mask = torch.zeros_like(t_cls, dtype=torch.bool)
        for b in range(bs):
            for t in targets[b]:
                if t[0] < 0:  # Skip fill values
                    continue
                # Coordinate transformation
                # abs_x = t[0] * self.stride * w
                # abs_y = t[1] * self.stride * h
                abs_x = t[0]
                abs_y = t[1]
                # relative_x = abs_x / 512
                # relative_y = abs_y / 512
                # t_cls[b, 0, int(relative_y*h), int(relative_x*w)] = 1.0
                # relative_wh = 24 / 512
                grid_x = min(int(abs_x // self.stride), w - 1)
                grid_y = min(int(abs_y // self.stride), h - 1)

                # Set classification target
                # t_cls[b, 0, grid_y, grid_x] = 1.0
                t_cls[b, 0, :, :] = gaussian_heatmap(h, w, (grid_y, grid_x), 3).unsqueeze(-1)

                pos_mask[b, 0, grid_y, grid_x] = True

                # # Calculate regression target (offsets)
                cell_center_x = grid_x * self.stride
                cell_center_y = grid_y * self.stride
                offset_x = (abs_x - cell_center_x) / self.stride
                offset_y = (abs_y - cell_center_y) / self.stride
                t_offsets[b, 0, grid_y, grid_x] = torch.tensor([offset_x, offset_y])

        neg_mask = ~pos_mask
        # print(neg_mask.sum(), pos_mask.sum())
        # Compute losses
        # cls_loss = self.focal_loss(p_cls, t_cls, pos_mask | neg_mask)
        # cls_loss = self.focal_loss(p_cls, t_cls, mask=neg_mask)
        bce_loss = F.binary_cross_entropy_with_logits(p_cls, t_cls, reduction='none')
        p_t = torch.exp(-bce_loss)
        alpha = t_cls * self.alpha + (1 - t_cls) * (1 - self.alpha)  # 正负样本不同α
        focal_loss = alpha * (1 - p_t) ** self.gamma * bce_loss
        # focal_loss = self.alpha * (1 - p_t) ** self.gamma * bce_loss
        # print(focal_loss.shape, focal_loss.sum()/bs)
        pos_loss = focal_loss[pos_mask].sum() / pos_mask.sum().clamp(min=1.0)
        neg_loss = focal_loss[neg_mask].sum() / neg_mask.sum().clamp(min=1.0)

        hard_neg_scores = torch.sigmoid(p_cls)[neg_mask]
        _, hard_neg_indices = torch.topk(hard_neg_scores, pos_mask.sum())
        hard_neg_mask = torch.zeros_like(neg_mask)
        hard_neg_mask.view(-1)[hard_neg_indices] = 1
        hard_neg_loss = bce_loss[hard_neg_mask].sum() / hard_neg_mask.sum().clamp(min=1.0)
        cls_loss = pos_loss + neg_loss + hard_neg_loss
        # num_pos = pos_mask.sum()
        # num_neg = neg_mask.sum()
        # num_hard_neg = hard_neg_mask.sum()
        # total_samples = num_pos + num_neg + num_hard_neg
        # cls_loss = (pos_loss + neg_loss + hard_neg_loss) / total_samples.clamp(min=1.0)
        # print(total_samples.clamp(min=1.0))

        # cls_loss = pos_avg_loss + neg_avg_loss + hard_neg_avg_loss
        # num_pos = pos_mask.sum()
        # num_neg = neg_mask.sum()
        # num_hard_neg = hard_neg_mask.sum()
        # cls_loss = (pos_avg_loss * num_pos + neg_avg_loss * num_neg + hard_neg_avg_loss * num_hard_neg) / (
        #             num_pos + num_neg + num_hard_neg)
        reg_loss = self.offset_loss(p_offsets, t_offsets, pos_mask) / pos_mask.sum()

        total_loss = self.cls_weight * cls_loss + self.reg_weight * reg_loss
        # print(f'pos_avg_loss: {pos_avg_loss.item()}, neg_avg_loss: {neg_avg_loss.item()}, hard_neg_avg_loss: {hard_neg_avg_loss.item()}, reg_loss: {reg_loss.item()}')
        return total_loss

    def select_negatives(self, pos_mask):
        # 去除多余的维度，确保 pos_mask 维度为 (batch_size, height, width)
        pos_mask = pos_mask.squeeze(1) if pos_mask.dim() == 4 else pos_mask
        # print(pos_mask.sum())
        neg_mask = ~pos_mask  # 计算负样本的掩码
        #
        # num_pos = pos_mask.sum()  # 计算正样本的数量
        # num_neg = min(neg_mask.sum(), self.max_neg_samples + num_pos)  # 计算负样本数量
        #
        # if num_neg == 0:
        #     return torch.zeros_like(neg_mask)
        #
        # # 创建一个全 1 的 tensor
        # probs = torch.ones_like(neg_mask, dtype=torch.float32)
        #
        # # 将正样本的位置的概率设为 0
        # probs[pos_mask] = 0.0  # 广播支持，确保 pos_mask 的维度与 probs 匹配
        #
        # # 从负样本中采样
        # neg_indices = torch.multinomial(probs.view(-1), num_neg)
        #
        # # 创建一个负样本掩码，指定负样本的索引
        # neg_mask = torch.zeros_like(neg_mask)
        # neg_mask.view(-1)[neg_indices] = 1
        return neg_mask.unsqueeze(1)  # 保持维度一致，返回一个 (batch_size, 1, height, width) 的 tensor

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

def gaussian_heatmap(height, width, center, diameter, sigma_factor=6):
    """Generate a 2D Gaussian heatmap with specified center and diameter."""
    # Calculate sigma from diameter (sigma_factor controls the spread)
    sigma = diameter / sigma_factor

    # Create grid for the heatmap
    y, x = np.ogrid[:height, :width]
    y, x = torch.from_numpy(y).float(), torch.from_numpy(x).float()

    # Compute the squared Euclidean distance from the center
    d2 = (x - center[1])**2 + (y - center[0])**2

    # Compute the Gaussian
    heatmap = torch.exp(-d2 / (2 * sigma**2))

    return heatmap

if __name__ == "__main__":
    gts = torch.tensor([[[299, 128, 58],
                         [300, 23, 79],
                         [-1.0000, -1.0000, -1.0000]],
                        [[161, 25, 107],
                         [173, 375, 310],
                         [-1.0000, -1.0000, -1.0000]]])
    p = torch.randn(2, 1, 64, 64, 52, 4)
    print(gts.shape)
    loss = ComputeLoss(stride=8)
    print(loss(p, gts))
    print(torch.tensor([[0.2, 0.2, 0.01, 0.01]]).shape)
    print('iou: ', bbox_iou(torch.tensor([[0.2, 0.2, 0.01, 0.01]]), torch.tensor([[0.2, 0.2, 0.01, 0.01]])))
    # gts = torch.tensor([[[0.5153, 0.2978]],
    #                     [[0.1616, 0.2525]]])
    p = torch.randn(2, 1, 64, 64, 3)
    # loss = ComputeLoss2d(stride=8)
    loss2 = ComputeLoss2dpn(stride=8)
    print(loss2(p, gts))
    # print(loss(p, gts), loss2(p, gts))

    bce = nn.BCEWithLogitsLoss(reduction="none")
    print(bce(torch.randn(2, 20, 2), torch.randn(2, 20, 2)).shape)