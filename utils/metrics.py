# -*- coding: utf-8 -*-
'''
@file: metrics.py
@author: author
@time: 2025/1/14 15:03
'''
import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.ops as ops

def empty_like(x):
    """Creates empty torch.Tensor or np.ndarray with same shape as input and float32 dtype."""
    return (
        torch.empty_like(x, dtype=torch.float32) if isinstance(x, torch.Tensor) else np.empty_like(x, dtype=np.float32)
    )
def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner. Note: ops per 2 channels faster than per channel.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)  # faster than clone/copy
    xy = x[..., :2]  # centers
    wh = x[..., 2:] / 2  # half width-height
    y[..., :2] = xy - wh  # top left xy
    y[..., 2:] = xy + wh  # bottom right xy
    return y
def nms_3d(predictions, confidence_threshold=0.5, distance_threshold=10.0):
    # Step 1: Filter low confidence predictions
    predictions = predictions[predictions[:, 3] > confidence_threshold]
    if len(predictions) == 0:
        return np.array([])  # Return empty if no predictions pass the threshold
    # Step 2: Sort by confidence
    predictions = predictions[np.argsort(predictions[:, 3])[::-1]]
    # Step 3: Apply NMS
    keep = []
    while len(predictions) > 0:
        # Pick the prediction with the highest confidence
        current = predictions[0]
        keep.append(current)
        # Calculate distances between the current point and the rest
        distances = np.sqrt(((predictions[:, :3] - current[:3]) ** 2).sum(axis=1))
        # Suppress points that are too close
        predictions = predictions[distances > distance_threshold]
    return np.array(keep)

def mAP(p_y, t_y, conf_thres=0.5, distance_thres=10.0, stride=np.array([512, 512, 416])):
    t_y_abs = []
    bs = t_y.shape[0]
    # print(t_y)
    for i in range(bs):
        t = t_y[i]
        temp = []
        for p in t:
            if p[0] != -1:
                temp.append(p*stride)
        t_y_abs.append(temp)
    t_y = t_y_abs
    tp = 0
    fp = 0
    fn = 0
    for pred_img, target_img in zip(p_y, t_y):
        # Apply NMS on the predicted points for the current image
        keeps = nms_3d(pred_img, conf_thres, distance_thres)
        # For each target, check if there is a corresponding prediction
        for target in target_img:
            matched = False
            for pred in keeps:
                # Calculate Euclidean distance between the predicted point and the target point
                dist = np.sqrt(np.sum((pred[:3] - target[:3]) ** 2))
                if dist <= distance_thres:
                    tp += 1
                    matched = True
                    break
            if not matched:
                fn += 1  # If no match, it's a False Negative
        fp += len(keeps) - tp  # False Positives are the remaining predictions after matching with targets
    # print(f'tp: {tp}, fp: {fp}, fn: {fn}')
    # Calculate precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    # Average Precision (AP) is typically the area under the precision-recall curve
    # Here we assume precision-recall is linearly increasing, so AP = precision (a simple approximation)
    ap = precision

    # Return mean Average Precision (mAP)
    return {'tp': tp, 'fp': fp, 'fn': fn, 'ap': ap, 'recall': recall}


import numpy as np


def nms_2d(predictions, confidence_threshold=0.5, distance_threshold=20.0):
    """2D 非极大值抑制"""
    # 过滤低置信度预测
    predictions = predictions[predictions[:, 2] > confidence_threshold]
    if len(predictions) == 0:
        return np.array([])

    # 按置信度降序排序 [y, x, confidence]
    predictions = predictions[np.argsort(predictions[:, 2])[::-1]]

    keep = []
    while len(predictions) > 0:
        current = predictions[0]
        keep.append(current)

        # 计算二维欧氏距离
        distances = np.sqrt(((predictions[:, :2] - current[:2]) ** 2).sum(axis=1))

        # 保留距离大于阈值的预测
        predictions = predictions[distances > distance_threshold]

    return np.array(keep)


def mAP_2d(p_y, t_y, conf_thres=0.5, distance_thres=20.0, stride=np.array([512, 512])):
    """2D 平均精度计算"""
    # 转换真实坐标为绝对坐标
    t_y_abs = []
    bs = t_y.shape[0]
    for i in range(bs):
        t = t_y[i]
        temp = []
        for p in t:
            if p[0] > 0:  # 假设用-1表示无效标记
                # temp.append(p * stride)  # 二维坐标转换
                temp.append(torch.cat((p, torch.tensor([24, 24]))))
        t_y_abs.append(temp)

    t_y = t_y_abs
    tp = 0
    fp = 0
    fn = 0

    for pred_img, target_img in zip(p_y, t_y):
        # 应用NMS
        keeps = nms_2d(pred_img, conf_thres, distance_thres)

        # ious = bbox_iou(t_y, keeps)

        # 匹配预测和真实目标
        matched = set()
        for target in target_img:
            min_dist = float('inf')
            best_idx = -1

            # 寻找最近邻预测
            for idx, pred in enumerate(keeps):
                dist = np.sqrt(np.sum((pred[:2] - target[:2]) ** 2))
                if dist <= distance_thres and dist < min_dist and idx not in matched:
                    min_dist = dist
                    best_idx = idx

            if best_idx != -1:
                tp += 1
                matched.add(best_idx)
            else:
                fn += 1

        # 未匹配的预测视为FP
        fp += len(keeps) - len(matched)

    # 计算指标
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # 简单AP计算（实际应计算PR曲线下面积）
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'ap': precision  # 简化版AP
    }

def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Calculates IoU, GIoU, DIoU, or CIoU between two boxes, supporting xywh/xyxy formats.

    Input shapes are box1(1,4) to box2(n,4).
    """
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
        w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw**2 + ch**2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU

def nms(pred_boxes, pred_scores, iou_threshold=0.5, score_threshold=0.5):
    """
    Args:
        pred_boxes (Tensor): Tensor of predicted bounding boxes, shape (N, 4) where N is the number of boxes.
        pred_scores (Tensor): Tensor of predicted scores, shape (N,) for each box.
        iou_threshold (float): The IoU threshold for deciding when to suppress a box.
        score_threshold (float): The score threshold to filter out low confidence boxes.
    Returns:
        selected_boxes (Tensor): The indices of the selected boxes after NMS.
        selected_scores (Tensor): The corresponding scores for the selected boxes.
    """
    # Filter out low confidence boxes
    pred_scores = pred_scores.flatten()
    mask = pred_scores >= score_threshold
    mask = mask.squeeze()
    # print('mmm', mask.shape, pred_scores.shape)
    pred_scores = pred_scores[mask]
    pred_boxes = pred_boxes[mask]

    if len(pred_boxes) == 0:
        return torch.tensor([]), torch.tensor([])

    # Sort the boxes by score in descending order
    sorted_scores, sorted_indices = torch.sort(pred_scores.squeeze(), descending=True)
    sorted_indices = sorted_indices.squeeze()  # 确保一维
    pred_boxes = pred_boxes[sorted_indices]
    # print('p', pred_boxes, sorted_indices, pred_scores.shape, sorted_scores)
    # pred_boxes = pred_boxes[sorted_indices]
    # print('p', pred_boxes.shape)
    keep = []
    while len(pred_boxes) > 0:
        # Select the box with the highest score and add it to the kept list
        # best_box = pred_boxes[0]
        # keep.append(best_box)
        best_idx = sorted_indices[0]
        keep.append(best_idx)
        # print(best_box, pred_boxes[1:])
        if len(pred_boxes) == 1:
            break
        # Compute IoU between the best box and the rest of the boxes
        iou = bbox_iou(pred_boxes[0].unsqueeze(0), pred_boxes[1:])  # Compute IoU for remaining boxes
        mask = iou.squeeze() <= iou_threshold
        mask = mask.squeeze()
        pred_boxes = pred_boxes[1:][mask]
        sorted_indices = sorted_indices[1:][mask].squeeze()
        # print(f'sorted_indices.shape: {sorted_indices.shape}')
    # print('kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk', keep)
    return torch.stack(keep), sorted_scores[:len(keep)]

def precision(pred_labels, true_labels):
    """Compute precision for each class."""
    tp = (pred_labels == true_labels) & (pred_labels == 1)  # True positives
    fp = (pred_labels != true_labels) & (pred_labels == 1)  # False positives
    precision = tp.sum() / (tp.sum() + fp.sum()) if tp.sum() + fp.sum() > 0 else 0
    return precision

def recall(pred_labels, true_labels):
    """Compute recall for each class."""
    tp = (pred_labels == true_labels) & (pred_labels == 1)  # True positives
    fn = (pred_labels != true_labels) & (true_labels == 1)  # False negatives
    recall = tp.sum() / (tp.sum() + fn.sum()) if tp.sum() + fn.sum() > 0 else 0
    return recall

def f1_score(pred_labels, true_labels):
    """Compute F1 score."""
    p = precision(pred_labels, true_labels)
    r = recall(pred_labels, true_labels)
    return 2 * (p * r) / (p + r) if p + r > 0 else 0

def average_precision(pred_boxes, pred_scores, true_boxes, iou_threshold=0.5):
    """
    Compute the Average Precision (AP) for a class.

    Args:
        pred_boxes (Tensor): Predicted bounding boxes.
        pred_scores (Tensor): Predicted scores.
        true_boxes (Tensor): Ground truth boxes.
        iou_threshold (float): IoU threshold for determining if a prediction is correct.

    Returns:
        ap (float): Average precision for the class.
    """
    # Sort predictions by score
    sorted_scores, sorted_indices = torch.sort(pred_scores, descending=True)
    pred_boxes = pred_boxes[sorted_indices]

    tp = torch.zeros(len(pred_boxes))
    fp = torch.zeros(len(pred_boxes))

    for i, box in enumerate(pred_boxes):
        iou = bbox_iou(box.unsqueeze(0), true_boxes)
        if iou.max() >= iou_threshold:
            tp[i] = 1
        else:
            fp[i] = 1

    # Calculate precision and recall at each threshold
    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)

    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-9)
    recall = tp_cumsum / (len(true_boxes) + 1e-9)
    print(precision, recall, torch.trapz(precision, recall))

    # Compute AP as the area under the precision-recall curve
    ap = torch.trapz(precision, recall)  # Area under precision-recall curve
    return ap

def ca_metrics(predictions, gts, iou_threshold=0.5, confidence_threshold=0.5):
    pred_boxes, pred_labels = predictions.split((2, 1), -1)
    # 将中心点坐标转换为完整边界框 (xywh格式)
    def expand_boxes(boxes):
        """将中心坐标转换为固定宽高的完整边界框"""
        if isinstance(boxes, torch.Tensor):
            expanded = torch.zeros((*boxes.shape[:-1], 4), device=boxes.device)
        else:
            expanded = np.zeros((*boxes.shape[:-1], 4))
        expanded[..., :2] = boxes[..., :2]  # 中心坐标x, y
        expanded[..., 2] = 24  # 宽度w
        expanded[..., 3] = 24  # 高度h
        return expanded
    pred_boxes = expand_boxes(pred_boxes)  # [b, 1, n_pred, 4] (x,y,w,h)
    # true_boxes = expand_boxes(true_boxes)  # [b, 1, n_gt, 4] (x,y,w,h)
    # print(pred_boxes.shape, true_boxes.shape)
    total_tp = 0
    total_fp = 0
    total_fn = 0
    aps = []
    for b in range(pred_boxes.shape[0]):
        batch_pred_boxes = pred_boxes[b]  # [n_pred, 4]
        batch_true_boxes = gts[b]   # [n_gt, 4]
        batch_true_boxes = expand_boxes(batch_true_boxes[batch_true_boxes[..., -1][0] > 0])
        # print(batch_true_boxes)
        batch_pred_labels = pred_labels[b]    # [n_pred]
        batch_true_labels = torch.ones_like(batch_pred_labels)     # [n_gt]
        # print(batch_pred_boxes.shape, batch_true_boxes.shape)
        # pred_scores = pred_labels[b][:, 1]
        # selected_indexs, selected_scores = nms(batch_pred_boxes, batch_pred_labels, iou_threshold=0.5)
        keep_indices = ops.nms(xywh2xyxy(batch_pred_boxes), batch_pred_labels.squeeze(), iou_threshold)

        # print(batch_pred_boxes)
        det_labels = (batch_pred_labels > 0.5).int()
        det_boxes = batch_pred_boxes[keep_indices]
        gt_matched = np.zeros(len(batch_true_boxes), dtype=bool)
        det_matched = np.zeros(len(det_boxes), dtype=bool)
        for i in range(len(det_boxes)):
            det_box = det_boxes[i]
            # det_label = det_labels[i]
            ious = bbox_iou(det_box, batch_true_boxes)
            # print(ious)
            best_iou, best_gt_idx = ious.max(dim=0)
            # print(best_iou >= iou_threshold)
            # if (best_iou >= iou_threshold and not gt_matched[best_gt_idx] and det_labels == batch_true_labels[best_gt_idx]):
            if (best_iou >= iou_threshold and not gt_matched[best_gt_idx] and det_labels[i] == batch_true_labels[best_gt_idx]):
                total_tp += 1
                gt_matched[best_gt_idx] = True
                det_matched[i] = True
            else:
                total_fp += 1

        # 统计漏检（FN）
        total_fn += np.sum(~gt_matched)

        # -------------------- 计算AP --------------------
        # 需要累积所有预测的置信度和匹配结果（此处简化计算）
        precision = total_tp / (total_tp + total_fp + 1e-9)
        recall = total_tp / (total_tp + total_fn + 1e-9)
        aps.append(precision)  # 简化版AP计算，实际应使用P-R曲线下面积

    metrics = {
        'TP': total_tp,
        'FP': total_fp,
        'FN': total_fn,
        'Precision': total_tp / (total_tp + total_fp + 1e-9),
        'Recall': total_tp / (total_tp + total_fn + 1e-9),
        'F1': 2 * total_tp / (2 * total_tp + total_fp + total_fn + 1e-9),
        'AP': np.mean(aps)
    }
    return metrics





if __name__ == '__main__':

    pred = torch.tensor([[[0.25, 0.25, 0.6], [0.45, 0.45, 0.8]]])  # 预测框
    # pred_scores = torch.tensor([[0.9]])  # 预测框的置信度
    # pred = torch.cat((pred_boxes, pred_scores), dim=1)
    true_boxes = torch.tensor([[[0.25, 0.25]]])  # 真实框
    true_labels = torch.tensor([[1]])  # 真实标签
    print(ca_metrics(pred, true_boxes))
    # # NMS后保留的框
    # selected_indices, selected_scores = nms(pred_boxes, pred_scores, iou_threshold=0.5)
    # print(selected_scores)
    #
    # # Precision, Recall, F1计算
    # precision_score = precision(true_labels, torch.tensor([1]))  # 示例预测标签
    # recall_score = recall(true_labels, torch.tensor([1]))
    # f1_score_value = f1_score(true_labels, torch.tensor([1]))
    #
    # # AP计算
    # ap = average_precision(pred_boxes, pred_scores, true_boxes)
    #
    # print(f"Precision: {precision_score}")
    # print(f"Recall: {recall_score}")
    # print(f"F1 Score: {f1_score_value}")
    # print(f"AP: {ap}")
