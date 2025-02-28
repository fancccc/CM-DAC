# -*- coding: utf-8 -*-
'''
@file: RPN.py
@author: author
@time: 2025/1/8 16:00
'''

import torch.nn as nn

class RPN3D(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super(RPN3D, self).__init__()

        # 1x1x1 Convolution to project the input feature map to a new channel dimension
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

        # Predict the 3D coordinates for the points (x, y, z)
        # For each voxel (point), we predict its 3D coordinates relative to the anchor or the grid.
        self.coord_predictor = nn.Conv3d(out_channels, 3, kernel_size=1)  # Output: 3D coordinates (x, y, z)

        # Predict the score of each point (objectness)
        self.score_predictor = nn.Conv3d(out_channels, 1, kernel_size=1)  # Output: score (foreground/background)

    def forward(self, x):
        """
        :param x: Feature map after FPN, expected to be of shape (B, C, D, H, W)
        :return: Predicted 3D coordinates and object scores
        """
        x = self.conv1(x)
        x = self.relu(x)

        # Predict 3D coordinates (x, y, z)
        coords = self.coord_predictor(x)  # Shape: (B, 3, D, H, W)

        # Predict scores (foreground vs background)
        scores = self.score_predictor(x)  # Shape: (B, 1, D, H, W)

        return coords, scores


class RPN3DPostProcessing(nn.Module):
    def __init__(self, threshold=0.5):
        super(RPN3DPostProcessing, self).__init__()
        self.threshold = threshold

    def forward(self, coords, scores):
        """
        Post-processing to filter the points based on score threshold.
        :param coords: Predicted 3D coordinates (x, y, z)
        :param scores: Predicted objectness scores
        :return: Filtered 3D points based on score threshold
        """
        batch_size, _, D, H, W = scores.shape
        points = []

        for b in range(batch_size):
            for d in range(D):
                for h in range(H):
                    for w in range(W):
                        if scores[b, 0, d, h, w] > self.threshold:
                            # Collect coordinates (x, y, z)
                            point = coords[b, :, d, h, w].cpu().numpy()
                            points.append(point)

        return points