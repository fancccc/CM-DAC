# -*- coding: utf-8 -*-
'''
@file: v11.py
@author: fanc
@time: 2025/2/22 下午10:03
'''
from ultralytics import YOLO
model = YOLO("./yolo11x.pt")  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
results = model.train(data="./datalc1.yaml", epochs=400, imgsz=640)