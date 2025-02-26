# -*- coding: utf-8 -*-
'''
@file: detection.py
@author: fanc
@time: 2025/2/17
'''
from detectron2.modeling import build_model
from detectron2.config import get_cfg
import torch
from detectron2.utils.events import EventStorage
import os
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer
def CSLP2D(phase='train'):
    res = []
    root = '/zhangyongquan/fanc/datasets/partA/1.25mm_2D_detection/'
    with open(os.path.join(root, 'ImageSets', f'{phase}.txt'), 'r') as f:
        files = f.read().strip().split()
    for i, fn in enumerate(files):
        temp = {'file_name': os.path.join(root, 'JPEGImages', f'{fn}.bmp'),
               'height': 512, 'width': 512, 'image_id': i
               }
        with open(os.path.join(root, 'labels', f'{fn}.txt'), 'r') as f:
            data = f.read().strip().split()
            data = [float(i) for i in data]
        annotations = [{'bbox': [data[1]*512, data[2]*512, data[3]*512, data[4]*512],
                       'bbox_mode': BoxMode.XYWH_ABS, 'category_id': 0,
                       'keypoints': [data[1]*512, data[1]*512, 1]}]
        temp['annotations'] = annotations
        res.append(temp)
    return res
DatasetCatalog.register("CSLP2D_train", lambda: CSLP2D(phase='train'))
DatasetCatalog.register("CSLP2D_val", lambda: CSLP2D(phase='val'))
# data = DatasetCatalog.get("CSLP2D_train")
MetadataCatalog.get("CSLP2D_train").thing_classes = ["nodule"]
MetadataCatalog.get("CSLP2D_train").keypoint_names = ['nodulecenter']
MetadataCatalog.get("CSLP2D_val").thing_classes = ["nodule"]
MetadataCatalog.get("CSLP2D_val").keypoint_names = ['nodulecenter']

cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.merge_from_file("../../configs/COCO-Detection/retinanet_R_50_FPN_1x.yaml")
cfg.DATASETS.TRAIN = ("CSLP2D_train",)
cfg.DATASETS.TEST = ("CSLP2D_val")
cfg.DATALOADER.NUM_WORKERS = 2
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
# cfg.SOLVER.MAX_ITER = 10000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
cfg.OUTPUT_DIR = '../results/detectron'
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()