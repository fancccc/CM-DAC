# -*- coding: utf-8 -*-
'''
@file: data.py
@author: fanc
@time: 2025/2/19 下午3:48
'''
# from detectron2.modeling import build_model
# from detectron2.config import get_cfg
# from detectron2.utils.events import EventStorage
from detectron2.structures import BoxMode
# from detectron2.data import DatasetCatalog
# from detectron2.data import MetadataCatalog
# from detectron2.engine import DefaultTrainer
import re
import os
import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset
import json
import SimpleITK as sitk


def LUAD2_2D(phase='train'):
    res = []
    root = '/zhangyongquan/fanc/datasets/LungCancer2'
    ann = pd.read_csv(os.path.join(root, f'{phase}.csv'))
    ann['bbox'] = ann['bbox'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    fcolumns = list(filter(lambda x: re.search('^f', x), list(ann.columns)))
    # files = os.listdir(os.path.join(root, '2d'))
    image_id = 0
    for i in range(len(ann)):
        temp = ann.iloc[i]
        clinical = torch.tensor(temp[fcolumns].tolist(), dtype=torch.float32)
        # print(temp['bbox'])
        bbox = temp['bbox']
        xyxy = [bbox[0], bbox[1], bbox[0]+bbox[3], bbox[1]+bbox[4]]
        label = int(temp['label'])
        label = 0
        annotations = [{'bbox': xyxy,
                       'bbox_mode': BoxMode.XYXY_ABS, 'category_id': label}]

        for j in [-2, -1, 0, 1, 2]:
            info = {'file_name': os.path.join(root, '2d', temp['bid'] + f'_{j}.tiff'), 'clinical': clinical,
                       'height': 512, 'width': 512, 'image_id': image_id, 'annotations': annotations}
            res.append(info)
    return res

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

class LUAD2_3D(Dataset):
    def __init__(self, root, phase='train'):
        df = pd.read_csv(os.path.join(root, f'{phase}.csv'))
        self.root = root
        self.bids = df['bid'].tolist()
        self.phase = phase
        self.use_flip = 'train' in phase
        self.labels = torch.tensor(df['label'].apply(lambda x: 2 if x == 3 else x).tolist(), dtype=torch.long)
        # if self.use_slice:
        #     self.ts = transforms.ToTensor()
        cols = list(filter(lambda x: x.startswith('f'),  df.columns.tolist()))
        self.clinical = df[cols].fillna(0)
        # self.bbox = bbox
        self.use_bbox = False

    # @property
    # def getlabels(self):
    #     return [self.labels[i] for i in range(len(self))]

    def __len__(self):
        return len(self.bids)

    def __getitem__(self, i):
        # ct, mask, clinical, bbox = torch.zeros((0,)), torch.zeros((0,)), torch.zeros((0,)), torch.zeros((0,))
        # ct, mask, clinical, bbox, slice, ct64, ct128, ct256, seg, radiomic = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        # bbox32, bbox128 = 0, 0
        bid = self.bids[i]
        label = self.labels[i]
        ct = self.get_ct_32(i)
        clinical = self.get_clinical(i)
        if self.use_flip:
            ct, _ = self.random_flip(ct)
        res = {'label': label, 'bid': bid, 'ct': ct, 'clinical': clinical}
        return res

    def get_ct_32(self, i):
        # if self.phase in ['train', 'val', 'all']:
        ct_path = os.path.join(self.root, 'cropped', '32sitk', f'{self.bids[i]}.nii.gz')
        # else:
            # ct_path = os.path.join()
            # pass
        # self.get_nii_file(ct_path)
        return self.get_nii_file(ct_path)

    def get_clinical(self, i):
        values = torch.tensor(self.clinical.iloc[i].values.tolist(), dtype=torch.float)
        return values#.unsqueeze(-1)

    def get_nii_file(self, file, normalize=True):
        if file.endswith('.nii.gz'):
            img = sitk.ReadImage(file)
            img = sitk.GetArrayFromImage(img).transpose(1, 2, 0)
        elif file.endswith('.npy'):
            img = np.load(file)
        if normalize:
            img = self.normalize(img)
        return torch.tensor(img, dtype=torch.float32).unsqueeze(0)

    def random_flip(self, img, bbox=None):
        # 随机选择翻转轴
        flip_x = random.choice([True, False])
        flip_y = random.choice([True, False])
        flip_z = random.choice([True, False])

        # 对图像进行翻转
        if flip_x:
            img = torch.flip(img, dims=[2])  # 翻转X轴 (第3个维度，index为2)
            if self.use_bbox:
                bbox[0] = 1 - bbox[3] - bbox[0]  # 反转X坐标 (标准化坐标，无需图像尺寸)

        if flip_y:
            img = torch.flip(img, dims=[1])  # 翻转Y轴 (第2个维度，index为1)
            if self.use_bbox:
                bbox[1] = 1 - bbox[4] - bbox[1]  # 反转Y坐标

        if flip_z:
            img = torch.flip(img, dims=[0])  # 翻转Z轴 (第1个维度，index为0)
            if self.use_bbox:
                bbox[2] = 1 - bbox[5] - bbox[2]  # 反转Z坐标
        return img, bbox

    def normalize(self, img, WL=-600, WW=1500):
        MAX = WL + WW / 2
        MIN = WL - WW / 2
        img[img < MIN] = MIN
        img[img > MAX] = MAX
        img = (img - MIN) / WW
        return img

if __name__ == '__main__':
    # full_dataset = LUAD2_3D(root='/zhangyongquan/fanc/datasets/LC1', phase='all')
    # full_labels = np.array([data['label'] for data in full_dataset])
    # root = args.root
    # full_dataset = LUAD2_3D(root, phase='all')
    # full_labels = np.array([d['label'] for d in full_dataset])
    # from sklearn.model_selection import StratifiedKFold
    # from torch.utils.data import Subset
    # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1900)
    # best = {0:0, 1:0, 2:0, 3:0, 4:0}
    # # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(full_labels)), full_labels)):
    #     traindataset = Subset(full_dataset, train_idx)
    #     testdataset = Subset(full_dataset, val_idx)
    #
    #     # 打印分布对比
    #     print(f"\nFold {fold + 1} Class Distribution:")
    #     print("Full dataset:", np.bincount(full_labels))
    #     print("Train subset:", np.bincount(full_labels[train_idx]))
    #     print("Val subset:  ", np.bincount(full_labels[val_idx]))
    # print(labels)
    dataset = LUAD2_3D(root='/zhangyongquan/fanc/datasets/LPCD', phase='train')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for i, batch in enumerate(dataloader):
        print(batch['bid'], batch['ct'].shape, batch['clinical'].shape)
    # pass
    # from detectron2.data import build_detection_train_loader, DatasetMapper
    # from detectron2.data import DatasetCatalog
    # from detectron2.data import MetadataCatalog
    # DatasetCatalog.register("CSLP2D_val", lambda: CSLP2D(phase='val'))
    # MetadataCatalog.get("CSLP2D_val").thing_classes = ["nodule"]
    # data = DatasetCatalog.get("CSLP2D_val")
    # dataloader = build_detection_train_loader(data, mapper=DatasetMapper(cfg, is_train=False), total_batch_size=4)
    # data = LUAD2_2D(phase='val')
    # print(data[0]['image'].shape)