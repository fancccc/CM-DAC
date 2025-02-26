# -*- coding: utf-8 -*-
'''
@file: dataloader.py
@author: fanc
@time: 2024/12/16 20:27
'''
from torch.utils.data import Dataset, DataLoader
# import pandas as pd
import os
import json
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import pandas as pd

class LUNA16Dataset(Dataset):
    def __init__(self, root, data, target_size=(512, 512, 512)):
        super(LUNA16Dataset, self).__init__()
        self.root = root
        self.target_size = target_size
        self.pad_value = -1024
        with open(os.path.join(self.root, data), 'r') as f:
            self.ann = json.load(f)
        self.max_num = max([len(i['points']) for i in self.ann])

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, i):
        ann = self.ann[i]
        img_path = os.path.join(self.root, 'resample', ann['suid'] + '.nii.gz')
        img = sitk.ReadImage(img_path)
        img = sitk.GetArrayFromImage(img).transpose((1, 2, 0))
        img = torch.tensor(img, dtype=torch.float)

        current_height, current_width, current_depth = img.shape
        pad_height = self.target_size[0] - current_height
        pad_width = self.target_size[1] - current_width
        pad_depth = self.target_size[2] - current_depth

        if pad_depth < 0 or pad_height < 0 or pad_width < 0:
            raise ValueError(f"img shape{img.shape}, target shape{self.target_size}")

        img = F.pad(img, (0, pad_depth, 0, pad_width, 0, pad_height), value=self.pad_value)
        img = self.window_level(img)
        img = img.unsqueeze(0)

        points = ann['points']
        relative_coords = torch.full((self.max_num, 3), -1, dtype=torch.float)
        for i, point in enumerate(points):
            relative_coord = [point[j] / self.target_size[j] for j in range(3)]
            relative_coords[i] = torch.tensor(relative_coord)

        return {'img': img, 'relative_coord': relative_coords}

    def window_level(self, img, window_min=-400, window_max=1000):
        img[img < window_min] = window_min
        img[img > window_max] = window_max
        return img


class CSPL2dDetDataset(Dataset):
    def __init__(self, root, data, target_size=(512, 512)):
        super(CSPL2dDetDataset, self).__init__()
        self.root = root
        self.labels = {}
        lb = pd.read_excel(os.path.join(root, 'VOC_CT.xlsx'))
        for _ in lb.index:
            temp = lb.loc[_]
            self.labels[temp['id']] = [temp['xmin'], temp['ymin'], temp['xmax'],temp['ymax']]
        # for filename in os.listdir(os.path.join(self.root, 'labels')):
        #     with open(os.path.join(os.path.join(self.root, 'labels', filename)), 'r') as f:
        #         self.labels[filename.strip('.txt')] = [float(i) for i in f.read().strip().split()[1:3]]
        with open(os.path.join(self.root, data), 'r') as f:
            self.images = f.read().strip().split()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        id_ = self.images[i]
        img_path = os.path.join(self.root, 'JPEGImages', f'{id_}.bmp')
        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = np.transpose(img, (2, 0, 1))
        img = sitk.ReadImage(img_path)
        img = sitk.GetArrayFromImage(img) / 255.
        img = torch.tensor(img, dtype=torch.float)
        img = np.transpose(img, (2, 0, 1))
        # print(img.shape)
        # img = img.unsqueeze(0)
        coordinate = torch.tensor(self.labels[id_])
        points = [(coordinate[0] + coordinate[2]) / 2 / 512, (coordinate[1] + coordinate[3]) / 2 / 512]
        points = torch.tensor(points, dtype=torch.float).unsqueeze(0)
        # print(points)
        # relative_coords = torch.full((self.max_num, 3), -1, dtype=torch.float)
        # for i, point in enumerate(points):
        #     relative_coord = [point[j] / self.target_size[j] for j in range(3)]
        #     relative_coords[i] = torch.tensor(relative_coord)

        return {'img': img, 'relative_coord': points, 'coordinate': coordinate}


class CSPL3dDataset(Dataset):
    def __init__(self, root, data, target_size=(512, 512, -1)):
        super(CSPL3dDataset, self).__init__()
        self.root = root
        self.target_size = target_size
        self.pad_value = -2048
        self.ann = pd.read_csv(os.path.join(self.root, data))
        self.ann['size'] = self.ann['size'].apply(lambda x: eval(x) if type(x) == str else x)
        self.ann['xyz'] = self.ann['xyz'].apply(lambda x: eval(x) if type(x) == str else x)
        self.ann = self.ann[self.ann['size'].apply(lambda x: x[-1] <= 416)]
        self.uid = self.ann['seriesuid'].unique()
        self.labels = []
        for uid, g in self.ann.groupby('seriesuid'):
            temp = {'uid': uid, 'points': g['xyz'].tolist()}
            self.labels.append(temp)
        self.max_num = max([len(i['points']) for i in self.labels])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        # ann = self.ann[i]
        ann = self.labels[i]
        img_path = os.path.join(self.root, 'resample', ann['uid'] + '.nii.gz')
        img = sitk.ReadImage(img_path)
        img = sitk.GetArrayFromImage(img).transpose((1, 2, 0))
        img = torch.tensor(img, dtype=torch.float)

        current_height, current_width, current_depth = img.shape

        # target_size = (current_height // 32 + 1) * 32 if current_height % 32 != 0 else current_height
        # self.target_size = (target_size, target_size, self.target_size[2])

        pad_height = self.target_size[0] - current_height
        pad_width = self.target_size[1] - current_width
        pad_depth = self.target_size[2] - current_depth
        if self.target_size[-1] == -1:
            pad_depth = current_depth

        if pad_depth < 0 or pad_height < 0 or pad_width < 0:
            raise ValueError(f"img shape{img.shape}, target shape{self.target_size}")

        img = F.pad(img, (0, pad_depth, 0, pad_width, 0, pad_height), value=self.pad_value)
        img = self.window_level(img)
        img = img.unsqueeze(0)

        # points = ann['points']
        # points = torch.full((self.max_num, 3), -1, dtype=torch.int32)
        points = torch.zeros((current_depth, 1, 2))
        # for i in range(current_depth):
        #     points[i] = ann['points']
        # relative_coords = torch.full((self.max_num, 3), -1, dtype=torch.float)
        for i, point in enumerate(ann['points']):
        #     relative_coord = [point[j] / self.target_size[j] for j in range(3)]
            d = point[-1]
            points[d] = torch.tensor(point[:2])

        return {'img': img, 'points': points}

    def window_level(self, img, WL=-600, WW=1500):
        MAX = WL + WW / 2
        MIN = WL - WW / 2
        img[img < MIN] = MIN
        img[img > MAX] = MAX
        img = (img - MIN) / WW
        return img



if __name__ == '__main__':
    ## luna16
    # import matplotlib.pyplot as plt
    # root = '/data6-home/fanchenchenzc/datasets/LUNA16'
    # dataset = LUNA16Dataset(root, 'test.json')
    # dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    # for i, data in enumerate(dataloader):
    #     print(data['img'].shape, data['relative_coord'].shape)
    #     img = data['img'].numpy()
    #     coord = data['relative_coord']
    #     print(coord)
    #
    #     break
    # plt.imshow(data['img'][0, 0, :, :, 0])
    # plt.show()

    ## 2d detection
    root = r'F:\fanc\datasets\CSPL\1.25mm_2D_detection'
    data = os.path.join('ImageSets', 'Main', 'train.txt')
    dataset = CSPL2dDetDataset(root, data)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for i, data in enumerate(dataloader):
        print(data['img'].shape, data['relative_coord'].shape, data['coordinate'])

        break




