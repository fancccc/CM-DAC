# -*- coding: utf-8 -*-
'''
@file: build_net.py
@author: author
@time: 2025/2/18 下午4:03
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
from data import LUAD2_2D, CSLP2D
from detectron2.data import build_detection_train_loader, DatasetMapper

from detectron2.modeling import build_backbone, RetinaNet, detector_postprocess
import torch.nn as nn
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.config import CfgNode as CN
from detectron2.evaluation import COCOEvaluator

from torch.nn import MultiheadAttention
import torch.nn.functional as F
# 定义额外输入字段
def add_extra_input(cfg):
    cfg.MODEL.EXTRA_INPUT = CN()
    cfg.MODEL.EXTRA_INPUT.DIM = 1  # 默认为禁用
    # cfg.MODEL.EXTRA_INPUT.SIZE = [256, 256]  # 可以根据需要调整输入大小

@META_ARCH_REGISTRY.register()
class RetinaNetEx(RetinaNet):
    def __init__(self, cfg):
        super().__init__(cfg)
        in_channels = cfg.MODEL.FPN.OUT_CHANNELS

        # 调整临床编码器输出维度
        self.clinical_encoder = nn.Sequential(
            nn.Linear(cfg.MODEL.EXTRA_INPUT.DIM, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 256 * 5)  # 直接生成各层特征向量
        )


        # 修正注意力模块结构
        self.attention_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1),  # 通道数保持不变
                nn.MultiheadAttention(
                    embed_dim=256,  # 与输入通道一致
                    num_heads=8,
                    kdim=256,
                    vdim=256
                )
            ) for _ in range(5)
        ])

        # 添加临床特征适配器
        self.clin_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 256, kernel_size=1),  # 通道转换
                nn.AdaptiveAvgPool2d((1, 1))  # 确保空间维度为1x1
            ) for _ in range(5)
        ])

    def forward(self, batched_inputs):
        # 图像特征处理（保持原逻辑）
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.head_in_features]

        # 临床特征处理
        clinical_data = torch.stack([x["clinical"] for x in batched_inputs])
        clinical_data = clinical_data.to(features[0].device)
        clinical_features = self.clinical_encoder(clinical_data).view(-1, 5, 256)

        # 特征融合
        fused_features = []
        for i, (feat, att_layer, clin_adapter) in enumerate(zip(features, self.attention_layers, self.clin_adapters)):
            B, C, H, W = feat.shape
            # 临床特征处理
            clin_feat = clinical_features[:,i].view(B,1,1,1)  # (B,1,1,1)
            clin_feat = clin_adapter(clin_feat)               # (B,256,1,1)
            clin_feat = clin_feat.expand(-1, -1, H, W)        # (B,256,H,W)

            # 特征融合
            fused = feat + clin_feat

            # 注意力机制
            conv_feat = att_layer[0](fused)  # (B,256,H,W)
            attn_input = conv_feat.flatten(2).permute(2, 0, 1)  # (H*W, B, 256)
            attn_out, _ = att_layer[1](attn_input, attn_input, attn_input)
            attn_out = attn_out.permute(1, 2, 0).view_as(fused)

            fused_features.append(attn_out)

            # 残差融合
            # fused = adapter(feat + attn_out.permute(1, 2, 0).view_as(feat))
            # fused_features.append(fused)
        # print(extra_features.shape, features[0].shape)

        # res.append(self.att0(extra_features, features[0].flatten(start_dim=2), features[0].flatten(start_dim=2))[0].view(features[0].shape))
        # extra_features = self.down_sample(extra_features)
        # res.append(self.att1(extra_features, features[1].flatten(start_dim=2), features[1].flatten(start_dim=2))[0].view(features[1].shape))
        # extra_features = self.down_sample(extra_features)
        # res.append(self.att2(extra_features, features[2].flatten(start_dim=2), features[2].flatten(start_dim=2))[0].view(features[2].shape))
        # extra_features = self.down_sample(extra_features)
        # res.append(self.att3(extra_features, features[3].flatten(start_dim=2), features[3].flatten(start_dim=2))[0].view(features[3].shape))
        # extra_features = self.down_sample(extra_features)
        # res.append(self.att4(extra_features, features[4].flatten(start_dim=2), features[4].flatten(start_dim=2))[0].view(features[4].shape))

        predictions = self.head(features)

        if self.training:
            assert not torch.jit.is_scripting(), "Not supported"
            assert "instances" in batched_inputs[0], "Instance annotations are missing in training!"
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            return self.forward_training(images, features, predictions, gt_instances)
        else:
            results = self.forward_inference(images, features, predictions)
            if torch.jit.is_scripting():
                return results

            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(
            dataset_name,
            output_dir=os.path.join(cfg.OUTPUT_DIR, "evaluation"),
        )
if __name__ == '__main__':
    cfg = get_cfg()
    add_extra_input(cfg)
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # cfg.merge_from_file("../../configs/COCO-Detection/retinanet_R_50_FPN_1x.yaml")
    cfg.merge_from_file("../../configs/retinanet_R_50_FPN_1x_extra.yaml")
    DatasetCatalog.register("LUAD2_2D_TRAIN", lambda: LUAD2_2D(phase='train'))
    DatasetCatalog.register("LUAD2_2D_VAL", lambda: LUAD2_2D(phase='val'))
    # data = DatasetCatalog.get("CSLP2D_train")
    MetadataCatalog.get("LUAD2_2D_TRAIN").thing_classes = ["0"]
    # MetadataCatalog.get("CSLP2D_train").keypoint_names = ['nodulecenter']
    MetadataCatalog.get("LUAD2_2D_VAL").thing_classes = ["0"]

    cfg.DATASETS.TRAIN = ("LUAD2_2D_TRAIN",)
    cfg.DATASETS.TEST = ("LUAD2_2D_VAL",)
    cfg.DATALOADER.NUM_WORKERS = 6
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.001  # pick a good LR
    # cfg.SOLVER.MAX_ITER = 10000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    cfg.OUTPUT_DIR = '../results/test'
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.MODEL.META_ARCHITECTURE = "RetinaNetEx"
    cfg.SOLVER.MAX_ITER = 1
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    # print(cfg)
    trainer.train()
    # MetadataCatalog.get("LUAD2_2D_VAL").keypoint_names = ['nodulecenter']
    # train_data = DatasetCatalog.get("LUAD2_2D_VAL")
    # train_loader = build_detection_train_loader(train_data, mapper=DatasetMapper(cfg, is_train=True), total_batch_size=36)
    # val_data = DatasetCatalog.get("LUAD2_2D_VAL")
    # val_loader = build_detection_train_loader(train_data, mapper=DatasetMapper(cfg, is_train=True), total_batch_size=36)
    # # for batch in dataloader:
    # #     print(batch)
    # #     break
    # # print(model(batch))
    # model = build_model(cfg)  # returns a torch.nn.Module
    # # print(model)
    # with EventStorage() as storage:
    #     for data in dataloader:
    #         outputs = model(data)
    #         break
    # print(outputs)
    # cfg = get_cfg()
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # cfg.merge_from_file("../../configs/COCO-Detection/retinanet_R_50_FPN_1x.yaml")
    # cfg.DATASETS.TRAIN = ("CSLP2D_train",)
    # cfg.DATASETS.TEST = ("CSLP2D_val")


