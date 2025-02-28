# -*- coding: utf-8 -*-
'''
@file: classify.py
@author: author
@time: 2025/2/21 上午10:04
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
# from data import LUAD2_2D, CSLP2D
from detectron2.data import build_detection_train_loader, DatasetMapper

from detectron2.modeling import build_backbone, RetinaNet, detector_postprocess
import torch.nn as nn
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.config import CfgNode as CN
from detectron2.evaluation import COCOEvaluator
from model.detectron2.data import LUAD2_3D
from model.backbone.ResNet3D import generate_model
from torch.nn import MultiheadAttention
from model.backbone.simple_tokenizer import SimpleTokenizer as _Tokenizer
from typing import Union, List
from model.backbone.module import Transformer, LayerNorm
import numpy as np

_tokenizer = _Tokenizer()

def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result

class CT_Encoder(nn.Module):
    def __init__(self, model_depth=50):
        super(CT_Encoder, self).__init__()
        self.encoder = generate_model(model_depth=18)
        # 通道调整层列表（匹配不同阶段输出）
        self.channel_adjust = nn.ModuleList([
            nn.Conv3d(64, 256, 1),
            nn.Conv3d(128, 256, 1),
            nn.Conv3d(256, 256, 1),
            nn.Conv3d(512, 256, 1)
        ])

    def forward(self, x):
        # 获取四个阶段的3D特征
        features = self.encoder(x)
        # 统一通道维度为256
        return [adjust(feat) for adjust, feat in zip(self.channel_adjust, features)]


class Tab_Encoder(nn.Module):
    def __init__(self, input_dim=27):
        super(Tab_Encoder, self).__init__()
        # 动态生成各层级特征
        self.proj_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 256)
            ) for _ in range(4)  # 生成四个独立投影头
        ])

    def forward(self, x):
        # 并行生成四个层级的临床特征
        return [proj(x) for proj in self.proj_layers]  # 每个形状: (B,256)


class MultiLayerAtt(nn.Module):
    def __init__(self):
        super(MultiLayerAtt, self).__init__()
        # 各层级注意力配置
        self.attentions = nn.ModuleList([
            MultiheadAttention(embed_dim=256, num_heads=8),
            MultiheadAttention(embed_dim=256, num_heads=4),
            MultiheadAttention(embed_dim=256, num_heads=2),
            MultiheadAttention(embed_dim=256, num_heads=1)
        ])
        # 临床特征扩展器
        self.clin_expanders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 256*8),  # 动态生成扩展参数
                nn.ReLU()
            ) for _ in range(4)
        ])

    def _process_3d_feat(self, feat):
        """将3D特征转换为注意力序列格式"""
        B, C, D, H, W = feat.shape
        return feat.view(B, C, -1).permute(2, 0, 1)  # (N, B, C)

    def forward(self, ct_features, clinical_features):
        fused = []
        for i, (ct, clin, attn) in enumerate(zip(ct_features, clinical_features, self.attentions)):
            # 转换CT特征维度 (B,C,D,H,W) → (N,B,C)
            ct_seq = self._process_3d_feat(ct)  # N=D*H*W

            # 动态扩展临床特征
            B = ct.size(0)
            clin = self.clin_expanders[i](clin)  # (B, 256*8)
            clin = clin.view(B, 256, 8).permute(2, 0, 1)  # (8, B, 256)

            # 循环扩展至匹配CT序列长度
            repeat_num = ct_seq.size(0) // 8 + 1
            clin_expanded = clin.repeat(repeat_num, 1, 1)[:ct_seq.size(0)]  # (N, B, 256)

            # 执行注意力机制
            attn_out, _ = attn(
                query=clin_expanded,  # 扩展后的临床特征 (N,B,C)
                key=ct_seq,  # CT特征作为Key
                value=ct_seq  # CT特征作为Value
            )

            # 残差连接（需要维度对齐）
            attn_map = attn_out.permute(1, 2, 0).view_as(ct)
            fused.append(ct + attn_map)

        return fused


class MultiScaleFusion(nn.Module):
    def __init__(self, clinical_dim=27):
        super().__init__()
        # 临床特征增强器
        self.clinical_enhancer = nn.Sequential(
            nn.Linear(clinical_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.proj = nn.Linear(5*256, 256)

    def forward(self, ct_features, clinical):
        ct_features = [self.avg_pool(i) for i in ct_features]
        ct_features = torch.cat(ct_features, 2).squeeze(-1).squeeze(-1)
        cli_feature = self.clinical_enhancer(clinical)
        features = torch.cat([ct_features, cli_feature.unsqueeze(-1)], dim=-1)
        features = self.proj(features.flatten(start_dim=1))

        # print(features.shape)
        return features.unsqueeze(1)

class CT_EHR(nn.Module):
    def __init__(self, clinical_dim=27, model_depth=18):
        super(CT_EHR, self).__init__()
        self.ct_encoder = CT_Encoder(model_depth)
        self.tab_encoder = Tab_Encoder(clinical_dim)
        self.attfusion = MultiLayerAtt()
        self.msfusion = MultiScaleFusion(clinical_dim)

    def forward(self, ct_vol, clinical_data):
        # 特征提取
        ct_features = self.ct_encoder(ct_vol)  # 4个(B,256,D,H,W)
        clinical_features = self.tab_encoder(clinical_data)  # 4个(B,256)

        # 分层跨模态融合
        fused_features = self.attfusion(ct_features, clinical_features)
        fused_features = self.msfusion(fused_features, clinical_data)
        # print(fused_features.shape)
        return fused_features

class TextEncoder(nn.Module):
    def __init__(self, model_depth=18):
        super(TextEncoder, self).__init__()

class CommentAddCT_EHR(nn.Module):
    def __init__(self, clinical_dim=27, model_depth=18):
        super(CommentAddCT_EHR, self).__init__()
        self.ct_encoder = generate_model(model_depth)
        self.tab_encoder = nn.Linear(clinical_dim, 512)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.proj = nn.Linear(512, 256)
        # self.attfusion = MultiLayerAtt()
        # self.msfusion = MultiScaleFusion(clinical_dim)

    def forward(self, ct_vol, clinical_data):
        # 特征提取
        ct_features = self.ct_encoder(ct_vol)  # 4个(B,256,D,H,W)
        clinical_features = self.tab_encoder(clinical_data).unsqueeze(1)  # 4个(B,256)
        ct_features = self.pool(ct_features[-1]).squeeze(-1).squeeze(-1)
        ct_features = ct_features.permute(0, 2, 1)
        features = self.proj(ct_features + clinical_features)
        # print(features.shape)
        return features
        # print(ct_features.shape, clinical_features.shape)


        # print(fused_features.shape)
        # return fused_features
class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 model_depth: int,
                 # text
                 clinical_dim: int,
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()
        self.ct_ehr = CT_EHR(clinical_dim=clinical_dim, model_depth=model_depth)

        self.context_length = context_length

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_image(self, ct, cli):
        return self.ct_ehr(ct, cli)

    def encode_text(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        # print(x.shape)

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, ct, cli, text):
        image_features = self.encode_image(ct, cli)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features.mean(dim=1)
        # print(image_features.shape, text_features.shape)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-8)
        # assert not torch.isnan(image_features).any(), "图像编码器输出NaN"
        # assert not torch.isnan(text_features).any(), "文本编码器输出NaN"
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text

class Ablation_CLIP(CLIP):
    def __init__(self,clinical_dim=27, model_depth=18,  **kwargs):
        super().__init__(clinical_dim=27, model_depth=18,  **kwargs)
        self.ct_ehr = CommentAddCT_EHR(clinical_dim=clinical_dim, model_depth=model_depth)

class Ablation_no_CL(CLIP):
    def __init__(self,clinical_dim=27, model_depth=18,  **kwargs):
        super().__init__(clinical_dim=27, model_depth=18,  **kwargs)
        self.ct_ehr = CommentAddCT_EHR(clinical_dim=clinical_dim, model_depth=model_depth)
        self.classifier = nn.Linear(256, 3)

    def forward(self, ct, cli, text):
        image_features = self.encode_image(ct, cli)
        cls = self.classifier(image_features).squeeze(1)
        # print(cls.shape)
        return cls

class Ablation_no_CL_exists_CMHF(CLIP):
    def __init__(self,clinical_dim=27, model_depth=18,  **kwargs):
        super().__init__(clinical_dim=27, model_depth=18,  **kwargs)
        # self.ct_ehr = CT_EHR(clinical_dim=clinical_dim, model_depth=model_depth)
        self.classifier = nn.Linear(256, 3)

    def forward(self, ct, cli, text):
        image_features = self.encode_image(ct, cli)
        # print(image_features.shape)
        cls = self.classifier(image_features).squeeze(1)
        # print(cls.shape)
        return cls




if __name__ == '__main__':
    ct = torch.randn(5, 1, 128, 128, 32)
    cli = torch.randn(5, 8)
    clinical_dim = 8
    text = ['class1', 'class2', 'class1', 'class2', 'CLASS3']
    token_encode = tokenize(text)
    # print(token_encode.shape)
    # model = CT_EHR()
    model = CLIP(embed_dim=256, model_depth=18, clinical_dim=clinical_dim, context_length=77, vocab_size=49408, transformer_width=128, transformer_heads=4, transformer_layers=2)
    logits_per_image, logits_per_text = model(ct, cli, token_encode)
    from utils.loss import ClipLoss

    loss_function = ClipLoss()
    logits = (logits_per_image, logits_per_text)
    print(torch.isnan(logits_per_image).any(), torch.isnan(logits_per_text).any())
    print(loss_function(logits))


    # 前向传播
    # outputs = model(ct, cli)
    # token_encode = tokenize('test')
    # print(token_encode)

    # 验证输出维度
    # for i, feat in enumerate(outputs):
    #     print(f"Layer {i + 1} output shape: {feat.shape}")

