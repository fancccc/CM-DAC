# -*- coding: utf-8 -*-
'''
@file: train_detection.py
@author: fanc
@time: 2025/2/5 00:14
'''
import model.backbone.classify
# from model.backbone.detection import Detection2d
from model.detectron2.data import LUAD2_3D, DataLoader
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
import os
from tqdm import tqdm
# from accelerate import Accelerator
from utils.loss import ClipLoss
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from model.backbone.classify import CLIP
from model.backbone.classify import tokenize
# from utils.metrics import mAP, mAP_2d
from sklearn.metrics import recall_score, roc_auc_score, cohen_kappa_score, f1_score, precision_score
from sklearn.preprocessing import label_binarize

def label2text2token(label):
    # 转换标签为整数
    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy().tolist()
    elif isinstance(label, list):
        label = label
    # 定义医学标签映射
    label_map = {
        0: "invasive adenocarcinoma",
        1: "minimally invasive adenocarcinoma",
        2: "adenocarcinoma in situ",
        3: "adenocarcinoma in situ"
    }
    # 生成医学描述文本
    texts = [
        f"A pulmonary nodule showing histologic features of {label_map[l]}."
        for l in label
    ]
    # 分词处理
    encoding = tokenize(texts)
    return encoding

def main(args):
    # accelerator = Accelerator()
    # device = accelerator.device
    # if accelerator.is_local_main_process:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # TensorBoard setup
    # if accelerator.is_local_main_process and args.phase == 'train':
    if args.phase == 'train':
        writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'logs'))

    # Dataset and DataLoader
    root = args.root
    # full_dataset = LUAD2_3D(root, phase='all')
    # full_labels = np.array([d['label'] for d in full_dataset])
    # from sklearn.model_selection import StratifiedKFold
    # from torch.utils.data import Subset
    # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1900)
    best = {}
    # for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(full_labels)), full_labels)):
    for fold in range(0, 1):
        # 使用Subset创建子集
        # traindataset = LUAD2_3D(root, phase=f'train_fold{fold}')
        # testdataset = LUAD2_3D(root, phase=f'val_fold{fold}')
        traindataset = LUAD2_3D(root, phase=f'train')
        testdataset = LUAD2_3D(root, phase=f'val')

        print(f'fold: {fold} train samples: {len(traindataset)}, val samples: {len(testdataset)}')
        trainloader = DataLoader(traindataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
        testloader = DataLoader(testdataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)
        # traindataset = LUAD2_3D(root, phase='train')
        # testdataset = LUAD2_3D(root, phase='val')
        # print(f' train samples: {len(traindataset)}, val samples: {len(testdataset)}')
        # trainloader = DataLoader(traindataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
        # testloader = DataLoader(testdataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)
        # Model, optimizer, scheduler
        from model.backbone.classify import Ablation_CLIP, Ablation_no_CL, Ablation_no_CL_exists_CMHF
        cls_model = Ablation_no_CL_exists_CMHF(embed_dim=256,
                         model_depth=18,
                         clinical_dim=27,
                         context_length=77,
                         vocab_size=49408,
                         transformer_width=128, transformer_heads=4, transformer_layers=2).to(device)
        optimizer = Adam(cls_model.parameters(), lr=args.lr, weight_decay=0.001, betas=(0.9, 0.99))
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

        all_labels = [0, 1, 2]
        scaler = torch.cuda.amp.GradScaler()  # 混合精度训练
        # det_model, optimizer, trainloader, testloader = accelerator.prepare(det_model, optimizer, trainloader, testloader)

        # Load checkpoint
        if args.MODEL_WEIGHT is not None:
            checkpoint = torch.load(args.MODEL_WEIGHT)
            cls_model.load_state_dict(checkpoint['cls_model'])
            optimizer.load_state_dict(checkpoint['optimizer'])

        criterion = torch.nn.CrossEntropyLoss()

        # Training phase
        # 训练流程优化
        if args.phase == 'train':
            best_val_loss = 9999
            best_acc = 0.0
            for epoch in range(args.epochs):
                # 训练阶段
                cls_model.train()
                train_loss = 0.0
                progress_bar = tqdm(trainloader, desc=f'Epoch {epoch + 1}/{args.epochs}')

                for batch in progress_bar:
                    optimizer.zero_grad()

                    # 数据加载
                    ct = batch['ct'].to(device, non_blocking=True)
                    clinical = batch['clinical'].to(device, non_blocking=True)
                    # clinical = torch.ones_like(clinical, device=device) # 临床消融

                    labels = batch['label'].to(device, non_blocking=True)

                    # 生成文本特征
                    with torch.no_grad():
                        text_features = label2text2token(labels).to(device)

                    # 混合精度训练
                    with torch.cuda.amp.autocast():
                        cls = cls_model(ct, clinical, text_features)
                        loss = criterion(cls, labels)

                    # 反向传播
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    # 记录损失
                    train_loss += loss.item()
                    progress_bar.set_postfix(loss=loss.item())

                # Validation phase

                # 修改后的验证代码
                val_loss = 0.0
                total_correct = 0
                total_samples = 0
                all_preds = []
                all_labels = []
                all_probs = []

                # 提前生成所有类别的文本特征（固定不变）
                all_class_labels = [0, 1, 2]
                all_class_text_features = label2text2token(all_class_labels).to(device)

                cls_model.eval()
                with (torch.no_grad()):
                    for batch in tqdm(testloader, desc='Validating'):
                        ct = batch['ct'].to(device, non_blocking=True)
                        clinical = batch['clinical'].to(device, non_blocking=True)
                        labels = batch['label'].to(device, non_blocking=True)
                        # clinical = torch.ones_like(clinical, device=device)  # 临床消融

                        text_features = label2text2token(labels).to(device)
                        cls = cls_model(ct, clinical, text_features)
                        val_loss += criterion(cls, labels).item()
                        # val_loss += criterion((logits_per_image, logits_per_text)).item()
                        labels = batch['label'].cpu().numpy()

                        # 使用所有类别的文本特征计算logits
                        # logits_per_image, _ = cls_model(ct, clinical, all_class_text_features)

                        # 计算预测结果和概率
                        probs = torch.softmax(cls, dim=1).cpu().numpy()
                        preds = np.argmax(probs, axis=1)

                        # 收集结果
                        all_preds.extend(preds)
                        all_labels.extend(labels)
                        all_probs.extend(probs)

                        # 计算准确率
                        total_correct += (preds == labels).sum()
                        total_samples += labels.shape[0]

                    # 计算所有指标
                    accuracy = total_correct / total_samples
                    recall = recall_score(all_labels, all_preds, average='macro')  # 宏平均Recall
                    kappa = cohen_kappa_score(all_labels, all_preds)

                    # 计算ROC-AUC（需要二值化标签）
                    y_true_binarized = label_binarize(all_labels, classes=all_class_labels)
                    roc_auc = roc_auc_score(y_true_binarized, all_probs, multi_class='ovr')
                    f1 = f1_score(all_labels, all_preds, average='macro')
                    precision = precision_score(all_labels, all_preds, average='macro')

                # 计算平均损失
                train_loss /= len(trainloader)
                val_loss /= len(testloader)
                print(f'''
                fold: {fold}-{epoch}-{args.epochs}
                TrainLoss: {train_loss:.4f}
                ValLoss: {val_loss:.4f}
                Validation Metrics:
                Accuracy: {accuracy:.4f}
                Precision: {precision:.4f}
                Macro Recall: {recall:.4f}
                F1 Score: {f1:.4f}
                ROC-AUC (OvR): {roc_auc:.4f}
                Kappa: {kappa:.4f}
                ''')

                # 学习率调整
                scheduler.step(val_loss)

                # 记录日志
                writer.add_scalar(f'Loss/Train{fold}', train_loss, epoch)
                writer.add_scalar(f'Loss/Validation{fold}', val_loss, epoch)
                writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
                writer.add_scalar(f'Acc/Val{fold}', accuracy, epoch)
                # writer.add_scalar('Acc/Train', accuracy, epoch)

                # 保存检查点
                checkpoint = {
                    'cls_model': cls_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch
                }
                torch.save(checkpoint, os.path.join(args.save_dir, f'last_model_{fold}.pt'))
                print(f'New model saved at {os.path.join(args.save_dir, f"last_model_{fold}.pt")}')

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(checkpoint, os.path.join(args.save_dir, f'best_model_{fold}.pt'))
                    print(f'New best model saved with val loss: {best_val_loss:.4f}')
                if accuracy >= best_acc:
                    best_acc = accuracy
                    best[fold] = {'fold': fold, 'acc': best_acc, 'recall': recall, 'kappa': kappa, 'auc': roc_auc}
                    torch.save(checkpoint, os.path.join(args.save_dir, f'best_acc_model_{fold}.pt'))
                    print(f'New best model saved with best_acc: {best_acc:.4f}')
                print(f'fold: {fold}, epoch:{epoch}/{args.epochs}, train_loss:{train_loss:.4f}, val_loss:{val_loss:.4f}, accuracy:{accuracy:.4f}, best_acc:{best_acc:.4f}')

            if writer:
                writer.close()
            for b in range(len(best)):
                print(best[b])


    # Testing phase
    # 验证流程
        elif args.phase == 'val':
            total_correct = 0
            total_samples = 0
            all_preds = []
            all_labels = []
            all_probs = []

            # 提前生成所有类别的文本特征（固定不变）
            all_class_labels = [0, 1, 2]
            all_class_text_features = label2text2token(all_class_labels).to(device)

            cls_model.eval()
            with torch.no_grad():
                for batch in tqdm(testloader, desc='Validating'):
                    ct = batch['ct'].to(device, non_blocking=True)
                    clinical = batch['clinical'].to(device, non_blocking=True)
                    # clinical = torch.ones_like(clinical, device=device)  # 临床消融
                    # labels = batch['label'].to(device, non_blocking=True)
                    labels = batch['label'].cpu().numpy()
                    # 使用所有类别的文本特征计算logits
                    cls = cls_model(ct, clinical, all_class_text_features)

                    # 计算预测结果和概率
                    probs = torch.softmax(cls, dim=1).cpu().numpy()
                    preds = np.argmax(probs, axis=1)

                    # 收集结果
                    all_preds.extend(preds)
                    all_labels.extend(labels)
                    all_probs.extend(probs)

                    # 计算准确率
                    total_correct += (preds == labels).sum()
                    total_samples += labels.shape[0]

                # 计算所有指标
                accuracy = total_correct / total_samples
                recall = recall_score(all_labels, all_preds, average='macro')  # 宏平均Recall
                kappa = cohen_kappa_score(all_labels, all_preds)

                # 计算ROC-AUC（需要二值化标签）
                y_true_binarized = label_binarize(all_labels, classes=all_class_labels)
                roc_auc = roc_auc_score(y_true_binarized, all_probs, multi_class='ovr')
                f1 = f1_score(all_labels, all_preds, average='macro')
                precision = precision_score(all_labels, all_preds, average='macro')
            print(f'''
            Validation Metrics:
            Accuracy: {accuracy:.4f}
            Precision: {precision:.4f}
            Macro Recall: {recall:.4f}
            F1 Score: {f1:.4f}
            ROC-AUC (OvR): {roc_auc:.4f}
            Kappa: {kappa:.4f}
            ''')

if __name__ == '__main__':
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--MODEL-WEIGHT', type=str, default=None)
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--root', type=str, default='/zhangyongquan/fanc/datasets/')
    args = parser.parse_args()
    if args.phase == 'train':
        now = time.strftime("%Y%m%d%H%M", time.localtime())
        args.save_dir = os.path.join('results', now)
        os.makedirs(args.save_dir, exist_ok=True)

    print(args)
    main(args)
