# -*- coding: utf-8 -*-
'''
@file: train_detection.py
@author: author
@time: 2025/2/5 00:14
'''
from model.backbone.detection import Detection2d
from data.dataloader import DataLoader, CSPL3dDataset
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
import os
from tqdm import tqdm
# from accelerate import Accelerator
from model.backbone.loss import ComputeLoss, ComputeLoss2dpn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from utils.metrics import mAP, mAP_2d, ca_metrics

def main(args):
    # accelerator = Accelerator()
    # device = accelerator.device
    # if accelerator.is_local_main_process:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # TensorBoard setup
    # if accelerator.is_local_main_process and args.phase == 'train':
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'logs'))

    # Model, optimizer, scheduler
    det_model = Detection2d(in_channels=1).to(device)
    optimizer = Adam(det_model.parameters(), lr=args.lr, weight_decay=0.001, betas=(0.9, 0.99))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # Dataset and DataLoader
    root = args.root
    traindataset = CSPL3dDataset(root, 'train.csv')
    testdataset = CSPL3dDataset(root, 'test.csv')
    trainloader = DataLoader(traindataset, shuffle=True, batch_size=1, num_workers=args.num_workers)
    testloader = DataLoader(testdataset, shuffle=False, batch_size=1, num_workers=args.num_workers)

    # det_model, optimizer, trainloader, testloader = accelerator.prepare(det_model, optimizer, trainloader, testloader)

    # Load checkpoint
    if args.MODEL_WEIGHT is not None:
        checkpoint = torch.load(args.MODEL_WEIGHT)
        det_model.load_state_dict(checkpoint['det_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    criterion = ComputeLoss2dpn()
    # Metrics = ca_metrics()

    # Training phase
    if args.phase == 'train':
        best_val_loss = float('inf')
        for epoch in tqdm(range(args.epochs), desc=f"Total"):
            det_model.train()
            train_loss = 0.0
            all_gts = []
            all_preds = []
            pbar_train = tqdm(enumerate(trainloader), desc=f"Epoch {epoch + 1}/{args.epochs}", total=len(trainloader))
            for i, data in pbar_train:

                img, gts = data['img'].to(device), data['points'].to(device)
                img, gts = img[0], gts[0] # drop batch
                img = img.permute(3, 0, 1, 2) # c, w, h, d -> d, c, w, h -> b, c, w, h
                # d = img.size(0)
                # 找到有效 slices（gts 绝对值求和后 > 0 视为有效）
                mask = gts.abs().sum(dim=(1, 2)) > 0  # (d,) 其中 True 表示有效 slice
                valid_indices = torch.where(mask)[0]
                num_valid = len(valid_indices)
                # if num_valid >= args.batch_size:
                #     selected_indices = valid_indices[torch.randperm(num_valid)[:args.batch_size]]
                # else:
                #     selected_indices = valid_indices.tolist()
                #     invalid_indices = torch.where(~mask)[0]
                #     num_extra_needed = args.batch_size - num_valid
                #     if len(invalid_indices) > 0:
                #         extra_indices = invalid_indices[
                #             torch.randperm(len(invalid_indices))[:num_extra_needed]].tolist()
                #         selected_indices.extend(extra_indices)
                #     selected_indices = torch.tensor(selected_indices, device=device)
                img, gts = img[valid_indices], gts[valid_indices]

                pred, y = det_model(img)
                loss = criterion(pred, gts)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                all_gts.append(gts)
                all_preds.append(y)

                train_loss += loss.item()
                # if accelerator.is_local_main_process:
                # metrics = ca_metrics(y.cpu().detach(), gts.cpu())
                pbar_train.set_postfix({"loss": loss.item()})
            all_gts = torch.cat(all_gts, dim=0)
            all_preds = torch.cat(all_preds, dim=0)
            t_metrics = ca_metrics(all_preds.detach().cpu(), all_gts.detach().cpu())

            train_loss /= len(trainloader)

            # Validation phase
            det_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                pbar_val = tqdm(enumerate(testloader), desc=f"Validation {epoch + 1}/{args.epochs}", total=len(testloader))
                all_gts = []
                all_preds = []
                for i, data in pbar_val:

                    img, gts = data['img'].to(device), data['points'].to(device)
                    img, gts = img[0], gts[0]  # drop batch
                    img = img.permute(3, 0, 1, 2)  # c, w, h, d -> d, c, w, h -> b, c, w, h
                    d = img.size(0)
                    # 找到有效 slices（gts 绝对值求和后 > 0 视为有效）
                    mask = gts.abs().sum(dim=(1, 2)) > 0  # (d,) 其中 True 表示有效 slice
                    valid_indices = torch.where(mask)[0]
                    num_valid = len(valid_indices)
                    # if num_valid >= args.batch_size:
                    #     selected_indices = valid_indices[torch.randperm(num_valid)[:args.batch_size]]
                    # else:
                    #     selected_indices = valid_indices.tolist()
                    #     invalid_indices = torch.where(~mask)[0]
                    #     num_extra_needed = args.batch_size - num_valid
                    #     if len(invalid_indices) > 0:
                    #         extra_indices = invalid_indices[
                    #             torch.randperm(len(invalid_indices))[:num_extra_needed]].tolist()
                    #         selected_indices.extend(extra_indices)
                    #     selected_indices = torch.tensor(selected_indices, device=device)
                    img, gts = img[valid_indices], gts[valid_indices]

                    pred, y = det_model(img)
                    loss = criterion(pred, gts)
                    val_loss += loss.item()
                    all_gts.append(gts)
                    all_preds.append(y)
                    # if accelerator.is_local_main_process:
                    pbar_val.set_postfix({"val_loss": loss.item()})
                all_gts = torch.cat(all_gts, dim=0)
                all_preds = torch.cat(all_preds, dim=0)
                metrics = ca_metrics(y.cpu().detach(), gts.cpu())
                # metrics = mAP_2d(all_preds.cpu().numpy(), all_gts.cpu().numpy())
                val_loss /= len(testloader)
            # Update scheduler
            scheduler.step(val_loss)
            # Log to TensorBoard
            # if accelerator.is_local_main_process:
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Validation', val_loss, epoch)
            writer.add_scalar('Metrics/Val', metrics['AP'], epoch)
            print(f"Epoch: {epoch} | Train Loss: {train_loss:.4f}")
            print(f"Epoch: {epoch} | Val Loss: {val_loss:.4f}")
            print(f"Epoch: {epoch} | Train Metrics: {t_metrics}")
            print(f"Epoch: {epoch} | Val Metrics: {metrics}")

            # Save checkpoint
            # if accelerator.is_local_main_process:
            checkpoint = {
                'det_model': det_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }
            torch.save(checkpoint, os.path.join(args.save_dir, 'last_model.pt'))
            print(f'Epoch {epoch + 1}/{args.epochs} - Train Loss: {train_loss}, Val Loss: {val_loss}, Saved checkpoint at {args.save_dir}!')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(checkpoint, os.path.join(args.save_dir, 'best_model.pt'))
                print(f'Saved best model with loss {best_val_loss}!')

        # if accelerator.is_local_main_process:
        writer.close()

    # Testing phase
    elif args.phase == 'test':
        det_model.eval()
        with torch.no_grad():
            for i, data in enumerate(testloader):
                img, gts = data['img'], data['relative_coord']
                preds, y = det_model(img)
                y = y.cpu().numpy()[0]
                y = y[y[..., 0] > 0.8]
                print(gts.cpu().numpy()[0, ], y)
                # Add post-processing here if needed


                if i >= 2:  # Limit testing to 2 batches for quick evaluation
                    break

    # accelerator.end_training()

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
    parser.add_argument('--root', type=str, default='/xxxxxx/author/datasets/')
    args = parser.parse_args()
    if args.phase == 'train':
        now = time.strftime("%Y%m%d%H%M", time.localtime())
        args.save_dir = os.path.join('results', now)
        os.makedirs(args.save_dir, exist_ok=True)

    print(args)
    main(args)
