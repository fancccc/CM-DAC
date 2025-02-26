# -*- coding: utf-8 -*-
'''
@file: train_detection.py
@author: fanc
@time: 2025/1/8 17:23
'''
from model.backbone.detection import Detection
from data.dataloader import LUNA16Dataset, DataLoader
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
import os
from tqdm import tqdm
from accelerate import Accelerator
from model.backbone.loss import ComputeLoss
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from utils.metrics import mAP

def main(args):
    accelerator = Accelerator()
    device = accelerator.device
    if accelerator.is_local_main_process:
        print('Using device:', device)

    # TensorBoard setup
    if accelerator.is_local_main_process and args.phase == 'train':
        writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'logs'))

    # Model, optimizer, scheduler
    det_model = Detection().to(device)
    optimizer = Adam(det_model.parameters(), lr=args.lr, weight_decay=0.001, betas=(0.9, 0.99))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # Dataset and DataLoader
    root = args.root
    traindataset = LUNA16Dataset(root, 'train.json', target_size=(512, 512, 416))
    testdataset = LUNA16Dataset(root, 'test.json', target_size=(512, 512, 416))
    trainloader = DataLoader(traindataset, shuffle=True, num_workers=args.num_workers)
    testloader = DataLoader(testdataset, shuffle=False, num_workers=args.num_workers)

    det_model, optimizer, trainloader, testloader = accelerator.prepare(det_model, optimizer, trainloader, testloader)

    # Load checkpoint
    if args.MODEL_WEIGHT is not None:
        checkpoint = torch.load(args.MODEL_WEIGHT)
        det_model.load_state_dict(checkpoint['det_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    criterion = ComputeLoss()

    # Training phase
    if args.phase == 'train':
        best_val_loss = float('inf')
        for epoch in tqdm(range(args.epochs), desc=f"Total"):
            det_model.train()
            train_loss = 0.0
            pbar_train = tqdm(enumerate(trainloader), desc=f"Epoch {epoch + 1}/{args.epochs}", total=len(trainloader))
            for i, data in pbar_train:
                img, gts = data['img'].to(device), data['relative_coord'].to(device)
                pred, y = det_model(img)
                loss = criterion(pred, gts)

                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()

                train_loss += loss.item()
                if accelerator.is_local_main_process:
                    metrics = mAP(y.cpu().detach().numpy(), gts.cpu().numpy())
                    pbar_train.set_postfix({"loss": loss.item(), 'ap': metrics['ap']})

            train_loss /= len(trainloader)

            # Validation phase
            det_model.eval()
            val_loss = 0.0
            with (torch.no_grad()):
                pbar_val = tqdm(enumerate(testloader), desc=f"Validation {epoch + 1}/{args.epochs}", total=len(testloader))
                all_gts = []
                all_preds = []
                for i, data in pbar_val:
                    img, gts = data['img'].to(device), data['relative_coord'].to(device)
                    pred, y = det_model(img)
                    loss = criterion(pred, gts)
                    val_loss += loss.item()
                    all_gts.append(gts)
                    all_preds.append(y)
                    if accelerator.is_local_main_process:
                        pbar_val.set_postfix({"val_loss": loss.item()})
                all_gts = torch.cat(all_gts, dim=0)
                all_preds = torch.cat(all_preds, dim=0)
                metrics = mAP(all_preds.cpu().numpy(), all_gts.cpu().numpy())
                val_loss /= len(testloader)
            # Update scheduler
            scheduler.step(val_loss)
            # Log to TensorBoard
            if accelerator.is_local_main_process:
                writer.add_scalar('Loss/Train', train_loss, epoch)
                writer.add_scalar('Loss/Validation', val_loss, epoch)
                writer.add_scalar('Metrics/Val', metrics['ap'], epoch)
                print(f"Epoch: {epoch} | Train Loss: {train_loss:.4f}")
                print(f"Epoch: {epoch} | Val Loss: {val_loss:.4f}")
                print(f"Epoch: {epoch} | Val Metrics: {metrics}")

            # Save checkpoint
            if accelerator.is_local_main_process:
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

        if accelerator.is_local_main_process:
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

    accelerator.end_training()

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
    parser.add_argument('--root', type=str, default='/data6-home/fanchenchenzc/datasets/LUNA16')
    args = parser.parse_args()
    if args.phase == 'train':
        now = time.strftime("%Y%m%d%H%M", time.localtime())
        args.save_dir = os.path.join('results', now)
        os.makedirs(args.save_dir, exist_ok=True)

    print(args)
    main(args)
