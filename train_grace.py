"""
Training script for loss-tolerant Grace models

This script fine-tunes existing Grace models to be robust to block/packet losses
"""

import os
import argparse
import torch
import logging
import numpy as np
import datetime
import json
from grace.net import VideoCompressor, load_model, save_model
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from PIL import Image
import glob

# Channel dimensions (must match grace/subnet/basics.py)
out_channel_M = 96   # Residual features: 96 × (H/16) × (W/16)
out_channel_N = 64   # Hyperprior z: 64 × (H/64) × (W/64)
out_channel_mv = 128  # Motion vector features: 128 × (H/16) × (W/16)

class VideoFrameDataset(Dataset):
    """
    Dataset for training Grace models
    Loads video frames and creates (current_frame, reference_frame) pairs
    """
    def __init__(self, video_dir, im_height=256, im_width=256, max_frames=None):
        self.video_dir = video_dir
        self.im_height = im_height
        self.im_width = im_width
        
        # Find all image files
        self.frame_paths = sorted(glob.glob(os.path.join(video_dir, "**/*.png"), recursive=True))
        
        # Optionally limit dataset size for faster training
        if max_frames is not None and len(self.frame_paths) > max_frames:
            self.frame_paths = self.frame_paths[:max_frames]
            print(f"Using {len(self.frame_paths)} frames (limited from full dataset) in {video_dir}")
        else:
            print(f"Found {len(self.frame_paths)} frames in {video_dir}")
        
        # Create noise templates
        self.noise_f = torch.zeros([out_channel_M, im_height//16, im_width//16])
        self.noise_z = torch.zeros([out_channel_N, im_height//64, im_width//64])
        self.noise_mv = torch.zeros([out_channel_mv, im_height//16, im_width//16])
    
    def __len__(self):
        return len(self.frame_paths) - 1  # Need pairs
    
    def __getitem__(self, idx):
        # Load current and reference frames
        ref_path = self.frame_paths[idx]
        input_path = self.frame_paths[idx + 1]
        
        ref_img = Image.open(ref_path).convert('RGB')
        input_img = Image.open(input_path).convert('RGB')
        
        # Random crop to training size
        i, j, h, w = transforms.RandomCrop.get_params(
            ref_img, output_size=(self.im_height, self.im_width)
        )
        ref_img = transforms.functional.crop(ref_img, i, j, h, w)
        input_img = transforms.functional.crop(input_img, i, j, h, w)
        
        # Random flip
        if np.random.random() > 0.5:
            ref_img = transforms.functional.hflip(ref_img)
            input_img = transforms.functional.hflip(input_img)
        
        # To tensor and normalize
        ref_img = transforms.functional.to_tensor(ref_img)
        input_img = transforms.functional.to_tensor(input_img)
        
        # Generate quantization noise (uniform [-0.5, 0.5])
        noise_f = torch.nn.init.uniform_(torch.zeros_like(self.noise_f), -0.5, 0.5)
        noise_z = torch.nn.init.uniform_(torch.zeros_like(self.noise_z), -0.5, 0.5)
        noise_mv = torch.nn.init.uniform_(torch.zeros_like(self.noise_mv), -0.5, 0.5)
        
        return input_img, ref_img, noise_f, noise_z, noise_mv


def create_block_loss_masks(shapes, loss_rate=0.3, block_size=100, device='cuda'):
    """
    Create binary masks simulating packet losses by dropping blocks
    
    Args:
        shapes: dict with 'mv', 'res', 'z' shapes
        loss_rate: Probability of block loss
        block_size: Number of elements per block
    
    Returns:
        dict of binary masks
    """
    def create_block_mask(shape):
        numel = torch.prod(torch.tensor(shape[1:])).item()  # Exclude batch dim
        nblocks = (numel - 1) // block_size + 1
        
        # Create block-level mask
        block_mask = (torch.rand(nblocks, device=device) > loss_rate).float()
        
        # Repeat to element level
        mask = block_mask.repeat_interleave(block_size)[:numel]
        mask = mask.reshape(shape[1:]).unsqueeze(0)  # Add batch dim
        
        # Expand to match batch size
        mask = mask.repeat(shape[0], 1, 1, 1)
        return mask
    
    return {
        'mv': create_block_mask(shapes['mv']) if shapes['mv'] is not None else None,
        'res': create_block_mask(shapes['res']) if shapes['res'] is not None else None,
        'z': create_block_mask(shapes['z']) if shapes['z'] is not None else None
    }


def train_epoch(model, train_loader, optimizer, epoch, 
                lambda_rd=2048, lambda_loss=0.5, loss_rate=0.3, 
                block_size=100, tb_logger=None):
    """
    Train one epoch with loss tolerance
    """
    model.train()
    
    total_loss_sum = 0
    rd_loss_sum = 0
    tolerance_loss_sum = 0
    psnr_clean_sum = 0
    psnr_damaged_sum = 0
    count = 0
    
    for batch_idx, (input_img, ref_img, noise_f, noise_z, noise_mv) in enumerate(train_loader):
        # Move to GPU
        input_img = input_img.cuda()
        ref_img = ref_img.cuda()
        noise_f, noise_z, noise_mv = noise_f.cuda(), noise_z.cuda(), noise_mv.cuda()
        
        # Get shapes for mask creation
        # These are estimated based on architecture
        batch_size = input_img.shape[0]
        h, w = input_img.shape[2:]
        shapes = {
            'mv': (batch_size, out_channel_mv, h//16, w//16),
            'res': (batch_size, out_channel_M, h//16, w//16),
            'z': (batch_size, out_channel_N, h//64, w//64)
        }
        
        # Pass 1: Clean reconstruction (no losses)
        recon_clean, mse_clean, warp, inter, bpp_f, bpp_z, bpp_mv, bpp = \
            model(input_img, ref_img, noise_f, noise_z, noise_mv)
        
        # Pass 2: Damaged reconstruction (with simulated losses)
        loss_masks = create_block_loss_masks(shapes, loss_rate, block_size)
        
        # Now VideoCompressor.forward() accepts block_loss_mask parameter!
        # It will add noise first, then apply the mask to simulate packet loss
        recon_damaged, mse_damaged, _, _, _, _, _, _ = \
            model(input_img, ref_img, noise_f, noise_z, noise_mv, block_loss_mask=loss_masks)
        
        # Compute losses
        mse_clean = torch.mean(mse_clean)
        mse_damaged = torch.mean(mse_damaged)
        bpp = torch.mean(bpp)
        
        # R-D loss (compression efficiency)
        rd_loss = lambda_rd * mse_clean + bpp
        
        # Loss-tolerance penalty
        tolerance_loss = lambda_loss * mse_damaged
        
        # Total loss
        total_loss = rd_loss + tolerance_loss
        
        # Backprop
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        # Logging
        count += 1
        total_loss_sum += total_loss.item()
        rd_loss_sum += rd_loss.item()
        tolerance_loss_sum += tolerance_loss.item()
        
        if mse_clean > 0:
            psnr_clean = 10 * torch.log10(1.0 / mse_clean)
            psnr_clean_sum += psnr_clean.item()
        
        if mse_damaged > 0:
            psnr_damaged = 10 * torch.log10(1.0 / mse_damaged)
            psnr_damaged_sum += psnr_damaged.item()
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                  f"Total={total_loss:.4f}, RD={rd_loss:.4f}, Tolerance={tolerance_loss:.4f}, "
                  f"PSNR(clean)={psnr_clean:.2f}dB, PSNR(damaged)={psnr_damaged:.2f}dB, "
                  f"Loss Rate={loss_rate:.2f}")
        
        if tb_logger and batch_idx % 50 == 0:
            step = epoch * len(train_loader) + batch_idx
            tb_logger.add_scalar('loss/total', total_loss.item(), step)
            tb_logger.add_scalar('loss/rd', rd_loss.item(), step)
            tb_logger.add_scalar('loss/tolerance', tolerance_loss.item(), step)
            tb_logger.add_scalar('psnr/clean', psnr_clean.item() if mse_clean > 0 else 100, step)
            tb_logger.add_scalar('psnr/damaged', psnr_damaged.item() if mse_damaged > 0 else 100, step)
            tb_logger.add_scalar('bpp', bpp.item(), step)
            tb_logger.add_scalar('curriculum/loss_rate', loss_rate, step)
    
    # Print epoch summary
    print(f"\nEpoch {epoch} Summary:")
    print(f"  Avg Total Loss: {total_loss_sum/count:.4f}")
    print(f"  Avg RD Loss: {rd_loss_sum/count:.4f}")
    print(f"  Avg Tolerance Loss: {tolerance_loss_sum/count:.4f}")
    print(f"  Avg PSNR (clean): {psnr_clean_sum/count:.2f}dB")
    print(f"  Avg PSNR (damaged): {psnr_damaged_sum/count:.2f}dB\n")


def main():
    parser = argparse.ArgumentParser(description='Train loss-tolerant Grace')
    parser.add_argument('--pretrain', required=True, help='Pretrained Grace model to fine-tune')
    parser.add_argument('--data-dir', required=True, help='Directory with training videos/frames')
    parser.add_argument('--output', required=True, help='Output model path')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size (paper uses 4)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (paper uses 1e-4)')
    parser.add_argument('--lr-decay', type=float, default=0.1, help='Learning rate decay (paper uses 0.1)')
    parser.add_argument('--lambda-rd', type=float, default=2048, help='R-D lambda')
    parser.add_argument('--lambda-loss', type=float, default=0.5, help='Loss-tolerance lambda')
    parser.add_argument('--max-loss-rate', type=float, default=0.5, help='Max loss rate')
    parser.add_argument('--loss-block-size', type=int, default=100, help='Loss block size')
    parser.add_argument('--max-frames', type=int, default=None, help='Max frames to use (None = all frames)')
    parser.add_argument('--log-dir', default='./runs_loss_tolerant', help='TensorBoard log dir')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("Training Loss-Tolerant Grace Model")
    logger.info(f"Pretrained model: {args.pretrain}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Lambda R-D: {args.lambda_rd}")
    logger.info(f"Lambda Loss: {args.lambda_loss}")
    logger.info(f"Max loss rate: {args.max_loss_rate}")
    logger.info("=" * 80)
    
    # Load pretrained Grace model
    model = VideoCompressor()
    logger.info(f"Loading pretrained model from {args.pretrain}")
    load_model(model, args.pretrain, map_location='cuda')
    model = model.cuda()
    model.train()  # Switch to training mode!
    
    # Optimizer (paper uses Adam with 1e-4 LR)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler (paper uses decay of 0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs//3, gamma=args.lr_decay)
    
    # TensorBoard
    tb_logger = SummaryWriter(args.log_dir)
    
    # Dataset
    logger.info(f"Loading dataset from {args.data_dir}")
    if args.max_frames:
        logger.info(f"Limiting to {args.max_frames} frames")
    train_dataset = VideoFrameDataset(args.data_dir, max_frames=args.max_frames)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    # Training loop with curriculum learning
    for epoch in range(args.epochs):
        # Curriculum: gradually increase loss rate
        current_loss_rate = min(args.max_loss_rate, 
                               0.1 + (args.max_loss_rate - 0.1) * epoch / args.epochs)
        
        logger.info(f"\nEpoch {epoch}/{args.epochs}, Loss Rate: {current_loss_rate:.2f}")
        
        train_epoch(
            model, train_loader, optimizer, epoch,
            lambda_rd=args.lambda_rd,
            lambda_loss=args.lambda_loss,
            loss_rate=current_loss_rate,
            block_size=args.loss_block_size,
            tb_logger=tb_logger
        )
        
        # Step the learning rate scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Learning rate: {current_lr:.6f}")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = args.output.replace('.model', f'_epoch{epoch}.model')
            save_model(model, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    save_model(model, args.output)
    logger.info(f"Training complete! Saved to {args.output}")
    
    tb_logger.close()


if __name__ == "__main__":
    main()

