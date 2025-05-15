import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import random
import math
import torch.nn.functional as F

# --- CONFIG ---
IMG_HEIGHT, IMG_WIDTH = 64, 128
NUM_DIGITS = 7
BATCH_SIZE = 32
EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- CHARSET ---
CHARS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
CHAR2IDX = {c: i for i, c in enumerate(CHARS)}
IDX2CHAR = {i: c for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS)
MAX_PLATE_LEN = 7

# --- DATASET PLACEHOLDER ---
class ArgPlateDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None, mixup_prob=0.3, cutmix_prob=0.3):
        self.data = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform
        self.mixup_prob = mixup_prob
        self.cutmix_prob = cutmix_prob
        # Filter out plates with length not in [6,7] or with invalid chars
        def valid_plate(p):
            return (6 <= len(p) <= 7) and all(c in CHAR2IDX for c in p)
        self.data = self.data[self.data['plate_text'].apply(valid_plate)].reset_index(drop=True)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, row['image_path'])
        img = Image.open(img_path).convert('L')
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        img = np.array(img, dtype=np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
        if self.transform:
            img = self.transform(img)
        plate = row['plate_text']
        label = [CHAR2IDX[c] for c in plate]
        # Pad to MAX_PLATE_LEN with -1 (for ignore_index in loss)
        label = label + [-1] * (MAX_PLATE_LEN - len(label))
        label = torch.tensor(label, dtype=torch.long)
        return img, label
    
    def mixup(self, x1, y1, x2, y2, alpha=0.2):
        # Mix images
        lam = np.random.beta(alpha, alpha)
        mixed_x = lam * x1 + (1 - lam) * x2
        return mixed_x, y1, y2, lam
    
    def cutmix(self, x1, y1, x2, y2, alpha=0.2):
        lam = np.random.beta(alpha, alpha)
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(x1.size(), lam)
        x1[:, :, bbx1:bbx2, bby1:bby2] = x2[:, :, bbx1:bbx2, bby1:bby2]
        # Adjust lambda to match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x1.size()[2] * x1.size()[3]))
        return x1, y1, y2, lam
    
    def _rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2

# --- DATA AUGMENTATION ---
train_transform = T.Compose([
    T.Lambda(lambda x: x),  # Placeholder for grayscale
    T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2), fill=0),
    T.RandomPerspective(distortion_scale=0.2, p=0.5, fill=0),
    T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.3),
    T.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
    T.RandomApply([
        T.Lambda(lambda x: T.functional.adjust_contrast(x, contrast_factor=random.uniform(0.8, 1.5)))
    ], p=0.3),
    T.RandomApply([
        T.Lambda(lambda x: torch.clamp(x + torch.randn_like(x) * 0.05, 0, 1))  # Add noise
    ], p=0.3),
])
val_transform = T.Lambda(lambda x: x)

# Squeeze and Excitation Block
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Dilated Convolution Block
class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(DilatedConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            padding=dilation * (kernel_size//2),
            dilation=dilation,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# Spatial attention module
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Calculate spatial attention map
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_pool, max_pool], dim=1)
        spatial = self.conv(spatial)
        attention = self.sigmoid(spatial)
        return x * attention

# DropBlock - more structured dropout for CNNs
class DropBlock2D(nn.Module):
    def __init__(self, block_size=7, drop_prob=0.1):
        super(DropBlock2D, self).__init__()
        self.block_size = block_size
        self.drop_prob = drop_prob
        
    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        
        # Compute gamma
        gamma = self.drop_prob / (self.block_size ** 2)
        
        # Sample mask and place drops
        mask = torch.bernoulli(torch.ones_like(x) * gamma)
        
        # Block dropout
        mask = F.max_pool2d(
            mask, 
            kernel_size=(self.block_size, self.block_size), 
            stride=(1, 1), 
            padding=self.block_size//2
        )
        
        # Scale to compensate for dropped values
        if mask.sum() > 0:
            scale = x.numel() / mask.sum()
            mask = mask * scale
        
        return x * (1 - mask)

# Feature Pyramid Module
class FPM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPM, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        self.dilation1 = DilatedConvBlock(in_channels, out_channels, dilation=1)
        self.dilation2 = DilatedConvBlock(in_channels, out_channels, dilation=2)
        self.dilation3 = DilatedConvBlock(in_channels, out_channels, dilation=4)
        
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        self.se = SEBlock(out_channels * 4)
        self.output_conv = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
    def forward(self, x):
        size = x.size()[2:]
        
        feat1 = self.conv1(x)
        feat2 = self.dilation1(x)
        feat3 = self.dilation2(x)
        feat4 = self.dilation3(x)
        
        global_feat = self.global_avg_pool(x)
        global_feat = F.interpolate(global_feat, size=size, mode='bilinear', align_corners=True)
        
        feat = torch.cat([feat1, feat2, feat3, feat4], dim=1)
        feat = self.se(feat)
        return self.output_conv(feat)

# --- MODEL ---
class AdvancedLicensePlateCNN(nn.Module):
    def __init__(self):
        super(AdvancedLicensePlateCNN, self).__init__()
        
        # Convolutional backbone
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),
            DropBlock2D(5, 0.1)
        )
        
        # Multi-scale features
        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            SEBlock(64),
            nn.MaxPool2d(2),
            DropBlock2D(5, 0.1)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            SEBlock(128),
            nn.MaxPool2d(2),
            DropBlock2D(5, 0.2)
        )
        
        # Feature Pyramid Module
        self.fpm = FPM(128, 256)
        
        # Attention mechanism
        self.spatial_attention = SpatialAttention()
        
        # Global pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dense MLP layers with strong regularization
        self.dense_layers = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            
            nn.Linear(512, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            
            nn.Linear(512, 384),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            
            nn.Linear(384, 384),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4)
        )
        
        # Character prediction heads
        self.digit_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(384, 256),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.3),
                nn.Linear(256, NUM_CLASSES)
            ) for _ in range(MAX_PLATE_LEN)
        ])
        
        # Plate-level classifier (auxiliary output)
        self.plate_classifier = nn.Linear(384, 1)

    def forward(self, x):
        # Stem and backbone
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        
        # Feature Pyramid
        x = self.fpm(x)
        
        # Spatial attention
        x = self.spatial_attention(x)
        
        # Global pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Dense layers
        features = self.dense_layers(x)
        
        # Character prediction outputs
        char_outputs = [head(features) for head in self.digit_heads]
        
        # Auxiliary plate-level output
        plate_out = self.plate_classifier(features)
        
        return char_outputs, plate_out

# Focal Loss for better handling of class imbalance
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, ignore_index=-1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        
    def forward(self, input, target):
        if target.dim() > 1:
            target = target.squeeze(1)
        
        # Create mask for ignored indices
        valid_mask = (target != self.ignore_index)
        
        # Get only valid targets
        target_valid = target[valid_mask]
        
        if len(target_valid) == 0:
            return torch.tensor(0.0, device=input.device, requires_grad=True)
        
        # Get valid logits
        input_valid = input[valid_mask.unsqueeze(1).expand_as(input)].view(-1, input.size(-1))
        
        # Compute CE loss
        log_pt = F.log_softmax(input_valid, dim=1)
        log_pt = log_pt.gather(1, target_valid.unsqueeze(1))
        log_pt = log_pt.view(-1)
        pt = log_pt.exp()
        
        # Compute focal loss
        focal_loss = -((1 - pt) ** self.gamma) * log_pt
        
        # Apply alpha if provided
        if self.alpha is not None:
            alpha = self.alpha.gather(0, target_valid)
            focal_loss = alpha * focal_loss
            
        return focal_loss.mean()

# --- TRAINING & EVAL ---
def train_epoch(model, loader, optimizer, criterion, mixup_criterion=None, mixup_prob=0.3, cutmix_prob=0.3):
    model.train()
    total_loss = 0
    char_correct = np.zeros(MAX_PLATE_LEN)
    char_total = np.zeros(MAX_PLATE_LEN)
    
    for imgs, labels in tqdm(loader, desc='Training', leave=False):
        batch_size = imgs.size(0)
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        
        # Apply mixup or cutmix
        do_mixup = random.random() < mixup_prob
        do_cutmix = random.random() < cutmix_prob
        
        if do_mixup or do_cutmix:
            # Create shuffled indices
            rand_idx = torch.randperm(batch_size).to(DEVICE)
            target_b = labels[rand_idx]
            
            # Apply augmentation
            if do_mixup:
                lam = np.random.beta(0.2, 0.2)
                imgs = lam * imgs + (1 - lam) * imgs[rand_idx]
            elif do_cutmix:
                lam = np.random.beta(0.2, 0.2)
                bbx1, bby1, bbx2, bby2 = _rand_bbox(imgs.size(), lam)
                imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[rand_idx, :, bbx1:bbx2, bby1:bby2]
                # Adjust lambda to match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs.size()[2] * imgs.size()[3]))
        
        optimizer.zero_grad()
        
        char_outputs, plate_out = model(imgs)
        
        # Character loss
        if do_mixup or do_cutmix:
            loss = 0
            for i, out in enumerate(char_outputs):
                loss += mixup_criterion(criterion, out, labels[:, i], target_b[:, i], lam)
            loss = loss / len(char_outputs)
        else:
            loss = sum(criterion(out, labels[:, i]) for i, out in enumerate(char_outputs))
            
        # Add auxiliary plate-level loss
        # If you have a plate-level target, use it here
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        total_loss += loss.item() * batch_size
        
        # Calculate accuracy
        with torch.no_grad():
            preds = [out.argmax(1) for out in char_outputs]
            
            for i, pred in enumerate(preds):
                mask = (labels[:, i] != -1)
                char_correct[i] += ((pred == labels[:, i]) & mask).sum().item()
                char_total[i] += mask.sum().item()
    
    avg_loss = total_loss / len(loader.dataset)
    char_acc = char_correct / np.maximum(char_total, 1)
    
    return avg_loss, char_acc

def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    char_correct = np.zeros(MAX_PLATE_LEN)
    char_total = np.zeros(MAX_PLATE_LEN)
    all_preds, all_labels, all_masks = [], [], []
    
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc='Evaluating', leave=False):
            batch_size = imgs.size(0)
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            char_outputs, plate_out = model(imgs)
            
            # Character loss
            loss = sum(criterion(out, labels[:, i]) for i, out in enumerate(char_outputs))
            
            total_loss += loss.item() * batch_size
            
            # Calculate metrics
            preds = [out.argmax(1) for out in char_outputs]
            stacked_preds = torch.stack(preds, dim=1)
            
            # Per-character accuracy
            for i, pred in enumerate(preds):
                mask = (labels[:, i] != -1)
                char_correct[i] += ((pred == labels[:, i]) & mask).sum().item()
                char_total[i] += mask.sum().item()
            
            # Store for plate-level accuracy calculation
            mask = (labels != -1)
            all_preds.append(stacked_preds.cpu())
            all_labels.append(labels.cpu())
            all_masks.append(mask.cpu())
    
    # Concatenate results
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    
    # Calculate plate-level accuracy (all characters correct)
    correct_chars = (all_preds == all_labels) | ~all_masks
    plate_acc = correct_chars.all(dim=1).float().mean().item()
    
    # Character accuracy
    char_acc = char_correct / np.maximum(char_total, 1)
    
    avg_loss = total_loss / len(loader.dataset)
    
    return avg_loss, char_acc, plate_acc, all_preds, all_labels, all_masks

# Helper function for cutmix
def _rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # Uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2

# Mixup criterion
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    a_mask = (y_a != -1)
    b_mask = (y_b != -1)
    
    loss = 0
    if a_mask.sum() > 0:
        loss += lam * criterion(pred[a_mask], y_a[a_mask])
    if b_mask.sum() > 0:
        loss += (1 - lam) * criterion(pred[b_mask], y_b[b_mask])
    return loss

# Cosine annealing with warmup learning rate scheduler
class CosineAnnealingWarmupRestarts(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, first_cycle_steps, cycle_mult=1.0, max_lr=0.1, min_lr=0.001, warmup_steps=0, gamma=1.0, last_epoch=-1):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        
        self.cycle = 0
        self.cycle_steps = first_cycle_steps
        self._step_count = 0  # Initialize _step_count to match PyTorch convention
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        self.init_lr()
    
    def init_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
    
    def get_lr(self):
        if self._step_count < self.warmup_steps:  # Use _step_count
            # Linear warmup
            return [self.min_lr + (base_lr - self.min_lr) * self._step_count / self.warmup_steps
                    for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            return [self.min_lr + (self.max_lr - self.min_lr) * 
                    (1 + math.cos(math.pi * (self._step_count - self.warmup_steps) / 
                                 (self.cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]
    
    def step(self, epoch=None):
        if epoch is None:
            self._step_count += 1  # Use _step_count
            if self._step_count >= self.cycle_steps:  # Use _step_count
                # Reset cycle
                self.cycle += 1
                self._step_count = 0  # Use _step_count
                self.cycle_steps = int(self.cycle_steps * self.cycle_mult)
                self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        else:
            self._step_count = epoch  # Use _step_count
        
        return super(CosineAnnealingWarmupRestarts, self).step(epoch)

# --- MAIN TRAINING LOOP ---
def main():
    # Data
    train_set = ArgPlateDataset(
        csv_path='data/arg_plate_dataset/train_anotaciones.csv',
        root_dir='data/arg_plate_dataset',
        transform=train_transform)
    val_set = ArgPlateDataset(
        csv_path='data/arg_plate_dataset/valid_anotaciones.csv',
        root_dir='data/arg_plate_dataset',
        transform=val_transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
    
    # Model
    model = AdvancedLicensePlateCNN().to(DEVICE)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999))
    
    # Loss function
    # You can use Focal Loss for better handling of class imbalance
    criterion = FocalLoss(gamma=2.0, ignore_index=-1)
    
    # Learning rate scheduler
    # Cosine annealing with warmup
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=EPOCHS,
        warmup_steps=5,
        max_lr=1e-3,
        min_lr=1e-6
    )
    
    # Training tracking
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    val_plate_accs = []
    best_val_loss = float('inf')
    best_plate_acc = 0.0
    patience, patience_limit = 0, 8  # Increased patience with better model
    
    print(f"\n{'='*80}\nTraining on {len(train_set)} samples, validating on {len(val_set)} samples\n{'='*80}")
    
    for epoch in range(EPOCHS):
        print(f"\n{'='*30} Epoch {epoch+1}/{EPOCHS} {'='*30}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        tr_loss, tr_acc = train_epoch(
            model, 
            train_loader, 
            optimizer, 
            criterion,
            mixup_criterion=mixup_criterion
        )
        
        # Update learning rate - moved here to be after optimizer.step()
        scheduler.step()
        
        # Validate
        val_loss, val_acc, val_plate_acc, _, _, _ = eval_epoch(model, val_loader, criterion)
        
        # Calculate mean accuracies
        tr_mean_acc = np.mean(tr_acc)
        val_mean_acc = np.mean(val_acc)
        
        # Store metrics for plotting
        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        train_accs.append(tr_acc)
        val_accs.append(val_acc)
        val_plate_accs.append(val_plate_acc)
        
        # Print epoch results
        print(f"Loss      - Train: {tr_loss:.4f}  |  Val: {val_loss:.4f}")
        print(f"Mean Acc  - Train: {tr_mean_acc:.4f}  |  Val: {val_mean_acc:.4f}")
        print(f"Plate Acc - Val: {val_plate_acc:.4f}")
        
        # Print per-character accuracies
        print("\nPer-character accuracies:")
        print("Position  | Train    | Val")
        print("-" * 30)
        for i in range(MAX_PLATE_LEN):
            print(f"Char {i}    | {tr_acc[i]:.4f}   | {val_acc[i]:.4f}")
        
        # Save based on both loss and plate accuracy
        save_checkpoint = False
        
        # Check if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint = True
            loss_message = f"✅ New best val_loss: {val_loss:.4f}"
        else:
            loss_message = f"No improvement in val_loss. Best: {best_val_loss:.4f}"
            
        # Check if plate accuracy improved    
        if val_plate_acc > best_plate_acc:
            best_plate_acc = val_plate_acc
            save_checkpoint = True
            acc_message = f"✅ New best plate_acc: {val_plate_acc:.4f}"
        else:
            acc_message = f"No improvement in plate_acc. Best: {best_plate_acc:.4f}"
        
        # Early stopping logic
        if save_checkpoint:
            patience = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_plate_acc': val_plate_acc,
            }, 'best_model.pt')
            print(f"\n{loss_message}")
            print(f"{acc_message}")
            print("✅ Model checkpoint saved!")
        else:
            patience += 1
            print(f"\n{loss_message}")
            print(f"{acc_message}")
            print(f"⚠️ No improvement. Patience: {patience}/{patience_limit}")
            if patience >= patience_limit:
                print("\n🛑 Early stopping triggered.")
                break
    
    print(f"\n{'='*30} Training Complete {'='*30}")
    
    # Load best model for final evaluation
    checkpoint = torch.load('best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation
    val_loss, val_acc, val_plate_acc, preds, labels, masks = eval_epoch(model, val_loader, criterion)
    print("\n--- Final Evaluation ---")
    print(f"Best validation loss: {checkpoint['val_loss']:.4f}")
    print(f"Best plate accuracy: {checkpoint['val_plate_acc']:.4f}")
    print(f"Per-character accuracy: {[f'{a:.4f}' for a in val_acc]}")
    
    # Plot curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(1, 3, 2)
    for i in range(MAX_PLATE_LEN):
        plt.plot([a[i] for a in val_accs], label=f'Char {i}')
    plt.legend()
    plt.title('Validation Character Accuracy')
    
    plt.subplot(1, 3, 3)
    plt.plot(val_plate_accs, label='Plate Accuracy')
    plt.legend()
    plt.title('Plate Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()
    
    # Show sample predictions
    plt.figure(figsize=(15, 10))
    for i in range(min(5, len(val_set))):
        plt.subplot(1, 5, i+1)
        img = val_set[i][0].squeeze().numpy()
        plt.imshow(img, cmap='gray')
        true_str = ''.join([IDX2CHAR[c.item()] for c in labels[i] if c != -1])
        pred_str = ''.join([IDX2CHAR[c.item()] for c, m in zip(preds[i], masks[i]) if m])
        plt.title(f"True: {true_str}\nPred: {pred_str}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png')
    plt.show()

if __name__ == '__main__':
    main()
