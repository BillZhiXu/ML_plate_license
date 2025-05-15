import os
import time
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
from torchvision import transforms

from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# 1. Improved Building Blocks with Batch Normalization and Residual Connections
# ---------------------------------------------------------------------------
class ImprovedConvBlock(nn.Module):
    """Enhanced [Conv + BN + ReLU] × 2 → MaxPool with residual connection"""
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=k, padding=p)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        # Residual connection if dimensions don't match
        self.residual = nn.Sequential()
        if in_ch != out_ch:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch)
            )
    
    def forward(self, x):
        # Save input for residual connection
        residual = self.residual(x)
        
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        # Second conv block with residual connection
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + residual  # Add residual connection
        x = self.relu2(x)
        
        # Pooling
        x = self.pool(x)
        
        return x


class DenseBlockWithBN(nn.Module):
    """Enhanced Dense → LayerNorm → ReLU → Dropout → Dense → LayerNorm → ReLU → Dropout"""
    def __init__(self, in_f, hidden_f, out_f, p=0.5):
        super().__init__()
        # First dense layer + normalization
        self.fc1 = nn.Linear(in_f, hidden_f)
        self.ln1 = nn.LayerNorm(hidden_f)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p)
        # Second dense layer + normalization
        self.fc2 = nn.Linear(hidden_f, out_f)
        self.ln2 = nn.LayerNorm(out_f)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p)
    
    def forward(self, x):
        # Apply first dense + LayerNorm + activation + dropout
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        # Apply second dense + LayerNorm + activation + dropout
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        return x


# ---------------------------------------------------------------------------
# 2. Squeeze and Excitation Block for Channel Attention
# ---------------------------------------------------------------------------
class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channel, reduction=16):
        super().__init__()
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


# ---------------------------------------------------------------------------
# 3. Improved Multi-Head CNN with SE blocks and Residual Connections
# ---------------------------------------------------------------------------
class ImprovedMultiHeadPlateCNN(nn.Module):
    """
    Improved CNN encoder with batch normalization, SE blocks, and residual connections
    
    Args
    ----
    in_channels : int
        Number of input image channels (1 for grayscale, 3 for RGB)
    num_classes : int
        Number of possible classes for each character (36 for alphanumerics + 1 for padding)
    num_heads : int
        Number of separate classification heads (maximum number of characters in license plate)
    input_size : tuple
        Height and width of input images (height, width)
    """
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 37,
                 num_heads: int = 8,
                 input_size: tuple = (64, 256)):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.input_size = input_size
        
        # --- improved convolutional feature extractor ----------
        self.enc = nn.Sequential(
            ImprovedConvBlock(in_channels, 64),     # Block-1
            SEBlock(64),                            # SE attention
            ImprovedConvBlock(64, 128),             # Block-2
            SEBlock(128),                           # SE attention
            ImprovedConvBlock(128, 256),            # Block-3
            SEBlock(256),                           # SE attention
            ImprovedConvBlock(256, 512)             # Block-4
        )
        
        # compute flattened feature size with dummy forward
        with torch.no_grad():
            height, width = input_size
            dummy = torch.zeros(1, in_channels, height, width)
            # Switch to eval mode for this forward pass
            was_training = self.training
            self.eval()
            dummy_out = self.enc(dummy)
            if was_training:
                self.train()  # Restore original mode
            flat_dim = dummy_out.view(1, -1).size(1)
            
        # --- improved fully-connected trunk -------------
        self.trunk = DenseBlockWithBN(flat_dim, 1024, 512, p=0.5)
        
        # --- independent classification heads with label weighting --------
        self.heads = nn.ModuleList([
            nn.Linear(512, num_classes) for _ in range(num_heads)
        ])
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                # Only initialize bias if it exists (skip bias=False layers)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Tensor of shape (batch_size, in_channels, height, width)
        
        Returns:
            list[Tensor]: List of length num_heads, each tensor of shape (batch_size, num_classes)
                         representing the logits for each position in the license plate
        """
        # Extract convolutional features
        feats = self.enc(x)          # (B, C', H', W')
        feats = feats.flatten(1)     # (B, flat_dim)
        
        # Pass through shared trunk
        shared = self.trunk(feats)   # (B, 512)
        
        # Get predictions from each head
        logits = [head(shared) for head in self.heads]
        
        return logits
    
    def predict(self, x, char_map='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_'):
        """
        Convenience method to get character predictions
        
        Args:
            x: Input image tensor
            char_map: String mapping indices to characters
        
        Returns:
            List of predicted license plates (strings)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            
        # Get batch size
        batch_size = x.size(0)
        
        # Store predictions for each sample in batch
        predictions = []
        
        for b in range(batch_size):
            plate = ""
            for head_idx, output in enumerate(logits):
                # Get the prediction for this sample
                _, pred_idx = torch.max(output[b], dim=0)
                char = char_map[pred_idx.item()]
                
                # Only add non-padding characters
                if char != '_':
                    plate += char
            
            predictions.append(plate)
        
        return predictions


# ---------------------------------------------------------------------------
# 4. Advanced Data Augmentation
# ---------------------------------------------------------------------------
def get_advanced_transforms(img_height=64, img_width=256):
    """
    Define improved image transformations with more aggressive augmentation
    """
    # More aggressive augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=10,  # More rotation
                translate=(0.1, 0.1),  # Translation up to 10%
                scale=(0.9, 1.1),  # Scaling up to 10%
                shear=5  # Slight shearing
            )
        ], p=0.7),
        transforms.ColorJitter(
            brightness=0.3,  # More brightness variation
            contrast=0.3,    # More contrast variation
            saturation=0.2,  # Saturation variation
            hue=0.1          # Slight hue shifts
        ),
        # Random perspective distortion to simulate camera angles
        transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
        # Random erasing to simulate occlusions
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # No augmentation for validation - just resize and normalize
    val_transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


# ---------------------------------------------------------------------------
# 5. Focal Loss for Handling Class Imbalance
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """
    Focal Loss to handle class imbalance
    
    Args:
        alpha: Weighting factor for one-hot classes
        gamma: Focusing parameter
        reduction: 'none', 'mean', or 'sum'
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ---------------------------------------------------------------------------
# 6. Data Preparation Functions
# ---------------------------------------------------------------------------
class ArgentinianLicensePlateDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, max_length=8):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            max_length (int): Maximum length of license plate text.
        """
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.max_length = max_length
        self.char_map = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_'

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # The CSV has format with columns 'image_path' and 'plate_text'
        img_path = self.annotations.iloc[idx, 0]  # image_path column
        img_name = os.path.join(self.img_dir, img_path)
        
        label = self.annotations.iloc[idx, 1]  # plate_text column

        try:
            image = Image.open(img_name).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            # Return a blank image in case of error
            image = Image.new('RGB', (224, 64), color='gray')

        if self.transform:
            image = self.transform(image)

        # Convert characters to indices (0-9 for digits, 10-35 for A-Z)
        target = torch.zeros(self.max_length, dtype=torch.long)

        for i, char in enumerate(str(label)[:self.max_length]):
            if '0' <= char <= '9':
                target[i] = ord(char) - ord('0')
            elif 'A' <= char <= 'Z':
                target[i] = 10 + ord(char) - ord('A')
            elif 'a' <= char <= 'z':
                # Convert lowercase to uppercase for consistency
                target[i] = 10 + ord(char) - ord('a')

        # Pad with a special token (36) for shorter license plates
        for i in range(len(str(label)), self.max_length):
            target[i] = 36  # Use 36 as padding token

        return image, target


def prepare_arg_dataloaders(train_csv, val_csv, train_img_dir, val_img_dir, 
                           batch_size=32, max_length=8, img_height=64, img_width=256):
    """
    Prepare training and validation dataloaders for Argentinian license plates
    """
    # Get advanced transforms
    train_transform, val_transform = get_advanced_transforms(img_height, img_width)
    
    # Create datasets using the specific Argentine dataset format
    train_dataset = ArgentinianLicensePlateDataset(
        csv_file=train_csv,
        img_dir=train_img_dir,
        transform=train_transform,
        max_length=max_length
    )
    
    val_dataset = ArgentinianLicensePlateDataset(
        csv_file=val_csv,
        img_dir=val_img_dir,
        transform=val_transform,
        max_length=max_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        drop_last=True,  # Drop last batch if smaller than batch_size
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        drop_last=False,  # Keep all validation samples
        pin_memory=True
    )
    
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# 7. Training and Evaluation Functions
# ---------------------------------------------------------------------------
class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_accuracy(outputs, targets, padding_idx=36):
    """
    Calculate character-level and full-plate-level accuracy
    """
    batch_size = targets.size(0)
    num_heads = len(outputs)
    
    # Character level accuracy
    char_correct = 0
    char_total = 0
    
    # Full plate accuracy (all characters correct)
    plate_correct = 0
    plate_total = batch_size
    
    for head_idx in range(num_heads):
        # Get predictions for this head
        _, preds = torch.max(outputs[head_idx], dim=1)
        
        # Check which predictions match targets
        correct = preds == targets[:, head_idx]
        
        # Count non-padding characters for accuracy
        mask = targets[:, head_idx] != padding_idx
        char_correct += torch.sum(correct & mask).item()
        char_total += torch.sum(mask).item()
        
    # Full plate is correct if all characters are correct
    all_correct = torch.ones(batch_size, dtype=torch.bool, device=targets.device)
    for head_idx in range(num_heads):
        _, preds = torch.max(outputs[head_idx], dim=1)
        head_correct = preds == targets[:, head_idx]
        
        # Only consider positions that aren't padding
        mask = targets[:, head_idx] != padding_idx
        
        # For each sample in batch, AND the correctness
        all_correct = all_correct & (head_correct | ~mask)
    
    plate_correct = torch.sum(all_correct).item()
    
    return char_correct / max(char_total, 1), plate_correct / plate_total


def train_epoch(model, train_loader, criterion, optimizer, scheduler, device):
    """
    Train the model for one epoch with learning rate scheduling
    """
    model.train()
    losses = AverageMeter()
    char_accs = AverageMeter()
    plate_accs = AverageMeter()
    
    pbar = tqdm(train_loader, desc="Training")
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Calculate loss for each head
        loss = 0
        for head_idx, output in enumerate(outputs):
            loss += criterion(output, targets[:, head_idx])
        
        loss /= len(outputs)  # Average loss across heads
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Step the scheduler if it's a batch-level scheduler
        if scheduler is not None and isinstance(scheduler, OneCycleLR):
            scheduler.step()
        
        # Calculate metrics
        char_acc, plate_acc = calculate_accuracy(outputs, targets)
        
        # Update meters
        losses.update(loss.item(), images.size(0))
        char_accs.update(char_acc, images.size(0))
        plate_accs.update(plate_acc, images.size(0))
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'char_acc': f'{char_accs.avg:.4f}',
            'plate_acc': f'{plate_accs.avg:.4f}'
        })
    
    return losses.avg, char_accs.avg, plate_accs.avg


def validate(model, val_loader, criterion, device):
    """
    Validate the model
    """
    model.eval()
    losses = AverageMeter()
    char_accs = AverageMeter()
    plate_accs = AverageMeter()
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss for each head
            loss = 0
            for head_idx, output in enumerate(outputs):
                loss += criterion(output, targets[:, head_idx])
            
            loss /= len(outputs)  # Average loss across heads
            
            # Calculate metrics
            char_acc, plate_acc = calculate_accuracy(outputs, targets)
            
            # Update meters
            losses.update(loss.item(), images.size(0))
            char_accs.update(char_acc, images.size(0))
            plate_accs.update(plate_acc, images.size(0))
            
            # Update progress bar
            pbar.set_postfix({
                'val_loss': f'{losses.avg:.4f}',
                'val_char_acc': f'{char_accs.avg:.4f}',
                'val_plate_acc': f'{plate_accs.avg:.4f}'
            })
    
    return losses.avg, char_accs.avg, plate_accs.avg


def visualize_results(model, val_loader, device, num_samples=5):
    """
    Visualize model predictions vs ground truth
    """
    model.eval()
    
    # Character mapping (0-9 for digits, 10-35 for A-Z, 36 for pad)
    char_map = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_'  # 37 classes (36 is pad)
    
    # Get a batch
    images, targets = next(iter(val_loader))
    images = images.to(device)
    targets = targets.to(device)
    
    with torch.no_grad():
        outputs = model(images)
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3*num_samples))
    
    for i in range(min(num_samples, len(images))):
        # Get the image
        img = images[i].cpu().permute(1, 2, 0).numpy()
        
        # Denormalize image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        # Get ground truth
        gt = ''.join([char_map[targets[i, j].item()] for j in range(targets.size(1))])
        gt = gt.replace('_', '')  # Remove padding
        
        # Get predictions
        pred = ''
        for head_idx, output in enumerate(outputs):
            _, pred_idx = torch.max(output[i], dim=0)
            pred += char_map[pred_idx.item()]
        pred = pred.replace('_', '')  # Remove padding
        
        # Plot
        axes[i].imshow(img)
        axes[i].set_title(f'GT: {gt} | Pred: {pred}')
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 8. Main Training Loop with Improved Optimizer and Scheduler
# ---------------------------------------------------------------------------
def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001, device='cuda'):
    """
    Train the model for multiple epochs with enhanced training strategy
    """
    model = model.to(device)
    
    # Loss function - use Focal Loss or CrossEntropyLoss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # Alternative: criterion = FocalLoss(alpha=1, gamma=2)
    
    # Improved optimizer - AdamW with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Learning rate scheduler - use OneCycleLR for better convergence
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=lr,
        steps_per_epoch=len(train_loader),
        epochs=num_epochs,
        pct_start=0.3,  # Warm up for 30% of training
        div_factor=25,   # Initial lr is max_lr/div_factor
        final_div_factor=1000  # Final lr is max_lr/final_div_factor
    )
    
    # Lists to store metrics
    train_losses, val_losses = [], []
    train_char_accs, val_char_accs = [], []
    train_plate_accs, val_plate_accs = [], []
    
    best_val_loss = float('inf')
    best_char_acc = 0
    best_model_path = 'best_model.pth'
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_char_acc, train_plate_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        
        # Validate
        val_loss, val_char_acc, val_plate_acc = validate(
            model, val_loader, criterion, device
        )
        
        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_char_accs.append(train_char_acc)
        val_char_accs.append(val_char_acc)
        train_plate_accs.append(train_plate_acc)
        val_plate_accs.append(val_plate_acc)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Char Acc: {train_char_acc:.4f}, Plate Acc: {train_plate_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Char Acc: {val_char_acc:.4f}, Plate Acc: {val_plate_acc:.4f}")
        
        # Save best model - track both loss and accuracy
        if val_char_acc > best_char_acc:
            best_char_acc = val_char_acc
            torch.save(model.state_dict(), 'best_acc_model.pth')
            print(f"Model saved to best_acc_model.pth (best accuracy: {best_char_acc:.4f})")
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Model saved to {best_model_path} (best loss: {best_val_loss:.4f})")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(train_char_accs, label='Train')
    plt.plot(val_char_accs, label='Val')
    plt.title('Char Accuracy')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(train_plate_accs, label='Train')
    plt.plot(val_plate_accs, label='Val')
    plt.title('Plate Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    
    # Load best model
    model.load_state_dict(torch.load(best_model_path))
    
    return model, (train_losses, val_losses, train_char_accs, val_char_accs, train_plate_accs, val_plate_accs)


# ---------------------------------------------------------------------------
# 9. Main Function - Updated for Argentinian License Plate Dataset
# ---------------------------------------------------------------------------
def main():
    # Define paths for the Argentinian license plate dataset
    train_csv = 'data/arg_plate_dataset/train_anotaciones.csv'
    val_csv = 'data/arg_plate_dataset/valid_anotaciones.csv'
    train_img_dir = 'data/arg_plate_dataset'  # Will be joined with image_path from CSV
    val_img_dir = 'data/arg_plate_dataset'    # Will be joined with image_path from CSV
    
    # Define hyperparameters
    batch_size = 128
    num_epochs = 500
    max_length = 7      # Most Argentinian plates have up to 7 characters
    learning_rate = 0.001
    
    # Image size - adjust based on actual dataset
    img_height = 64     # Typical height
    img_width = 192     # Typical width for Argentinian plates
    
    # Select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare dataloaders with enhanced augmentation
    train_loader, val_loader = prepare_arg_dataloaders(
        train_csv=train_csv,
        val_csv=val_csv,
        train_img_dir=train_img_dir,
        val_img_dir=val_img_dir,
        batch_size=batch_size,
        max_length=max_length,
        img_height=img_height,
        img_width=img_width
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Create improved model
    model = ImprovedMultiHeadPlateCNN(
        in_channels=3, 
        num_classes=37,  # 0-9, A-Z, and padding
        num_heads=max_length,
        input_size=(img_height, img_width)
    )
    
    # Print model summary
    dummy = torch.zeros(1, 3, img_height, img_width).to(device)
    model = model.to(device)
    outputs = model(dummy)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Output shape: {len(outputs)} heads, each {outputs[0].shape}")
    
    # Train model with improved training strategy
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        lr=learning_rate,
        device=device
    )
    
    # Visualize some results
    _ = visualize_results(trained_model, val_loader, device)
    plt.savefig('argentine_license_plate_results.png')
    
    print("Training completed!")


# ---------------------------------------------------------------------------
# 10. Run Main Function
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main() 