import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.utils import save_image, make_grid
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
import json
from pathlib import Path
from datetime import datetime

class LicensePlateDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, transform=None):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.transform = transform
        self.clean_files = sorted(os.listdir(clean_dir))
        self.noisy_files = sorted(os.listdir(noisy_dir))
        
    def __len__(self):
        return min(len(self.clean_files), len(self.noisy_files))
    
    def __getitem__(self, idx):
        clean_path = os.path.join(self.clean_dir, self.clean_files[idx])
        noisy_path = os.path.join(self.noisy_dir, self.noisy_files[idx])
        
        clean_img = Image.open(clean_path).convert('RGB')
        noisy_img = Image.open(noisy_path).convert('RGB')
        
        if self.transform:
            clean_img = self.transform(clean_img)
            noisy_img = self.transform(noisy_img)
            
        return noisy_img, clean_img

class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        nc = 128
        nc4 = int(nc / 4)  # 32

        # Encoder
        self.enc = nn.Sequential(
            # 50x200, 3 channels
            nn.Conv2d(3, nc, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 25x100
            nn.Conv2d(nc, nc4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 12x50
            nn.Conv2d(nc4, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Latent space
        self.fc_mu = nn.Linear(32 * 12 * 50, latent_dim)
        self.fc_var = nn.Linear(32 * 12 * 50, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 32 * 12 * 50)
        
        self.dec = nn.Sequential(
            # 12x50
            nn.Conv2d(32, nc4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # 24x100
            nn.Conv2d(nc4, nc, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # 48x200
            nn.Conv2d(nc, 3, kernel_size=5, stride=1, padding=(3, 2)),
            # 50x200
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.enc(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 32, 12, 50)
        return self.dec(x)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

def loss_function(recon_x, x, mu, log_var):
    # sum over pixels, then divide by batch_size → per-sample loss
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return (BCE + KLD) / x.size(0)

# Hyperparameters
class Config:
    def __init__(self):
        # Data parameters
        # self.image_size = (50, 200)
        self.image_size = (64, 128)
        self.batch_size = 32
        self.num_workers = 4
        self.train_ratio = 0.8
        self.random_seed = 42
        
        # Model parameters
        self.latent_dim = 128
        self.nc = 128
        self.nc4 = int(self.nc / 4)
        
        # Training parameters
        self.epochs = 50
        self.learning_rate = 1e-4
        self.warmup_epochs = 5
        self.early_stop_patience = 10
        self.min_lr = 1e-6
        
        # Logging parameters
        self.log_interval = 100
        self.vis_interval = 10
        self.num_vis_samples = 5
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = os.path.join('outputs', f'run_{timestamp}')
        self.model_dir = os.path.join(self.output_dir, 'models')
        
        # Data directories
        self.clean_dir = 'sampled_license_plates_10k'
        self.noisy_dir = 'augmented_license_plates_10k'
        
    def create_dirs(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        
    def save_config(self):
        config_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('__')}
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=4)

def train_vae():
    config = Config()
    # Set random seed
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    
    # Create directories
    config.create_dirs()
    config.save_config()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
    ])

    # Create dataset and split
    full_dataset = LicensePlateDataset(
        clean_dir=config.clean_dir,
        noisy_dir=config.noisy_dir,
        transform=transform
    )
    
    train_size = int(config.train_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )

    # Initialize model
    model = VAE(latent_dim=config.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Learning rate schedulers
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=config.warmup_epochs * len(train_loader)
    )
    
    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=(config.epochs - config.warmup_epochs) * len(train_loader),
        eta_min=config.min_lr
    )

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(config.epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch_idx, (noisy_data, clean_data) in enumerate(train_loader):
            noisy_data = noisy_data.to(device)
            clean_data = clean_data.to(device)
            
            optimizer.zero_grad()
            recon_batch, mu, log_var = model(noisy_data)
            loss = loss_function(recon_batch, clean_data, mu, log_var)
            
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
            # Update learning rate
            if epoch < config.warmup_epochs:
                warmup_scheduler.step()
            else:
                main_scheduler.step()

            if batch_idx % config.log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(noisy_data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(noisy_data):.6f}')
        
        avg_train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        print(f'====> Epoch: {epoch} Average train loss: {avg_train_loss:.4f}')

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for noisy_data, clean_data in test_loader:
                noisy_data = noisy_data.to(device)
                clean_data = clean_data.to(device)
                recon_batch, mu, log_var = model(noisy_data)
                val_loss += loss_function(recon_batch, clean_data, mu, log_var).item()
        
        avg_val_loss = val_loss / len(test_loader.dataset)
        val_losses.append(avg_val_loss)
        print(f'====> Epoch: {epoch} Average val loss: {avg_val_loss:.4f}')

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(config.model_dir, 'best_model.pth'))
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config.early_stop_patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break

        # Visualization
        if epoch % config.vis_interval == 0:
            model.eval()
            with torch.no_grad():
                test_iter = iter(test_loader)

                # grab batch 0 and batch 1
                for batch_idx in range(2):
                    noisy_data, clean_data = next(test_iter)
                    noisy = noisy_data[:config.num_vis_samples].to(device)
                    clean = clean_data[:config.num_vis_samples].to(device)

                    recon_batch, _, _ = model(noisy)

                    # stack noisy / recon / clean and make a grid
                    grid = make_grid(
                        torch.cat([noisy, recon_batch, clean], dim=0),
                        nrow=config.num_vis_samples
                    )
                    save_image(
                        grid,
                        os.path.join(
                            config.output_dir,
                            f'reconstruction_epoch_{epoch}_batch{batch_idx}.png'
                        )
                    )



    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    with open(os.path.join(config.output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f)

    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(config.output_dir, 'training_curves.png'))
    plt.close()

if __name__ == '__main__':
    train_vae()
