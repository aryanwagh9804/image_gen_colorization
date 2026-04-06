import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage import color, metrics
import gc
import shutil
from tqdm import tqdm

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_PATH = "dataset/train"
TEST_PATH = "dataset/test"
EPOCHS = 100
BATCH_SIZE = 16
LR = 0.002

# 1. Clean up & Verify Dataset
def cleanup_dataset():
    print("🧹 Cleaning up dataset...")
    for d in [TRAIN_PATH, TEST_PATH]:
        for f in os.listdir(d):
            fpath = os.path.join(d, f)
            if not f.lower().endswith('.jpg') or os.path.getsize(fpath) < 2000:
                os.remove(fpath)
    print(f"✅ Cleaned. Train: {len(os.listdir(TRAIN_PATH))}, Test: {len(os.listdir(TEST_PATH))}")

class ColorizationNet(nn.Module):
    def __init__(self):
        super(ColorizationNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 2, kernel_size=3, padding=1), nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class ColorDataset(Dataset):
    def __init__(self, path, augment=False):
        self.path = path
        self.files = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith('.jpg')]
        self.augment = augment
        self.aug_transform = transforms.Compose([
            transforms.RandomResizedCrop(128, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3)
        ])
        self.base_transform = transforms.Compose([transforms.Resize((128, 128))])

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.files[idx]).convert('RGB')
        except:
            return self.__getitem__((idx + 1) % len(self))
        if self.augment: img = self.aug_transform(img)
        else: img = self.base_transform(img)
        img_np = np.array(img)
        lab = color.rgb2lab(img_np)
        L = lab[:, :, 0] / 50.0 - 1.0
        AB = lab[:, :, 1:] / 128.0
        L = torch.FloatTensor(L).unsqueeze(0)
        AB = torch.FloatTensor(AB).permute(2, 0, 1)
        return L, AB

def train_model(train_loader, name="Model"):
    print(f"🚀 Training {name} for {EPOCHS} epochs...")
    model = ColorizationNet().to(DEVICE)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    for epoch in range(EPOCHS):
        model.train()
        for L, AB in train_loader:
            L, AB = L.to(DEVICE), AB.to(DEVICE)
            optimizer.zero_grad()
            output = model(L)
            loss = criterion(output, AB)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"[{name}] Epoch {epoch+1}/{EPOCHS} complete.")
            gc.collect()
            torch.cuda.empty_cache()
    return model

def lab_to_rgb(L, AB):
    L_np = (L.detach().cpu().numpy().squeeze() + 1.0) * 50.0
    AB_np = AB.detach().cpu().numpy().squeeze().transpose(1, 2, 0) * 128.0
    lab = np.zeros((128, 128, 3))
    lab[:, :, 0] = L_np
    lab[:, :, 1:] = AB_np
    return (color.lab2rgb(lab) * 255).astype(np.uint8)

def run_experiment():
    cleanup_dataset()
    
    # Data
    ds_base = ColorDataset(TRAIN_PATH, augment=False)
    loader_base = DataLoader(ds_base, batch_size=BATCH_SIZE, shuffle=True)
    
    ds_aug = ColorDataset(TRAIN_PATH, augment=True)
    loader_aug = DataLoader(ds_aug, batch_size=BATCH_SIZE, shuffle=True)
    
    # Train
    model_baseline = train_model(loader_base, "Baseline")
    model_augmented = train_model(loader_aug, "Augmented")
    
    # Eval
    test_ds = ColorDataset(TEST_PATH, augment=False)
    L, AB_true = test_ds[0]
    L_batch = L.unsqueeze(0).to(DEVICE)
    
    model_baseline.eval()
    model_augmented.eval()
    with torch.no_grad():
        p_base = model_baseline(L_batch)
        p_aug = model_augmented(L_batch)
        
    rgb_true = lab_to_rgb(L, AB_true)
    rgb_base = lab_to_rgb(L, p_base[0])
    rgb_aug = lab_to_rgb(L, p_aug[0])
    
    psnr_base = metrics.peak_signal_noise_ratio(rgb_true, rgb_base)
    psnr_aug = metrics.peak_signal_noise_ratio(rgb_true, rgb_aug)
    ssim_base = metrics.structural_similarity(rgb_true, rgb_base, channel_axis=2)
    ssim_aug = metrics.structural_similarity(rgb_true, rgb_aug, channel_axis=2)
    
    # Plot
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(rgb_true); plt.title("Ground Truth"); plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(rgb_base); plt.title(f"Baseline\nPSNR: {psnr_base:.2f}, SSIM: {ssim_base:.3f}"); plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(rgb_aug); plt.title(f"Augmented\nPSNR: {psnr_aug:.2f}, SSIM: {ssim_aug:.3f}"); plt.axis('off')
    
    plt.suptitle("Task 4: Evidence of Data Augmentation Benefits", fontsize=16)
    plt.tight_layout()
    plt.savefig("scientific_proof_final.png")
    print(f"📊 Final Results Saved to scientific_proof_final.png")
    print(f"PSNR - Baseline: {psnr_base:.2f}, Augmented: {psnr_aug:.2f}")

if __name__ == "__main__":
    run_experiment()
