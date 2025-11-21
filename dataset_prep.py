from PIL import Image
import os
from torch.utils.data import DataLoader, Dataset
import random
from torchvision import transforms, utils
import numpy as np
from config import BATCH_SIZE
import matplotlib.pyplot as plt

# Custom Dataset for CelebA
class CelebADataset(Dataset):
    def __init__(self, img_dir, image_files, transform=None):
        self.img_dir = img_dir
        self.image_files = image_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


def get_data_loaders(image_directory, total_num, train_per, val_per, shuffle=True):
    # List and shuffle images
    all_images = [f for f in os.listdir(image_directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if shuffle:
        random.seed(42)
        random.shuffle(all_images)

    # Split into train/val/test sets
    num_total = total_num
    num_train = int(train_per * num_total)
    num_val = int(val_per * num_total)
    # num_test = num_total - num_train - num_val

    train_images = all_images[:num_train]
    val_images = all_images[num_train:num_train+num_val]
    test_images = all_images[num_train+num_val:]

    # ✅ Keep original resolution (178x218), just convert to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Datasets
    train_dataset = CelebADataset(img_dir=image_directory, image_files=train_images, transform=transform)
    val_dataset = CelebADataset(img_dir=image_directory, image_files=val_images, transform=transform)
    test_dataset = CelebADataset(img_dir=image_directory, image_files=test_images, transform=transform)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


def display_batch(loader):
    data_iter = iter(loader)
    images = next(data_iter)

    # Show first 9 images in grid
    img_grid = utils.make_grid(images[:9], nrow=3)

    np_img_grid = img_grid.numpy()
    plt.imshow(np.transpose(np_img_grid, (1, 2, 0)))  # (C,H,W) → (H,W,C)
    plt.axis('off')
    plt.show()
