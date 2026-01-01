import os
import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

TARGET_NAMES = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
TARGET_WEIGHTS = [0.1, 0.1, 0.1, 0.2, 0.5]
DATA_SPLIT_SEED = 42

class CSIRODataset(Dataset):
    def __init__(self, data_dir: str, img_dir: str, used_imgs: list, transform=None):
        self.data_dir = data_dir
        self.img_dir = img_dir
        self.used_imgs = used_imgs
        self.transform = transform
        self.df = pd.read_csv(os.path.join(data_dir, "train.csv"))
        self.df.loc[:,"target_log1p"] = np.log1p(self.df["target"].values)
    
    def __len__(self):
        return len(self.used_imgs)
    
    def __getitem__(self, idx):
        current_img = self.used_imgs[idx]

        # Load inpute image
        img_path = os.path.join(self.data_dir, self.img_dir, current_img)
        input_image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            input_image = self.transform(input_image)
        
        # load target from df
        current_df = self.df[self.df["image_path"] == "train/" + current_img]
        target = torch.zeros(5, dtype=torch.float32)
        for i, tname in enumerate(TARGET_NAMES):
            target[i] = current_df.loc[current_df["target_name"] == tname, "target"].values[0]
        
        return input_image, target
    
    def get_yw(self):
        used_df = self.df[self.df["image_path"].isin(["train/" + img for img in self.used_imgs])]
        yws_list = []
        for tname in TARGET_NAMES:
            mean_val = used_df[used_df["target_name"] == tname]["target"].mean()
            yws_list.append(float(mean_val))  # Convert to Python float
        return torch.tensor(yws_list, dtype=torch.float32)

    def get_yw_log1p(self):
        used_df = self.df[self.df["image_path"].isin(["train/" + img for img in self.used_imgs])]
        yws_list = []
        for tname in TARGET_NAMES:
            mean_val = used_df[used_df["target_name"] == tname]["target_log1p"].mean()
            yws_list.append(float(mean_val))  # Convert to Python float
        return torch.tensor(yws_list, dtype=torch.float32)

class CSIRODataModule:
    def __init__(self, data_dir="data", img_dir="train", image_resize=(512, 1024), train_split=0.9):
        self.data_dir = data_dir
        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.Resize(image_resize),  # (height, width)
            transforms.ToTensor(),
        ])

        # list all images and create train-val split
        all_imgs = os.listdir(os.path.join(data_dir, img_dir))
        random.seed(DATA_SPLIT_SEED)
        random.shuffle(all_imgs)
        split_idx = int(len(all_imgs) * train_split)
        self.train_imgs = all_imgs[:split_idx]
        self.val_imgs = all_imgs[split_idx:]

    def setup(self):
        self.train_dataset = CSIRODataset(
            self.data_dir, self.img_dir, 
            self.train_imgs, 
            transform=self.transform
        )
        
        self.val_dataset = CSIRODataset(
            self.data_dir, self.img_dir, 
            self.val_imgs, 
            transform=self.transform
        )

    def get_train_yw(self):
        return self.train_dataset.get_yw()
    
    def get_val_yw(self):
        return self.val_dataset.get_yw()
    
    def get_train_yw_log1p(self):
        return self.train_dataset.get_yw_log1p()
    
    def get_val_yw_log1p(self):
        return self.val_dataset.get_yw_log1p()

    def train_dataloader(self, batch_size=8, num_workers=4):
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

    def val_dataloader(self, batch_size=8, num_workers=4):
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
