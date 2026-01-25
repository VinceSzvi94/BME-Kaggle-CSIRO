import os
import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from PIL import Image
import torchvision.transforms as transforms

TARGET_NAMES = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
TARGET_WEIGHTS = [0.1, 0.1, 0.1, 0.2, 0.5]
# DATA_SPLIT_SEED = 42

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
        return yws_list

    # def get_yw_log1p(self):
    #     used_df = self.df[self.df["image_path"].isin(["train/" + img for img in self.used_imgs])]
    #     yws_list = []
    #     for tname in TARGET_NAMES:
    #         mean_val = used_df[used_df["target_name"] == tname]["target_log1p"].mean()
    #         yws_list.append(float(mean_val))  # Convert to Python float
    #     return yws_list

class CSIRODataModule:
    def __init__(self, data_dir="data", img_dir="train", image_resize=(448, 896), train_split=0.9, cv_fold_no=1):
        self.data_dir = data_dir
        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.Resize(image_resize),  # (height, width)
            transforms.ToTensor(),
        ])

        # list all images and create train-val split
        all_imgs = os.listdir(os.path.join(data_dir, img_dir))
        # random.seed(DATA_SPLIT_SEED) everything seeded outside!
        random.shuffle(all_imgs)
        split_idx = int(len(all_imgs) * train_split)
        self.all_imgs = set(all_imgs)
        self.train_imgs = all_imgs[:split_idx]
        self.val_imgs = all_imgs[split_idx:]
        self.cv_fold_no = cv_fold_no

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

        if self.cv_fold_no > 1:
            # Implement cross-validation split logic if needed
            df = pd.read_csv(os.path.join(self.data_dir, "train.csv"))
            df = df[df["image_path"].isin(["train/" + img for img in self.all_imgs])]
            df = df[df['target_name'] == TARGET_NAMES[0]]  # Use row for each pic
            # states = df['State'].unique()
            # species = df['Species'].unique()
            df.loc[:, 'strat_label'] = df['State'] + "_" + df['Species']
            df.loc[:, "image_name"] = df["image_path"].apply(lambda x: x.split("/")[1])

            # Create stratified k-fold splits
            skf = StratifiedKFold(n_splits=self.cv_fold_no, shuffle=True)
            
            # Store all folds as disjoint sets
            self.folds = []
            for _, fold_idx in skf.split(df['image_name'], df['strat_label']):
                fold_imgs = set(df.iloc[fold_idx]['image_name'].tolist())
                self.folds.append(fold_imgs)

    def get_train_yw(self):
        return self.train_dataset.get_yw()
    
    def get_val_yw(self):
        return self.val_dataset.get_yw()
    
    # def get_train_yw_log1p(self):
    #     return self.train_dataset.get_yw_log1p()
    
    # def get_val_yw_log1p(self):
    #     return self.val_dataset.get_yw_log1p()

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
    
    def get_kfolds(self, batch_size=8, num_workers=4):
        if self.cv_fold_no <= 1:
            return [(self.train_dataloader(batch_size, num_workers), self.val_dataloader(batch_size, num_workers), self.get_train_yw(), self.get_val_yw())]
        
        fold_dataloaders = []
        for fold_imgs in self.folds:
            rest_imgs = self.all_imgs - fold_imgs

            fold_train_dataset = CSIRODataset(
                self.data_dir, self.img_dir,
                list(rest_imgs),
                transform=self.transform
            )
            fold_val_dataset = CSIRODataset(
                self.data_dir, self.img_dir,
                list(fold_imgs),
                transform=self.transform
            )
            fold_dataloaders.append((
                DataLoader(
                    fold_train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=True
                ),
                DataLoader(
                    fold_val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True
                ),
                fold_train_dataset.get_yw(),
                fold_val_dataset.get_yw()
            ))
        return fold_dataloaders


