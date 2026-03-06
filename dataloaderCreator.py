import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from createDf import build_master_df # decided to pull this out and put it somewhere else.

class ChestXRayDataset(Dataset):
  def __init__(self, records, image_dir, transform=None):
    """
    records    : a dataframe slice (post train test split)
    image_dir : path to folder containing all .dcm.png files
    transform : torchvision transforms pipeline
    """
    self.records = records.reset_index(drop=True)
    self.image_dir = image_dir
    self.transform = transform

  def __len__(self):
    return len(self.records)

  def __getitem__(self, idx):
    row = self.records.iloc[idx]

    # for image stuff
    img_path = os.path.join(self.image_dir, row['filename'])
    image = Image.open(img_path).convert('RGB') # X-ray gray scale but resnet doesn't like that
    if self.transform:
      image = self.transform(image)

    # label time
    label = torch.tensor(row["pneumonia_label"], dtype=torch.float32)

    return image, label



def get_dataloaders(projections_path, reports_path, image_dir, test_size=0.2,
                    batch_size=32, image_size=224, num_workers=0):

  df = build_master_df(projections_path, reports_path)

  train_df, test_df = train_test_split(
    df,
    test_size=test_size,
    random_state=42,
    stratify=df["pneumonia_label"], # maintain balanced split between train test for convergence.
  )

  # DOING some standard imageNet transformations
  train_transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    # ok what does this mean and why do pycharm and claude agree thats what the values should be
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
  ])

  test_transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
  ])

  train_dataset = ChestXRayDataset(train_df, image_dir, transform=train_transforms)
  test_dataset = ChestXRayDataset(test_df, image_dir, transform=test_transforms)

  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

  # get the df for NLP purposes, will probably find a better way to do it later.
  return train_loader, test_loader, df