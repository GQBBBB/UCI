import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

def build_transform_classification(normalize, crop_size=224, resize=256, mode="train", test_augment=True):
    transformations_list = []

    if normalize.lower() == "imagenet":
      normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    elif normalize.lower() == "chestx-ray":
      normalize = transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])
    elif normalize.lower() == "cxr":
      normalize = transforms.Normalize([0.5180, 0.5180, 0.5181], [0.1959, 0.1959, 0.1959])
    elif normalize.lower() == "none":
      normalize = None
    else:
      print("mean and std for [{}] dataset do not exist!".format(normalize))
      exit(-1)
    if mode == "train":
      transformations_list.append(transforms.RandomResizedCrop(crop_size))
      transformations_list.append(transforms.RandomVerticalFlip())
      transformations_list.append(transforms.RandomHorizontalFlip())
      transformations_list.append(transforms.RandomRotation(7))
      transformations_list.append(transforms.ToTensor())
      if normalize is not None:
        transformations_list.append(normalize)
    elif mode == "valid":
      transformations_list.append(transforms.Resize((resize, resize)))
      transformations_list.append(transforms.CenterCrop(crop_size))
      transformations_list.append(transforms.ToTensor())
      if normalize is not None:
        transformations_list.append(normalize)
    elif mode == "test":
      if test_augment:
        transformations_list.append(transforms.Resize((resize, resize)))
        transformations_list.append(transforms.TenCrop(crop_size))
        transformations_list.append(
          transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        if normalize is not None:
          transformations_list.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
      else:
        transformations_list.append(transforms.Resize((resize, resize)))
        transformations_list.append(transforms.CenterCrop(crop_size))
        transformations_list.append(transforms.ToTensor())
        if normalize is not None:
          transformations_list.append(normalize)
    transformSequence = transforms.Compose(transformations_list)

    return transformSequence

class CXR_train(Dataset):
    def __init__(self, img_dir, label_file, mode="train"):
        self.img_dir = img_dir
        self.label_file = label_file
        self.mode = mode

        self.img_idx = []
        self.label = []

        if not os.path.isfile(self.label_file):
            print('Error: ' + self.label_file + 'does not exist!')

        with open(self.label_file, "r") as f:
            all_name = f.readlines()
            for name in all_name:
                name = name.strip('\n')
                self.img_idx.append(name)
            
                if name[:3] == 'cov':
                    self.label.append(2)
                elif name[:3] == 'nor':
                    self.label.append(0)
                elif name[:3] == 'pne':
                    self.label.append(1)
                else:
                    assert 0
                 
        self.augmentation = build_transform_classification(normalize="CXR", mode="train")

    def __len__(self):
        return len(self.img_idx)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.mode, self.img_idx[idx])
        
        if not os.path.isfile(img_path):
            print('Error: ' + img_path + ' does not exist!')
            assert 0

        image = Image.open(img_path).convert('RGB')

        label = self.label[idx]

        image = self.augmentation(image)

        return image, label


class CXR_val(Dataset):
    def __init__(self, img_dir, label_file, mode="train"):
        self.img_dir = img_dir
        self.label_file = label_file
        self.mode = mode

        self.img_idx = []
        self.label = []

        if not os.path.isfile(self.label_file):
            print('Error: ' + self.label_file + 'does not exist!')

        with open(self.label_file, "r") as f:
            all_name = f.readlines()
            for name in all_name:
                name = name.strip('\n')
                self.img_idx.append(name)
            
                if name[:3] == 'cov':
                    self.label.append(2)
                elif name[:3] == 'nor':
                    self.label.append(0)
                elif name[:3] == 'pne':
                    self.label.append(1)
                else:
                    assert 0
        
        self.augmentation = build_transform_classification(normalize="CXR", mode="valid")

    def __len__(self):
        return len(self.img_idx)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.mode, self.img_idx[idx])

        if not os.path.isfile(img_path):
            print('Error: ' + img_path + ' does not exist!')
            assert 0

        image = Image.open(img_path).convert('RGB')

        label = self.label[idx]

        image = self.augmentation(image)

        return image, label
        
        
class CXR_test(Dataset):
    def __init__(self, img_dir, label_file, multicrops=False, mode="validation"):
        self.img_dir = img_dir
        self.label_file = label_file
        self.mode = mode

        self.img_idx = []
        self.label = []

        if not os.path.isfile(self.label_file):
            print('Error: ' + self.label_file + 'does not exist!')

        with open(self.label_file, "r") as f:
            all_name = f.readlines()
            for name in all_name:
                name = name.strip('\n')
                self.img_idx.append(name)
            
                if name[:3] == 'cov':
                    self.label.append(2)
                elif name[:3] == 'nor':
                    self.label.append(0)
                elif name[:3] == 'pne':
                    self.label.append(1)
                else:
                    assert 0
        
        self.augmentation = build_transform_classification(normalize="CXR", mode="test", test_augment=multicrops)

    def __len__(self):
        return len(self.img_idx)

    def __getitem__(self, idx):
        img_path = self.img_dir + self.img_idx[idx]

        if not os.path.isfile(img_path):
            print('Error: ' + img_path + ' does not exist!')
            assert 0

        image = Image.open(img_path).convert('RGB')

        label = self.label[idx]

        #image = self.val_augmentation(image)
        image = self.augmentation(image)

        return image, label