import os
import csv
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm


# path
IMAGE_DIR_ROOT = 'tiny_imagenet_data/'
IMAGE_DIR_TRAIN = IMAGE_DIR_ROOT + 'train/'
IMAGE_DIR_VAL = IMAGE_DIR_ROOT + 'val/'
IMAGEDIR_TEST = IMAGE_DIR_ROOT + 'test/'

# encoder dict
label_dict = {}
for i, line in enumerate(open(IMAGE_DIR_ROOT+'/wnids.txt','r')):
    label_dict[line.replace('\n', '')] = i

inverted_label_dict = {v: k for k, v in label_dict.items()}

label_word_dict = {}
with open(IMAGE_DIR_ROOT+'label_list.txt', mode='r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    for row in reader:
        label_word_dict[row[0]] = row[1]
    
# train
filename_train = []
label_train = []
for label in os.listdir(IMAGE_DIR_TRAIN):
    for filename in os.listdir(IMAGE_DIR_TRAIN + label + '/images/'):
        filename_train.append(IMAGE_DIR_TRAIN + label + '/images/' + filename)
        label_train.append(label)

# val
filename_val = []
label_val = []
for i, line in enumerate(open(IMAGE_DIR_VAL+'/val_annotations.txt','r')):
    items = line.split('\t')
    filename_val.append(IMAGE_DIR_VAL + 'images/' + items[0])
    label_val.append(items[1])

# test
filename_test = []
for filename in os.listdir(IMAGEDIR_TEST + 'images/'):
    filename_test.append(IMAGEDIR_TEST + 'images/' + filename)

class ImageDataset(Dataset):
    def __init__(self, filename, label, label_dict,  transform, mode='train'):
        super().__init__()
        self.filename = filename
        self.label = label
        self.label_dict = label_dict
        self.transform = transform
        self.mode = mode
    
    def __len__(self):
        return len(self.filename)
    
    def __getitem__(self, idx):
        image  = Image.open(self.filename[idx]).convert('RGB')
        
        if self.mode == 'train' or self.mode == 'val':
            x = self.transform(image)
            y = self.label_dict[self.label[idx]]
        elif self.mode == 'test':
            x = self.transform(image)
            y = self.filename[idx]
            
        return x, y
    
data_transform = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
}


train_dataset = ImageDataset(filename_train, label_train, label_dict, data_transform['train'], 'train')
val_dataset = ImageDataset(filename_val, label_val, label_dict, data_transform['val'], 'val')
test_dataset = ImageDataset(filename_test, None, None, data_transform['val'], 'test')


train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)