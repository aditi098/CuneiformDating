import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import torch
from PIL import Image
import os
import json
import numpy as np
import random
from utils import * 

config_data = read_file("./configs/default.json") #TODO: find a way to pick this up from Experiment

with open(config_data["period_to_label_mapping_path"], 'r') as f:
    period_to_label = json.load(f)

with open(config_data["train_ids_class_wise_path"], 'r') as f:
    train_ids_class_wise = json.load(f)
    
    
class CuneiformDataset(Dataset):
    def __init__(self, pid_list, csv_file, root_dir_images, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.catalogue = pd.read_csv(csv_file)
        self.pid_list = pid_list
        self.root_dir = root_dir_images
        self.transform = transform

    def __len__(self):
        return len(self.pid_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # pid = self.catalogue.iloc[idx]["id"]
        pid = self.pid_list[idx]
        image_name = "P"+ str(pid).zfill(6)+".jpg"
        image_path = os.path.join(self.root_dir,image_name)
        
        
        image = Image.open(image_path)
        period = self.catalogue.loc[self.catalogue["id"] == pid]["period.period"].item()
        label = period_to_label[period]
        
        if self.transform:
            image = self.transform(image)

        sample = {'image': image,
                  'label': label,
                  'period': period,
                  'pid':pid,
                 }
        return sample
    
    
def getDataTransforms():
    
    data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.Grayscale(num_output_channels=3)

    ]),
    'val': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.Grayscale(num_output_channels=3)
    ]),
    }
    
    return data_transforms


    
def getDataloaders(catalogue_csv_path, root_dir_images, train_val_test_split_path, batch_size, class_size, takeAllTrain = False):
    
    data_transforms = getDataTransforms()
    
    with open(train_val_test_split_path, 'r') as f:
        splits = json.load(f)
    train_pid_list = splits["train"]
    val_pid_list = splits["val"]
    test_pid_list = splits["test"]
    
    # print(len(train_pid_list), len(val_pid_list), len(test_pid_list))
    
    if not takeAllTrain:
        train_pid_list = []

        for key, value in train_ids_class_wise.items():
            train_pid_list = train_pid_list + value[:class_size]
    
    
    train_dataset = CuneiformDataset(pid_list = train_pid_list, csv_file=catalogue_csv_path, 
                                   root_dir_images=root_dir_images, transform=data_transforms['train'])
    
    val_dataset = CuneiformDataset(pid_list = val_pid_list, csv_file=catalogue_csv_path, 
                                   root_dir_images=root_dir_images, transform=data_transforms['val'])
    
    
    test_dataset = CuneiformDataset(pid_list = test_pid_list, csv_file=catalogue_csv_path, 
                                   root_dir_images=root_dir_images, transform=data_transforms['val'])
    
    # print(train_dataset[0])
    train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    return (train_dataset_loader, val_dataset_loader, test_dataset_loader)

def upsample(idlist, size):
    out = []
    while len(out)<size:
        out = out + idlist
    random.shuffle(out)
    return out[:size]


#This returns a list of train dataloaders, a single val dataloader and a single test dataloader
def getBalancedDataloaders(catalogue_csv_path, root_dir_images, train_val_test_split_path, batch_size, class_size, max_loaders = 10):
    
    data_transforms = getDataTransforms()
    
    with open(train_val_test_split_path, 'r') as f:
        splits = json.load(f)
    train_pid_list = splits["train"]
    val_pid_list = splits["val"]
    test_pid_list = splits["test"]
    
    val_dataset = CuneiformDataset(pid_list = val_pid_list, csv_file=catalogue_csv_path, 
                                   root_dir_images=root_dir_images, transform=data_transforms['val'])
    
    
    test_dataset = CuneiformDataset(pid_list = test_pid_list, csv_file=catalogue_csv_path, 
                                   root_dir_images=root_dir_images, transform=data_transforms['val'])
    
    val_dataset_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    train_dataset_loaders = []

    classesParsed = np.zeros((len(train_ids_class_wise.keys())))
    current_indices = np.zeros((len(train_ids_class_wise.keys())))
    
    allClassesParsed = np.all(classesParsed)
    
    while(allClassesParsed==False and len(train_dataset_loaders)< max_loaders):
        train_pid_list = []
        
        for label, idList in train_ids_class_wise.items():
            label = int(label)
            if(len(idList)<class_size):
                train_pid_list = train_pid_list + upsample(idList, class_size)
                classesParsed[label] = 1
            else:
                start = int(current_indices[label])
                end = int(current_indices[label]+class_size)
                curr_list = idList[start:end]
                if end > len(idList):
                    classesParsed[label] = 1
                    start = 0
                    end = class_size - len(curr_list)
                    curr_list = curr_list+ idList[start:end]
                train_pid_list = train_pid_list + curr_list
                current_indices[label] = end
            
        train_dataset = CuneiformDataset(pid_list = train_pid_list, csv_file=catalogue_csv_path, 
                                   root_dir_images=root_dir_images, transform=data_transforms['train']) 
        train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_dataset_loaders.append(train_dataset_loader)
            
        allClassesParsed = np.all(classesParsed)
    
    return train_dataset_loaders, val_dataset_loader, test_dataset_loader
                    
        
            
