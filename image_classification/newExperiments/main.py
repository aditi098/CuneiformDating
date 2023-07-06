import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import getDataloaders
from model_factory import ResNet101
from experiment import train_model, test_model


batch_size = 16
num_epochs = 7
lr = 0.0001
weight_decay = 1e-1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_size = 300

def main(balanced = True):
    
    catalogue_csv_path = '../../../full_data/expanded_catalogue.csv'
    root_dir_images='../../../full_data/segmented_images/'
    train_val_test_split_path = "../../train_val_test_split.json"
    batch_size = 16
    
    
    resnet101_model = ResNet101().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(resnet101_model.parameters(), lr=lr, weight_decay=weight_decay)
    
    
    if(!balanced):
        (train_dataset_loader, val_dataset_loader, test_dataset_loader) = getDataloaders(
            catalogue_csv_path, root_dir_images,train_val_test_split_path, batch_size, class_size)
        print(len(train_dataset_loader), len(val_dataset_loader), len(test_dataset_loader))
        resnet101_model = train_model(resnet101_model, train_dataset_loader, 
                                      val_dataset_loader, optimizer, num_epochs, criterion)

    else:
        (train_dataset_loaders, val_dataset_loader, test_dataset_loader) = getBalancedDataloaders(
            catalogue_csv_path, root_dir_images, train_val_test_split_path, batch_size, class_size, 10)
        print(len(train_dataset_loaders), len(train_dataset_loaders[0]) ,len(val_dataset_loader), len(test_dataset_loader))

        resnet101_model = train_model_with_balanced_loaders(
            resnet101_model, train_dataset_loaders, val_dataset_loader, optimizer, num_epochs, criterion, device)
        test_model(resnet101_model, test_dataset_loader, optimizer, criterion, device)
        