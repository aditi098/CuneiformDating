import tqdm
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

batch_size = 16
num_epochs = 7
lr = 0.0001
weight_decay = 1e-1
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def train_model(model, train_data_loader, val_data_loader, optimizer, num_epochs, criterion, device):
    train_dataset_size = len(train_data_loader)
    val_dataset_size = len(val_data_loader)
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 100)

        model.train()
        
        running_loss = 0.0
        train_true_labels = []
        train_pred_labels = []
        
        for idx, sample in tqdm.tqdm(enumerate(train_data_loader)):
            inputs = sample["image"]
            labels = sample["label"]
            train_true_labels = train_true_labels + labels.tolist()
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                logits = outputs.detach().cpu().numpy()
                train_pred_labels = train_pred_labels+ np.argmax(logits, axis =1).flatten().tolist()
                
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
        epoch_train_loss = running_loss / train_dataset_size
        epoch_train_accuracy = accuracy_score(train_true_labels, train_pred_labels)
        print('Train Loss: {:.4f}'.format(epoch_train_loss), 'Train Accuracy: ',epoch_train_accuracy )
        
        
        model.eval()
        
        
        running_loss = 0.0
        val_true_labels = []
        val_pred_labels = []
        for idx, sample in tqdm.tqdm(enumerate(val_data_loader)):
            inputs = sample["image"]
            labels = sample["label"]
            val_true_labels = val_true_labels + labels.tolist()
            inputs = inputs.to(device)
            labels = labels.to(device)
    
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                logits = outputs.detach().cpu().numpy()
                val_pred_labels = val_pred_labels+ np.argmax(logits, axis =1).flatten().tolist()
                loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
        epoch_val_loss = running_loss / val_dataset_size
        epoch_val_accuracy = accuracy_score(val_true_labels, val_pred_labels)
        print('Val Loss: {:.4f}'.format(epoch_val_loss), 'Val Accuracy: ',epoch_val_accuracy )
        
    return model


def train_model_with_balanced_loaders(model, train_data_loaders, val_data_loader, optimizer, num_epochs, criterion, device):
    
    val_dataset_size = len(val_data_loader)
    no_of_loaders = len(train_data_loaders)
    
    for epoch in range(num_epochs):
        train_data_loader = train_data_loaders[epoch%no_of_loaders]
        train_dataset_size = len(train_data_loader)
        
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 100)

        model.train()
        
        running_loss = 0.0
        train_true_labels = []
        train_pred_labels = []
        
        for idx, sample in tqdm.tqdm(enumerate(train_data_loader)):
            inputs = sample["image"]
            labels = sample["label"]
            train_true_labels = train_true_labels + labels.tolist()
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                logits = outputs.detach().cpu().numpy()
                train_pred_labels = train_pred_labels+ np.argmax(logits, axis =1).flatten().tolist()
                
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
        epoch_train_loss = running_loss / train_dataset_size
        epoch_train_accuracy = accuracy_score(train_true_labels, train_pred_labels)
        print('Train Loss: {:.4f}'.format(epoch_train_loss), 'Train Accuracy: ',epoch_train_accuracy )
        
        
        model.eval()
        
        
        running_loss = 0.0
        val_true_labels = []
        val_pred_labels = []
        for idx, sample in tqdm.tqdm(enumerate(val_data_loader)):
            inputs = sample["image"]
            labels = sample["label"]
            val_true_labels = val_true_labels + labels.tolist()
            inputs = inputs.to(device)
            labels = labels.to(device)
    
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                logits = outputs.detach().cpu().numpy()
                val_pred_labels = val_pred_labels+ np.argmax(logits, axis =1).flatten().tolist()
                loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
        epoch_val_loss = running_loss / val_dataset_size
        epoch_val_accuracy = accuracy_score(val_true_labels, val_pred_labels)
        print('Val Loss: {:.4f}'.format(epoch_val_loss), 'Val Accuracy: ',epoch_val_accuracy )
        
    return model



def test_model(model, test_dataset_loader, optimizer, criterion, device):

    running_loss = 0.0
    test_dataset_size = len(test_dataset_loader)
    test_true_labels = []
    test_pred_labels = []
    for idx, sample in tqdm.tqdm(enumerate(test_dataset_loader)):
        inputs = sample["image"]
        labels = sample["label"]
        test_true_labels = test_true_labels + labels.tolist()
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            logits = outputs.detach().cpu().numpy()
            test_pred_labels = test_pred_labels+ np.argmax(logits, axis =1).flatten().tolist()
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
    epoch_test_loss = running_loss / test_dataset_size
    epoch_test_accuracy = accuracy_score(test_true_labels, test_pred_labels)
    print('Test Loss: {:.4f}'.format(epoch_test_loss), 'Test Accuracy: ',epoch_test_accuracy )
    
    return test_true_labels, test_pred_labels