import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from dataloader import getDataloaders, getBalancedDataloaders
from model_factory import ResNet101
from utils import *
import datetime

class Experiment(object):
    def __init__(self, name, directory=None):
        config_data = read_file_in_dir("./configs", name + ".json")
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)
        
        self.__name = config_data["experiment_name"]
        self.__experiment_dir = directory if directory is not None else config_data["experiment_directory"]+"/"+ str(datetime.datetime.now())+"/"
        os.makedirs(self.__experiment_dir, exist_ok=True)
        print("Created directory", self.__experiment_dir)
        self.__device = torch.device("cuda:"+str(config_data["gpu_number"]) if torch.cuda.is_available() else "cpu")
        self.__epochs = config_data["num_epochs"]
        self.__catalogue_csv_path = config_data["catalogue_csv_path"]
        self.__root_dir_images = config_data["root_dir_images"]
        self.__train_val_test_split_path = config_data["train_val_test_split_path"]
        self.__batch_size = config_data["batch_size"]
        self.__max_loaders = config_data["max_loaders"]
        self.__lr = config_data["lr"]
        self.__weight_decay = config_data["weight_decay"]
        self.__class_size = config_data["class_size"]
        self.__balance_data = config_data["balance_data"]
        
        # Load Datasets
        if config_data["balance_data"]:
            (self.__train_dataset_loaders, 
             self.__val_dataset_loader, 
             self.__test_dataset_loader) = getBalancedDataloaders(self.__catalogue_csv_path, 
                                                                  self.__root_dir_images, self.__train_val_test_split_path, 
                                                                  self.__batch_size, self.__class_size, self.__max_loaders)
        
        else:
            (self.__train_dataset_loader, 
             self.__val_dataset_loader, 
             self.__test_dataset_loader) = getDataloaders(self.__catalogue_csv_path, self.__root_dir_images, 
                                                          self.__train_val_test_split_path, self.__batch_size, 
                                                          self.__class_size)


        
#         self.__training_losses = []
#         self.__val_losses = []
#         self.__best_model = (
#             None  # Save your best model in this field and use this in test method.
#         )

        self.__model = ResNet101().to(self.__device)
        self.__criterion = nn.CrossEntropyLoss()
        self.__optimizer = torch.optim.AdamW(self.__model.parameters(), lr=self.__lr, weight_decay=self.__weight_decay)
        
     
    def train(self):
        if self.__balance_data:
            print(len(self.__train_dataset_loaders[0]), len(self.__val_dataset_loader), len(self.__test_dataset_loader))
            self.train_model_with_balanced_loaders()
        else:
            print(len(self.__train_dataset_loader), len(self.__val_dataset_loader), len(self.__test_dataset_loader))
            self.train_model()
        self.__save_model()    
            
    def train_model(self):
        train_dataset_size = len(self.__train_dataset_loader)
        val_dataset_size = len(self.__val_dataset_loader)

        for epoch in range(self.__epochs):
            self.__log('Epoch {}/{}'.format(epoch + 1, self.__epochs))
            self.__log('-' * 100)

            self.__model.train()

            running_loss = 0.0
            train_true_labels = []
            train_pred_labels = []

            for idx, sample in tqdm.tqdm(enumerate(self.__train_dataset_loader)):
                inputs = sample["image"]
                labels = sample["label"]
                train_true_labels = train_true_labels + labels.tolist()
                inputs = inputs.to(self.__device)
                labels = labels.to(self.__device)

                self.__optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    outputs = self.__model(inputs)
                    logits = outputs.detach().cpu().numpy()
                    train_pred_labels = train_pred_labels+ np.argmax(logits, axis =1).flatten().tolist()

                    loss = self.__criterion(outputs, labels)
                    loss.backward()
                    self.__optimizer.step()

                running_loss += loss.item() * inputs.size(0)
            epoch_train_loss = running_loss / train_dataset_size
            epoch_train_accuracy = accuracy_score(train_true_labels, train_pred_labels)
            self.__log('Train Loss: {:.4f}'.format(epoch_train_loss)+'Train Accuracy: '+str(epoch_train_accuracy) )


            self.__model.eval()


            running_loss = 0.0
            val_true_labels = []
            val_pred_labels = []
            val_pids = []
            for idx, sample in tqdm.tqdm(enumerate(self.__val_dataset_loader)):
                inputs = sample["image"]
                labels = sample["label"]
                val_true_labels = val_true_labels + labels.tolist()
                inputs = inputs.to(self.__device)
                labels = labels.to(self.__device)
        
                self.__optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    outputs = self.__model(inputs)
                    logits = outputs.detach().cpu().numpy()
                    val_pred_labels = val_pred_labels+ np.argmax(logits, axis =1).flatten().tolist()
                    loss = self.__criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
            epoch_val_loss = running_loss / val_dataset_size
            epoch_val_accuracy = accuracy_score(val_true_labels, val_pred_labels)
            self.__log('Val Loss: {:.4f}'.format(epoch_val_loss)+'Val Accuracy: '+str(epoch_val_accuracy) )



    def train_model_with_balanced_loaders(self):

        val_dataset_size = len(self.__val_dataset_loader)
        no_of_loaders = len(self.__train_dataset_loaders)

        for epoch in range(self.__epochs):
            train_data_loader = self.__train_dataset_loaders[epoch%no_of_loaders]
            train_dataset_size = len(train_data_loader)

            self.__log('Epoch {}/{}'.format(epoch + 1, self.__epochs))
            self.__log('-' * 100)

            self.__model.train()

            running_loss = 0.0
            train_true_labels = []
            train_pred_labels = []

            for idx, sample in tqdm.tqdm(enumerate(train_data_loader)):
                inputs = sample["image"]
                labels = sample["label"]
                train_true_labels = train_true_labels + labels.tolist()
                inputs = inputs.to(self.__device)
                labels = labels.to(self.__device)

                self.__optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    outputs = self.__model(inputs)
                    logits = outputs.detach().cpu().numpy()
                    train_pred_labels = train_pred_labels+ np.argmax(logits, axis =1).flatten().tolist()

                    loss = self.__criterion(outputs, labels)
                    loss.backward()
                    self.__optimizer.step()

                running_loss += loss.item() * inputs.size(0)
            epoch_train_loss = running_loss / train_dataset_size
            epoch_train_accuracy = accuracy_score(train_true_labels, train_pred_labels)
            self.__log('Train Loss: {:.4f}'.format(epoch_train_loss)+'Train Accuracy: '+str(epoch_train_accuracy) )


            self.__model.eval()


            running_loss = 0.0
            val_true_labels = []
            val_pred_labels = []
            for idx, sample in tqdm.tqdm(enumerate(self.__val_dataset_loader)):
                inputs = sample["image"]
                labels = sample["label"]
                val_true_labels = val_true_labels + labels.tolist()
                inputs = inputs.to(self.__device)
                labels = labels.to(self.__device)

                self.__optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    outputs = self.__model(inputs)
                    logits = outputs.detach().cpu().numpy()
                    val_pred_labels = val_pred_labels+ np.argmax(logits, axis =1).flatten().tolist()
                    loss = self.__criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
            epoch_val_loss = running_loss / val_dataset_size
            epoch_val_accuracy = accuracy_score(val_true_labels, val_pred_labels)
            self.__log('Val Loss: {:.4f}'.format(epoch_val_loss)+'Val Accuracy: '+str(epoch_val_accuracy) )


    def test(self, val = False):
        running_loss = 0.0
        test_true_labels = []
        test_pred_labels = []
        test_pids = []
        dataloader = self.__val_dataset_loader if val else self.__test_dataset_loader
        test_dataset_size = len(dataloader)
        for idx, sample in tqdm.tqdm(enumerate(dataloader)):
            inputs = sample["image"]
            labels = sample["label"]
            pids = sample["pid"]
            test_true_labels = test_true_labels + labels.tolist()
            test_pids = test_pids + pids.tolist()
            inputs = inputs.to(self.__device)
            labels = labels.to(self.__device)

            self.__optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = self.__model(inputs)
                logits = outputs.detach().cpu().numpy()
                test_pred_labels = test_pred_labels+ np.argmax(logits, axis =1).flatten().tolist()
                loss = self.__criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
        epoch_test_loss = running_loss / test_dataset_size
        epoch_test_accuracy = accuracy_score(test_true_labels, test_pred_labels)
        self.__log('Test Loss: {:.4f}'.format(epoch_test_loss)+'Test Accuracy: '+str(epoch_test_accuracy) )

        predictions = {"pids":test_pids, "true labels": test_true_labels, "predicted labels":test_pred_labels}
        self.__savePredictions(predictions, val)
    
    
    def __save_model(self):
        model_name = self.__name + ".pt"
        root_model_path = os.path.join(self.__experiment_dir, model_name)
        model_dict = self.__model.state_dict()
        state_dict = {"model": model_dict, "optimizer": self.__optimizer.state_dict()}
        torch.save(state_dict, root_model_path)
        
    def load_experiment(self, model_path):
        state_dict = torch.load(model_path)
        self.__model.load_state_dict(state_dict["model"])
        self.__optimizer.load_state_dict(state_dict["optimizer"])
        
    def __log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.__experiment_dir, "all.log", log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name, log_str)
            
    def __savePredictions(self, predictions, val):
        filename = "val_predictions.json" if val else "test_predictions.json"
        with open(os.path.join(self.__experiment_dir, filename), 'w') as f:
            json.dump(predictions,f)
