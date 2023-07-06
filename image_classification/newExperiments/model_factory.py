import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
from torchinfo import summary

class ResNet101(nn.Module):
    def __init__(self):
        super().__init__()

        resnet101_weights = torchvision.models.ResNet101_Weights.DEFAULT
        self.resnet = torchvision.models.resnet101(weights=resnet101_weights)
        
        # TODO: Try finetuning just the last few layers? Make earlier layers untrainable
        # do not train this part of the model
        # self.model_untrainable = nn.Sequential(
        #     *(list(self.resnet.children())[:-1]), nn.Flatten()
        # )
        
        self.resnet.fc = nn.Sequential(
            nn.BatchNorm1d(2048),
            nn.Dropout(p=0.25),
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=18),
        )
        
    def forward(self, x):
        return self.resnet(x)
    
def printModelSummary(model):
    print(summary(model=model, 
        input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape"
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
    ))