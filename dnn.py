import torchvision,torch
from dataloader import *
from torch import nn
import torch.nn.functional as F


class DNN(nn.Module):
    def __init__(self,num_classes=2,subsample_size=(64,64)):
        super(DNN,self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3*64*64, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5), 
     
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            
     
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
 
            nn.Linear(128, num_classes)
            )


    def forward(self, x):
        x=F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
        x = self.model(x)
        return x
    

