import torchvision
from torch import nn
from torchvision import models

class Resnet(nn.Module):
    def __init__(self,num_classes=2):
        super(Resnet,self).__init__()
        self.model = models.resnet18(pretrained=True)
        num_features=self.model.fc.in_features

        self.model.fc=nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
            )

        for param in self.model.parameters():
            param.requires_grad = False
        
        for param in self.model.fc.parameters():
            param.requires_grad = True
 
    def forward(self, x):
        x = self.model(x)
        return x

