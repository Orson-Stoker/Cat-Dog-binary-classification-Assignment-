import torchvision,torch
from dataloader import *
from torch import nn
import torch.nn.functional as F
from torchvision import models

class RNN(nn.Module):
    def __init__(self,num_classes=2,num_layers=2,input_size=256*3,hidden_size=512):
        super(RNN,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.LSTM(
            input_size=input_size,  
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 128), 
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):

        batch_size = x.size(0)
        x = x.view(batch_size, 3, 256, 256)
        x = x.permute(0, 2, 1, 3)  # (batch_size, 256, 3, 256)
        x = x.contiguous().view(batch_size, 256, -1)  
    
        rnn_out, (hidden, cell) = self.rnn(x)  
        attention_weights = F.softmax(self.attention(rnn_out), dim=1)  # (batch_size, 256, 1)
        context_vector = torch.sum(attention_weights * rnn_out, dim=1)  # (batch_size, hidden_size*2)
    
        output = self.classifier(context_vector)
        return output
    