from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image
import os
import torch

def transform_label(name):
    if name == "cat":
        label=torch.tensor(0)
    else:
        label=torch.tensor(1)

    return label


class MyData(Dataset):
    def __init__(self,data_path):
        self.data_path=data_path
        self.name_list=os.listdir(self.data_path)

    def __getitem__(self, index):
        transform_image=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([256,256])])
        
        image_name=self.name_list[index]
        image_item_path=os.path.join(self.data_path,image_name)
        img=transform_image(Image.open(image_item_path))
        label=transform_label(image_name.split(".")[0])
        
        return img,label
        
    def __len__(self):
        return len(self.name_list)
    
trainset=MyData("data/train")
valset=MyData("data/val")
train_loader=DataLoader(dataset=trainset,batch_size=16,shuffle=True)
val_loader=DataLoader(dataset=valset,batch_size=16,shuffle=True)

