import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision
import os

class BaseDataset(data.Dataset):

    def __init__(self, opt):
        self.root = opt.dataroot
        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if(opt.isTrain):
            self.dataset = torchvision.datasets.ImageFolder(os.path.join(self.root, 'train'), torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(), self.normalize]))
        else:
            self.dataset = torchvision.datasets.ImageFolder(os.path.join(self.root, 'test'), torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(), self.normalize]))    
        
