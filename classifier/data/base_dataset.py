import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os

class BaseDataset(data.Dataset):

    def __init__(self, opt):
        self.root = opt.dataroot
        
        self.preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()
                ])
        
        if(opt.isTrain):
            self.dataset = datasets.ImageFolder(os.path.join(self.root, 'train'), self.preprocess)
        else:
            self.dataset = datasets.ImageFolder(os.path.join(self.root, 'test'), self.preprocess)
        
        self.compute_statistics()
        
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        
        self.preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                self.normalize])
        
        if(opt.isTrain):
            self.dataset = datasets.ImageFolder(os.path.join(self.root, 'train'), self.preprocess)
        else:
            self.dataset = datasets.ImageFolder(os.path.join(self.root, 'test'), self.preprocess)
        
    def compute_statistics(self):
        sumEl = 0.0
        countEl = 0
        
        for img, _ in self.dataset:
            sumel += img.sum([1, 2])
            countel += torch.numel(img[0])
        
        self.mean = sumel/countel
            
        for img, _ in self.dataset:
            img = (img - mean.unsqueeze(1).unsqueeze(1))**2
            sumel += img.sum([1, 2])
            countel += torch.numel(img[0])
        self.std = torch.sqrt(sumel/countel)
        
