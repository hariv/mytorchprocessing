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
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        self.preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                self.normalize])
        
        if(opt.isTrain):
            self.dataset = datasets.ImageFolder(os.path.join(self.root, 'train'), self.preprocess)
        else:
            self.dataset = datasets.ImageFolder(os.path.join(self.root, 'test'), self.preprocess)    
        
