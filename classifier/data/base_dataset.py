import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision
import os

class BaseDataset(data.Dataset):
    
    def __init__(self, opt):
        self.root = opt.dataroot
        if(opt.isTrain):
            self.dataset = torchvision.datasets.ImageFolder(os.path.join(self.root, 'train'))
        else:
            self.dataset = torchvision.datasets.ImageFolder(os.path.join(self.root, 'test'))
            
    
        
