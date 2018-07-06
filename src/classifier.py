from __future__ import print_function, division
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import os
import copy
import json
from torch.autograd import Variable
from PIL import Image
import numpy as np

class Classifier():
	
	def __init__(self, category, model="alexnet"):
		self.model = torchvision.models.__dict__[model](pretrained=True)
		self.category = category

	def predict(self, img):
		prediction = self.model(img)
		return prediction

classifier = Classifier(category="dog")
#imsize = 256*6*6
#loader = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])

im = np.array(Image.open('/Users/harivenugopalan/Downloads/1.jpg').convert("RGB"))
im = Variable(torch.Tensor(im))
im = im.unsqueeze(0)
#im = loader(im).float()
#im = Variable(im, requires_grad=True)
print("hi")
print(classifier.predict(im))