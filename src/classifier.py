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

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize(256),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   normalize
])

im = Image.open("/Users/harivenugopalan/Downloads/1.jpg")
im = preprocess(im)
im.unsqueeze_(0)
im = Variable(im)
#im = np.array(Image.open('/Users/harivenugopalan/Downloads/1.jpg').convert("RGB"))
#im = Variable(torch.Tensor(im))
#im = im.unsqueeze_(0)
#im = loader(im).float()
#im = Variable(im, requires_grad=True)
print(classifier.predict(im))