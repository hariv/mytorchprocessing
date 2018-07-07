from __future__ import print_function, division
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import os
from torch.autograd import Variable
from PIL import Image
from labelParser import LabelParser

class Classifier():
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

	def __init__(self, category, model="vgg19"):
		self.model = torchvision.models.__dict__[model](pretrained=True)
		self.category = category

	def readImage(self, path):
		if(os.path.exists(path)):
			return Image.open(path)
		else:
			raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)

	def prepare(self, img):
		img = Classifier.preprocess(img)
		img.unsqueeze_(0)
		return img
	
	def getEnergy(self, img):
		return self.model(img)

	def getProbability(self, energy):
		return nn.functional.softmax(energy)

	def getPredictionProbability(self, img):
		return self.getProbability(self.getEnergy(img))

	def loadLabels(self):
		labelParser = LabelParser('./labels.txt')
		labelParser.parseLabels()
		self.labels = labelParser.getLabels()

classifier = Classifier(category="dog")
image = classifier.readImage("/Users/harivenugopalan/Downloads/1.jpg")
image = classifier.prepare(image)
print(classifier.getPredictionProbability(image))