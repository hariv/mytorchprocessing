from __future__ import print_function, division
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import os
import numpy as np
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

	def __init__(self, categories, model="alexnet", labelsPath="./labels.txt"):
		self.model = torchvision.models.__dict__[model](pretrained=True)
		self.categories = np.asarray(categories)
		self.labelsPath = labelsPath
		self.loadLabels()

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
		return nn.functional.softmax(energy,dim=1)

	def getPredictionProbability(self, img):
		return self.getProbability(self.getEnergy(img))

	def loadLabels(self):
		labelParser = LabelParser(self.labelsPath)
		labelParser.parseLabels()
		self.labels = np.asarray(labelParser.getLabels())

	def getTopProbability(self, probability):
		return self.getTopKProbabilities(probability, 1)

	def getTopProbabilityIndex(self, probability):
		return self.getTopKProbabilityIndices(probability, 1)

	def getTopClassLabel(self, probability):
		return self.getTopKClassLabels(probability, 1)

	def getTopKProbabilities(self, probability, k):
		maxProbability, index = torch.topk(probability, k, dim=1)
		return maxProbability[0].detach().numpy()

	def getTopKProbabilityIndices(self, probability, k):
		maxProbability, index = torch.topk(probability, k, dim=1)
		return index[0].numpy()

	def getTopKClassLabels(self, probability, k):
		indices = self.getTopKProbabilityIndices(probability, k)
		return self.labels[indices]

	def getCategoryProbabilities(self, probability):
		indices = np.where(np.isin(self.labels, self.categories))[0]
		probabilities = probability.detach().numpy()[0]
		return probabilities[indices]