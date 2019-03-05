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
	# Normalize input image.
	normalize = transforms.Normalize(
   		mean=[0.485, 0.456, 0.406],
   		std=[0.229, 0.224, 0.225]
		)
	
	# Preprocess inputs before passing to model. Static method.
	preprocess = transforms.Compose([
   		transforms.Resize(256),
   		transforms.CenterCrop(224),
   		transforms.ToTensor(),
   		normalize
		])
	
	def __init__(self, categories, labelsPath, model="alexnet"):
		self.model = torchvision.models.__dict__[model](pretrained=True)
		# Categories are standard types passed in by the user whose probabilities are always reported.
		self.categories = np.asarray(categories)
		# Labels are set of all labels.
		self.labelsPath = labelsPath
		self.loadLabels()

	# Load all 1000 labels from file to memory.
	def loadLabels(self):
		labelParser = LabelParser(self.labelsPath)
		labelParser.parseLabels()
		self.labels = np.asarray(labelParser.getLabels())
	
	def readImage(self, path):
		if(os.path.exists(path)):
			return Image.open(path)
		else:
			raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)

	def imshow(self, img):
		img = img / 2 + 0.5
		npimg = img.numpy()
		plt.imshow(np.transpose(npimg, (1, 2, 0)))

	def prepare(self, img):
		img = Classifier.preprocess(img)
		img.unsqueeze_(0)
		return img
	
	# Energy is model output just before going to softmax.
	def getEnergy(self, img):
		energy = self.model(img)
		return energy.detach().numpy()
	#return self.model(img)

	# Model output with softmax.	
	def getProbability(self, energy):
		return nn.functional.softmax(energy,dim=1)

	# Get probability directly from input image.
	def getPredictionProbability(self, img):
		return self.getProbability(self.getEnergy(img))

	# Get max probability.
	def getTopProbability(self, probability):
		return self.getTopKProbabilities(probability, 1)

	# Get index of element with max probability.
	def getTopProbabilityIndex(self, probability):
		return self.getTopKProbabilityIndices(probability, 1)

	# Get label of class with highest probability.
	def getTopClassLabel(self, probability):
		return self.getTopKClassLabels(probability, 1)

	# Get top 'k' probabilities.
	def getTopKProbabilities(self, probability, k):
		maxProbability, index = torch.topk(probability, k, dim=1)
		return maxProbability[0].detach().numpy()

	# Get indices of elements with the highest 'k' probabilities.
	def getTopKProbabilityIndices(self, probability, k):
		maxProbability, index = torch.topk(probability, k, dim=1)
		return index[0].numpy()

	# Get label of top 'k' classes with highest probability.
	def getTopKClassLabels(self, probability, k):
		indices = self.getTopKProbabilityIndices(probability, k)
		return self.labels[indices]

	# Get the probabilities of the categories specified by the user.
	def getCategoryProbabilities(self, probability):
		indices = np.where(np.isin(self.labels, self.categories))[0]
		probabilities = probability.detach().numpy()[0]
		return probabilities[indices]
