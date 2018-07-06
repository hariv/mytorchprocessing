from __future__ import print_function, division
import torch
import torch.nn as nn
import time
import os
import copy

class Classifier(nn.Module):
	
	def __init__(self, category, model="alexnet"):
		self.model = torchvision.models.__dict__[model](pretrained=True)
		self.category = category