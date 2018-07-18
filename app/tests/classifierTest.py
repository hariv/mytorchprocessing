import unittest
import torch
import torch.nn as nn
from unittest.mock import patch
#import mock
from ..src.classifier import Classifier


class ClassifierTest(unittest.TestCase):	
	@patch('..src.classifier.loadLabels')
	def testGetProbability(self):
		classifier = Classifier(categories=['street sign,', 'suit,', 'magpie,', 'jay,', 'sunglasses,'], labelsPath='../labels.txt')
		self.assertEqual(classifier.getProbability(1), nn.functional.softmax(1,dim=1))
		
if __name__ == '__main__':
	unittest.main()
