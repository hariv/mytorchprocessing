import unittest
import torch
import torch.nn as nn
from classifier import Classifier

class ClassifierTest(unittest.TestCase):
	
	def testGetProbability(self):
		classifier = Classifier()
		self.assertEqual(classifier.getProbability(1), nn.functional.softmax(1,dim=1))
if __name__ == '__main__':
    unittest.main()