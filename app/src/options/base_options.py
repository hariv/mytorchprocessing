import argparse
import os
from util import util
import torch
import models
import data


class BaseOptions():
	def __init__(self):
		self.initialized = False

	def initialize(self, parser):
		parser.add_argument('--dataroot', required=True, help='path to images')
        parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        parser.add_argument('--loadWidth', type=int, default=1024, help='scale images to this size')
        parser.add_argument('--fineSize', type=int, default=1024, help='then crop to this size')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--name', type=str, default='general', help='which trained model to run')
        