import os
from options.test_options import TestOptions

from models import create_model
from data import create_dataset

if __name__ == '__main__':
    opt = TestOptions().parse()
    model = create_model(opt)
    dataset = create_dataset(opt)
