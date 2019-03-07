import os
from options.test_options import TestOptions
from models import create_model

if __name__ == '__main__':
    opt = TestOptions().parse()
    model = create_model(opt)
