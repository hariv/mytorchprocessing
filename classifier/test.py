import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.display_id = -1
    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)         
    
    if opt.eval:
        model.eval()
        
    for i, data in enumerate(dataset):
        model.set_input(data)
        model.test()
