import os
from options.test_options import TestOptions
import torch

from models import create_model
from data import create_dataset

if __name__ == '__main__':
    opt = TestOptions().parse()
    
    
    dataset = create_dataset(opt)
    model = create_model(opt)
    
    if opt.name != "imagenet":
        model.setup(opt)
    
    model.eval()
        
    for i, (input, target) in enumerate(dataset):
        model.set_input(input, target)
        model.test()
        prediction = model.get_prediction()
        loss = model.get_loss()
        
        print("Prediction", prediction)
        
        
    
    
