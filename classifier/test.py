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
    
    labels = []

    with open(opt.label_dir+"/"+opt.name+".json") as label_file:
        content = label_file.readlines()
        content = [x.strip() for x in content]
        for line in content:
            labels.append(line)
    
    for i, (input, target) in enumerate(dataset):
        model.set_input(input, target)
        model.test()
        prediction = torch.argmax(model.get_prediction(), dim=1)
        prediction = prediction.tolist()

        print(labels[prediction[0]])        
        
        
    
    
