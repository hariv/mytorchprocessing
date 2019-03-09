import os
from options.test_options import TestOptions

from models import create_model
from data import create_dataset

if __name__ == '__main__':
    opt = TestOptions().parse()
    
    #print(opt)
    '''dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    
    if opt.eval:
        model.eval()
        
    for i, (input, target) in enumerate(dataset):
        if i >= opt.num_test:
            break
        model.set_input(input, target)
        model.test()
        prediction = model.get_prediction()
        loss = model.get_loss()
        
        print("Prediction", prediction)'''
        
        
    
    
