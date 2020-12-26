import argparse
import os
import shutil
import time
import json
import csv
import cv2
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

def predict(model, test_tensor, classes):
    test_tensor = test_tensor.permute(0, 3, 1, 2).type(torch.FloatTensor).cuda()
        
    with torch.no_grad():
        input_var = torch.autograd.Variable(test_tensor)
        
    output = model(input_var)
    _, prediction = torch.max(output, 1)
    return classes[prediction.data.tolist()[0]]

def load_test_tensor(image_path, lw, lh):
    MEANS = [123.68, 116.28, 103.53]
    STDS = [58.40, 57.12, 57.38]
    
    test_image = cv2.imread(image_path)
    test_image = cv2.resize(test_image, (lw, lh),
                            interpolation=cv2.INTER_LANCZOS4)
    for i in range(3):
        test_image[:, :, i] = (test_image[:, :, i] - MEANS[i]) / STDS[i]

    test_image = np.expand_dims(test_image, axis=0)

    return torch.from_numpy(test_image)
    
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch classifier')
    parser.add_argument('image', metavar='DIR', help='path to test image')
    parser.add_argument('--network', default='alexnet', type=str, dest='network_name', help='which model to use')
    parser.add_argument('--experiment', default='sample_classification', type=str, dest='experiment', help='name of the experiment')
    parser.add_argument('--workers', default=0, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('--use_gpu', dest='use_gpu', action='store_true', help='flag to run on GPU')
    parser.add_argument('--classes', default='class_1,class_2', type=str, dest='classes', help='name of classes')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
    parser.add_argument('-lw', default=224, type=int, dest='width', help='image width fed to model')
    parser.add_argument('-lh', default=400, type=int, dest='height', help='image height fed to model')
    return parser.parse_args()

def dispatch():
    args = parse_args()

    num_classes = len(args.classes.split(','))

    model = nn.Sequential(models.__dict__[args.network_name](),
                          nn.Linear(1000, num_classes)).cuda()
    
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        if args.use_gpu:
            model.load_state_dict(torch.load(args.resume))
        else:
            model.load_state_dict(torch.load(args.resume,
                                             map_location=torch.device("cpu")))
        print("=> loaded checkpoint '{}'".format(args.resume))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        exit()

    cudnn.benchmark = True
    
    if not os.path.exists(args.image):
        print("=> {} does not exist".format(args.image))
        
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    print("Loading test image....")
    test_tensor = load_test_tensor(args.image, args.width, args.height)
    print("Loaded")

    prediction = predict(model, test_tensor, args.classes.split(','))
    print(prediction)
    
if __name__ == '__main__':
    dispatch()
