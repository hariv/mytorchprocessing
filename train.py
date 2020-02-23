import argparse
import os
import shutil
import time
import json
import csv
import torch
import random
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

def train_single_batch(input, target, model, criterion, optimizer):
    input = input.cuda()
    target = target.cuda()

    # Sets these as parts of what needs to be differentiated for backprop
    # Target is the class (DataLoader that we use makes the target represent
    # a unique number for each input representing a class)
    input_var = torch.autograd.Variable(input)
    target_var = torch.autograd.Variable(target)

    # Model output. This is a vector of size equal to number of classes for
    # each input.
    output = model(input_var)

    # loss is the difference between predicted and target classes
    # The implementation of CrossEntropyLoss will take care of figuring
    # out the predicted class from the output vector (the index of the largest value).
    loss = criterion(output, target_var)

    # Clean the optimizer so it doesn't hold onto any values from
    # previous runs of backprop.
    optimizer.zero_grad()
    # Do backpropagation of loss to compute changes to be made to
    # model weights.
    loss.backward()
    # Update model weights with the backpropagation computation.
    optimizer.step()

    # What comes below is not part of the training.
    # It is for logging purposes for us to keep track of
    # improvements in accuracy with training.
    
    _, prediction = torch.max(output, 1)
    
    accuracy = (target == prediction.squeeze()).float().mean()
    return accuracy, loss

def validate(args, model, criterion, val_loader):
    val_acc = []
    val_loss = []

    # Some model functions are implented differently when
    # model is in training mode and validation mode. This sets
    # that we're currently using the model for validation.
    model.eval()
    
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

        output = model(input_var)
        loss = criterion(output, target_var)
        
        _, prediction = torch.max(output, 1)
        
        accuracy = (target == prediction.squeeze()).float().mean()
        val_acc.append(accuracy.item())
        val_loss.append(loss.item())
        
    return sum(val_acc) / float(len(val_acc)), sum(val_loss) / float(len(val_loss))

def save_checkpoint(ckpt_dir, experiment, network_name, model, optimizer, accuracy, epoch, iteration):
    file_name = ckpt_dir + '/' + experiment + '-' +  network_name + '-Epoch-' + str(epoch) + '-Iteration-' + str(iteration) + '.pth'
    torch.save(model.state_dict(), file_name)

def write_log(log_dir, experiment, network_name, epoch, iteration, training_loss, training_accuracy, val_loss, val_accuracy):
    file_name = log_dir + '/' + experiment + '' +  network_name + '-log-file.csv'
    with open(file_name, 'a') as log_file:
        csv_writer = csv.writer(log_file)
        csv_writer.writerow([epoch, iteration, training_loss, training_accuracy, val_loss, val_accuracy])

def train(args, model, criterion, optimizer, train_loader, val_loader, iteration):
    train_acc = []
    train_loss = []

    # One epoch is one full look through the entire training set.
    for epoch in range(args.start_epoch, args.epochs):
        model.train()

        # each input, target here is going to refer to one batch based
        # on the size specified when invoking the script. Each backprop
        # calculation is done on the entire batch.
        for i, (input, target) in enumerate(train_loader):
            print("Training Epoch: " + str(epoch) + " iteration: " + str(iteration))
            batch_train_acc, batch_train_loss = train_single_batch(input, target, model, criterion, optimizer)
            train_acc.append(batch_train_acc.item())
            train_loss.append(batch_train_loss.item())
            
            iteration += 1

            # Validate every few iterations
            if iteration % args.niter_eval == 0:
                print("Validating")
                val_acc, val_loss = validate(args, model, criterion, val_loader)
                write_log(args.log_dir, args.experiment, args.network_name, epoch, iteration, sum(train_loss) / float(len(train_loss)), sum(train_acc) / float(len(train_acc)), val_loss, val_acc)

            # Checkpoint every few iterations
            if iteration % args.niter_save == 0: 
                save_checkpoint(args.ckpt_dir, args.experiment, args.network_name, model, optimizer, val_acc, epoch, iteration)
    print('Finished training')

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch classifier')
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--network', default='alexnet', type=str, dest='network_name', help='which model to use')
    parser.add_argument('--experiment', default='sample_classification', type=str, dest='experiment', help='name of the experiment')
    parser.add_argument('--workers', default=0, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('--classes', default='class_1,class_2', type='str', dest='classes', help='name of classes')
    parser.add_argument('--epochs', default=1000, type=int, metavar='N', help='number of training epochs to run for')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='epoch number from which to restart training')
    parser.add_argument('-b', '--batch_size', default=256, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='load pretrained model')
    parser.add_argument('--lr', '--learning_rate', default=0.01, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--beta', default=0.5, type=float, help='momentum term for adam')
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay')
    parser.add_argument('--dropout_prob', default=0.0, type=float, dest='dropout_prob', help='dropout probability of zeroing out neurons')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    parser.add_argument('-lw', default=224, type=int, dest='width', help='image width fed to model')
    parser.add_argument('-lh', default=400, type=int, dest='height', help='image height fed to model')
    parser.add_argument('--ckpt_dir', default='./checkpoints', type=str, dest='ckpt_dir', help='location of saved checkpoints')
    parser.add_argument('--log_dir', default='./logs', type=str, dest='log_dir', help='location to save logs')
    parser.add_argument('--niter_eval', default=20, type=int, dest='niter_eval', help='Number of iterations to run before evaluating on validation set')
    parser.add_argument('--niter_save', default=20, type=int, dest='niter_save', help='Number of iterations to run before checking to save model')
    parser.add_argument('--nval_images', default=20, type=int, dest='nval_images', help='Number of images to validate model on')
    return parser.parse_args()

def dispatch():
    args = parse_args()

    num_classes = len(args.classes.split('.'))

    # Works with any standard network supported by Pytorch. Write custom model class to use custom model.
    # Extend the final layer of the pretrained imagenet model having 1000 classes to have as many classes as needed for the problem
    model = nn.Sequential(models.__dict__[args.arch](pretrained=args.pretrained), nn.Linear(1000, num_classes)).cuda()

    # Loss definition
    criterion = nn.CrossEntropyLoss().cuda()
    # Optimizer definition that takes care of updating gradient descent weights
    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(args.beta, 0.999))
    iteration = 0

    # Option to load a previously trained model to continue training or evaluation.
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            model.load_state_dict(args.resume)
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        print("=> Creating new " + args.network_name + " model")

    # This enables an inbuilt tuner to pick implementation of algorithms based on hardware
    # Ref: https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
    cudnn.benchmark = True
    
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    # Normalize inputs based on imagenet values.
    # Imagenet covers millions of natural images.
    # It's usually a good idea to use these values
    # if you're also going to be working on natural images.
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.226, 0.226, 0.226])
    
    print("Loading training data....")

    # The dataloader is responsible for loading the data per batch.
    # Everything else is a form of on the fly data augmentation
    # to get more samples from existing samples.
    # Augmentation randomly picks one or more of these and applies
    # them to the loaded images from training directory.
    # Also ensures that images are loaded in a size compatible with
    # the model based on the input paramters.
    
    random_scale = random.randint(1, 5)*30
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Resize((random_scale*16, random_scale*9)),
            transforms.Resize((args.height, args.width)),
            transforms.ToTensor(), normalize])),
        batch_size = args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=True)
    print("Loaded")
    print("Loading validation data...")

    # Same thing, but load the validation data. We don't care about
    # augmentation here.
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
                    transforms.Resize((args.height, args.width)),
                    transforms.ToTensor(), normalize])),
        batch_size=args.nval_images, shuffle=True, num_workers=args.workers,
        pin_memory=True)
    print("Loaded")

    # Onlt run validation
    if args.evaluate:
        validate(args, model, criterion, val_loader)
        return

    train(args, model, criterion, optimizer, train_loader, val_loader, iteration)

if __name__ == '__main__':
    dispatch()
