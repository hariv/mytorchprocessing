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
        parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--name', type=str, default='imagenet', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        
        parser.add_argument('--model', type=str, default='alexnet', help='chooses which model to use. [alexnet | resnet | vgg]')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')

        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser,add_argument('--num_classes', type=int, default=2, help='# of output classes')

        self.initialize = True
        return parser
    
    def gather_options(self):
        
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
            
        opt, _ = parser.parse_known_args()
        #model_name = opt.model
        #model_option_setter = models.get_option_setter(model_name)
        #parser = model_option_setter(parser, self.isTrain)
        #opt, _ = parser.parse_known_args()
        
        #dataset_name = opt.dataset_mode
        #dataset_option_setter = data.get_option_setter(dataset_name)
        #parser = dataset_option_setter(parser, self.isTrain)
        
        
        self.parser = parser
        return parser.parse_args()

    
    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
        
        
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
            
    def parse(self):
        opt = self.gather_options()
        opt.isTrain = self.isTrain
        
        self.print_options(opt)
        
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
        
