import os
import torch
from collections import OrderedDict
from . import networks
import torchvision.models

class BaseModel():
    
    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        
        self.criterion = torch.nn.CrossEntropyLoss().cuda()
        self.net_name = opt.model
        data = {'num_classes': opt.num_classes}
        self.net = torchvision.models.__dict__[self.net_name](pretrained=opt.pretrained, **data)
        self.net = self.net.to(self.device)
        if(self.isTrain):
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.metric = None
        self.name = opt.name
        self.save_dir = os.path.join(opt.checkpoints_dir, self.name)
        
    def set_input(self, input, target):
        self.image = input.to(self.device)
        self.target = target.to(self.device)
        
    def get_prediction(self):
        return self.probabilities
    
    def get_loss(self):
        return self.criterion(self.output, self.target)
    
    def forward(self):
        self.output = self.net(self.image)
        self.probabilities = torch.nn.functional.softmax(self.output, dim=1)
    
    def backward(self):
        self.loss = self.criterion(self.output, self.target)
        self.loss.backward()
            
    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad(self.net, True)
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()
    
    def setup(self, opt):
        if self.isTrain:
            self.scheduler = networks.get_scheduler(self.optimizer, opt)

        if not self.isTrain or opt.continue_train:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)

        self.print_networks(opt.verbose)
        
    def eval(self):
        self.net.eval()

    def test(self):
        with torch.no_grad():
            self.forward()
                
    def update_learning_rate(self):
        self.scheduler.step(self.metric)
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def save_networks(self, epoch):
        save_filename = '%s_net_%s_%s.pth' % (epoch, self.net_name, self.name)
        save_path = os.path.join(self.save_dir, save_filename)
        
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(self.net.cpu().state_dict(), save_path)
            self.net.cuda(self.gpu_ids[0])
        else:
            torch.save(self.net.cpu().state_dict(), save_path)
            
    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):
            if module.__class__.__name__.startswith('InstanceNorm') and (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)
            
    def load_networks(self, epoch):
        load_filename = '%s_net_%s_%s.pth' % (epoch, self.net_name, self.name)
        load_path = os.path.join(self.save_dir, load_filename)
        
        if isinstance(self.net, torch.nn.DataParallel):
            self.net = self.net.module
            
        print('loading the model from %s' % load_path)
        state_dict = torch.load(load_path, map_location=str(self.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        
        for key in list(state_dict.keys()):
            self.__patch_instance_norm_state_dict(state_dict, self.net, key.split('.'))
        self.net.load_state_dict(state_dict)
        
        
    def print_networks(self, verbose):
        print('---------- Networks initialized -------------')
        num_params = 0
        for param in self.net.parameters():
            num_params += param.numel()
        if verbose:
            print(self.net)
        print('[Network %s] Total number of parameters : %.3f M' % (self.net_name, num_params / 1e6))
        print('-----------------------------------------------')
    
    def set_requires_grad(self, nets, requires_grad=False):
        for param in self.net.parameters():
            param.requires_grad = requires_grad
