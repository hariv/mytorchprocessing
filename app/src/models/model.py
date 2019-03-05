import torch
#from .base_model import BaseModel
from torchvision import models as m

class Model():
    def name(self):
        return self.name

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def initialize(self, opt, data):
        self.opt = opt
        self.isTrain = opt.isTrain
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.loss_name = 'L1'
        self.model_name = opt.arch
        self.name = opt.name
        self.network = m.__dict__[self.model_name](**data)
        self.criterion = nn.L1Loss()
        self.optimizer = torch.optim.SGD(self.network.parameters(), opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)

    def set_input(self, input):
        self.input = input

    def set_target(self, target):
        self.target = torch.transpose(torch.from_numpy(np.vstack((target.numpy(), 1-target.numpy()))).float(), 0, 1)
        
    def forward(self):
        self.output = nn.functional.softmax(self.network(input))

    def backward(self):
        self.loss = self.criterion(self.output, self.target)
        self.loss.backward()
        
    def optimize_parameters(self):
        self.forward()
        self.backward()
        
    def setup(self, opt, parser=None):
        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.which_epoch)
        self.print_networks(opt.verbose)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):
            return
        self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i+1)

    def load_networks(self, which_epoch):
        load_filename = '%s_net_%s.pth' % (which_epoch, self.name)
        load_path = os.path.join(self.save_dir, load_filename)
        net = getattr(self, 'network')
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model from %s' % load_path)
        state_dict = torch.load(load_path,map_location=str(self.device))
        del state_dict.metadata

        for key in list(state_dict.keys()):
            self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
            net.load_state_dict(state_dict)

    def save_networks(self, which_epoch):
        save_filename = "%s_net_%s.pth" % (which_epoch, self.name)
        save_path = os.path.join(self.save_dir, save_filename)
        net = getattr(self, 'network')

        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(net.module.cpu().state_dict(), save_path)
            net.cuda(self.gpu_ids[0])
        else:
            torch.save(net.cpu().state_dict(), save_path)

    def print_networks(self, verbose):
        print('---------- Networks initialized -------------')
        net = getattr(self, 'network')
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        if verbose:
            print(net)
        print('[Network %s] Total number of parameters : %.3f M' %(self.model_name, num_params/1e6))
        print('-----------------------------------------------')
