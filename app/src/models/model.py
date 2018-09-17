import torch
from .base_model import BaseModel
from torchvision import models as m

class Model(BaseModel):
    def name(self):
        return self.name

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
        return parser

    def initialize(self, opt, data):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.lossName = 'L1'
        self.network = m.__dict__[opt.arch](**data)
        self.criterion = nn.L1Loss()
        self.optimizer = torch.optim.SGD(self.network.parameters(), opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
