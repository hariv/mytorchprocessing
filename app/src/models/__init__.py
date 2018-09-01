import torchvision.models as models

from models.base_model import BaseModel

class Model(BaseModel):
    def name(self):
        return self.name

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.name = opt.arch
        self.lossName = 'L1'

        self.network = models.__dict__[opt.arch]
        


def create_model(opt):
    model = Model(opt)
    #model = models.__dict__[opt.arch]