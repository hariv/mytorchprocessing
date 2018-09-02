import torchvision.models as models

from models.base_model import BaseModel

class Model(BaseModel):
    def name(self):
        return self.name

    def initialize(self, opt, data):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.name = opt.arch
        self.lossName = 'L1'
        self.network = models.__dict__[opt.arch](**data)

def create_model(opt, num_classes):
    data = {'num_classes': num_classes}
    model = Model(opt, data)
    #model = models.__dict__[opt.arch]