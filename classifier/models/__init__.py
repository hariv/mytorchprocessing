import importlib
import torchvision.models
from models.base_model import BaseModel

def create_model(opt):
    data = {'num_classes': opt.num_classes}
    model = torchvision.models.__dict__[opt.model](**data)
    print("model [%s] was created" % opt.model)
    return instance
