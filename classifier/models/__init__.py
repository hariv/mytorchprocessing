import torchvision.models
from models.base_model import BaseModel

def create_model(opt):
    model = BaseModel(opt)
    print("model [%s] was created" % opt.model)
    return model
