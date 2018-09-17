import importlib
from models.base_model import BaseModel

def get_model_class():
    model_filename = "models.model"
    modellib = importlib.import_module(model_filename)

    model = None
    for name, cls in modellib.__dict__.items():
        if(issubclass(cls, BaseModel)):
            model = cls

    if model is None:
        print("Unable to find model")
        exit(0)

    return model
                
def get_option_setter(model_name):
    model_class = get_model_class()
    
def create_model(opt, num_classes):
    model = get_model_class()
    instance = model()
    data = {'num_classes': num_classes}
    instance.initialize(opt, data)
    print("model [%s] was created" % (instance.name()))
    return instance
