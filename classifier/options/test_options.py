from .base_options import BaseOptions

class TestOptions(BaseOptions):
    
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--label_dir', type=str, default='./labels', help='Directory containing class labels as json file')
        self.isTrain = False
        
        return parser
        
