from .base_options import BaseOptions

class TestOptions(BaseOptions):
    
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.set_defaults(model='test')
        self.isTrain = False
        
        return parser
        
