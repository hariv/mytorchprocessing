from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        # To avoid cropping, the loadSize should be the same as fineSize
        parser.set_defaults(loadSize=parser.get_default('fineSize'))
        
        self.isTrain = False
        return parser
