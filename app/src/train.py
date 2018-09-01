from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model


if __name__ == '__main__':
	opt = TrainOptions().parse()
	data_loader = CreateDataLoader(opt)
	dataset = data_loader.load_data()

	dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    