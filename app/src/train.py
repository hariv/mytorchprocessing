from options.train_options import TrainOptions

if __name__ == '__main__':
	opt = TrainOptions().parse()
	data_loader = CreateDataLoader(opt)