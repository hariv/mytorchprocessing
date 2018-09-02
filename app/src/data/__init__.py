import importlib
import torch.utils.data
import os
from data.base_data_loader import BaseDataLoader
from data.base_dataset import BaseDataset

def CreateDataLoader(opt):
    data_loader = CustomDatasetDataLoader()
    data_loader.initialize(opt)
    return data_loader

## Wrapper class of Dataset class that performs
## multi-threaded data loading
class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        
        self.traindir = os.path.join(opt.dataset_name, 'train')
        self.valdir = os.path.join(opt.dataset_name, 'val')
        self.testdir = os.path.join(opt.dataset_name, 'test')

        self.num_classes = sum(os.path.isdir(i) for i in os.listdir(opt.dataset_name))
        
        self.train_loader = torch.utils.data.DataLoader(
            self.traindir,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

        self.val_loader = torch.utils.data.DataLoader(
            self.valdir,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

        self.test_loader = torch.utils.data.DataLoader(
            self.testdir,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_train_data(self):
        return self.train_loader

    def load_val_data(self):
        return self.val_loader

    def load_test_data(self):
        return self.test_loader

    def get_num_classes(self):
        return self.num_classes

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batchSize >= self.opt.max_dataset_size:
                break
            yield data
