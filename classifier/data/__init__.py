import torch.utils.data
from data.base_dataset import BaseDataset

def create_dataset(opt):
    dataloader = BaseDataset(opt)
    dataset = torch.utils.data.DataLoader(dataloader.dataset, batch_size=opt.batch_size, shuffle=not opt.serial_batches, num_workers=int(opt.num_threads))
    return dataset
