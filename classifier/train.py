import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model

if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    
    print('The number of training images = %d' % dataset_size)
    
    model = create_model(opt)
    model.setup(opt)
    
    total_iters = 0
    
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        
        epoch_iter = 0
        
        for i, (input, target) in enumerate(dataset):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            
            model.set_input(input, target)
            model.optimize_parameters()
            
            if total_iters % opt.print_freq == 0:
                loss = model.get_loss()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
            
            if total_iters % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)
                
            iter_data_time = time.time()
        
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
        
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
            
