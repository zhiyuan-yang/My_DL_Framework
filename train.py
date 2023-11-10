import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
from config import args
import datetime
from utils.logger import Logger
from utils.set_random_seed import set_random_seed
import logging


# recording setting
logger = Logger(log_file_name=args.log_file_name, logger_name=args.logger_name, log_level=logging.DEBUG).get_log()
log_dir = './TensorboardLog/' + datetime.datetime.now().strftime('%Y-%m-%d') + '/'
writer = SummaryWriter(log_dir=log_dir)

# training setting
batch_size = args.batch_size
num_workers = args.num_workers
learning_rate = args.lr_rate
num_epochs = args.num_epochs
beta1 = args.beta1
beta2 = args.beta2
eps = args.eps
decay =args.decay
gamma = args.gamma
device = torch.device('cpu' if args.cpu else 'cuda')


# validation setting
val_freq = args.val_freq


def batch_to_device(batch):
    for key in batch.keys():
        batch[key] = batch[key].to(device)
    return batch


def main():
    #=====LOAD DATASET HERE=====
    trainDataset = np.array() #change it with dataset class in ./datasets
    trainDataLoader = DataLoader(dataset=trainDataset,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 shuffle=True)
    valDataset = np.array()
    valDataLoader = DataLoader(dataset=valDataset,
                                 batch_size=batch_idx,
                                 num_workers=1,
                                 shuffle=False)
    
    
    #=====LOAD MODEL HERE====== 
    model = np.array().to(device)  #change it with model in ./models
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr = learning_rate,
                                 betas=(beta1, beta2), 
                                 eps=eps)
    schduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=decay, gamma=gamma)
    start =time.time()


    logger.info(f'Start Training\n')
    
    
    #training stage
    for i in range(num_epochs):
        model.train()
        curr_epoch_train_loss = 0
        for batch_idx, batch in enumerate(trainDataLoader):
            optimizer.zero_grad()
            batch = batch_to_device(batch)
            outcome = model(batch)  #change it with true model
            
            #====Calculate LOSS HERE====
            curr_batch_train_loss = np.array()  #change it with loss in ./utils/loss
            curr_batch_train_loss.backward()
            optimizer.step()
            curr_epoch_train_loss += curr_batch_train_loss.item() * trainDataLoader.batch_size
            
            if (batch_idx + 1)%1000 == 0:
                logger.info(f'Current Epoch: {i}, Batch: {batch_idx}, Batch_loss: {curr_epoch_train_loss/(batch_idx + 1)}\n')
        
        schduler.step()
        curr_epoch_train_loss = curr_epoch_train_loss/len(trainDataLoader.dataset)


        if (i+1) % val_freq == 0:
            model.eval()
            curr_epoch_val_loss = 0
            curr_val_SSIM = 0
            curr_val_PSNR = 0
            with torch.no_grad():
                for val_batch in valDataLoader:
                    val_batch = batch_to_device(val_batch)
                    valoutcome = model(val_batch)
                    curr_batch_val_loss = np.array()  #change it with true loss
                    curr_epoch_val_loss += curr_batch_val_loss.item() * valDataLoader.batch_size
                    

            curr_epoch_val_loss =curr_epoch_train_loss/len(valDataLoader.dataset)
            
            if curr_epoch_val_loss < min_val_loss:
                min_val_loss = curr_epoch_val_loss
                
            torch.save(model.state_dict(), './weight/epoch_'+ str(i+1)+'.pt')
            
            writer.add_scalar("validation loss", curr_epoch_val_loss, (i+1)/val_freq)
            logger.info(f'Current Epoch: {i+1} Validation Loss: {curr_epoch_val_loss}\n')
        
        logger.info(f'Training Epoch:{i+1} Epoch Loss:{curr_epoch_train_loss}\n')
        writer.add_scalar("train loss", curr_epoch_train_loss, i+1)    


    end = time.time()
    logger.info(f'Running Time: {(end-start)/3600}h')


if __name__ == '__main__':
    set_random_seed(0)
    main()