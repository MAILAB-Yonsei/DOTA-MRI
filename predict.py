import argparse
import os
import numpy as np
from math import sqrt

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from importlib import import_module
from datasets import ImageDataset

import scipy.io as sio

parser = argparse.ArgumentParser()
parser.add_argument('--add_path_str', type=str, default='', help='string to be added to directory name')
parser.add_argument('--model_name', type=str, default='Cascade_I25_4', help='model name')
parser.add_argument('--dataset_name', type=str, default='HCP_MGH_T1w', help='name of the dataset')
parser.add_argument('--acc_rate', type=int, default=5, help='acceleration ratio')
parser.add_argument('--acs_num', type=int, default=16, help='the number of acs lines')

parser.add_argument('--epoch', type=int, default=1, help='epoch to start testing from')
parser.add_argument('--n_epochs', type=int, default=300, help='number of epochs of testing')

parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.00005, help='adam: learning rate') # start: 0.00005
parser.add_argument('--b1', type=float, default=0.9, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')

parser.add_argument('--im_height', type=int, default=256, help='size of image height')
parser.add_argument('--im_width', type=int, default=256, help='size of image width')
parser.add_argument('--channels', type=int, default=2, help='number of image channels')

parser.add_argument('--n_cpu', type=int, default=1, help='number of cpu threads to use during batch generation')

parser.add_argument('--checkpoint_interval', type=int, default=1, help='interval between model checkpoints')
parser.add_argument('--mask_index', type=int, default=0, help='validation/test mask index')

parser.add_argument('--data_augment', type=bool, default=True, help='16-fold data augmentation')
parser.add_argument('--random_sampling', type=bool, default=False, help='Generate random sampling patterns during training')

parser.add_argument('--save_results', type=bool, default=True, help='save results to mat file')
opt = parser.parse_args()
print(opt)

GeneratorNet = getattr(import_module('models'), opt.model_name)

add_path_str = opt.dataset_name + '_acc%d_acs%d_mask%d' % (opt.acc_rate, opt.acs_num, opt.mask_index)
if opt.data_augment:
    add_path_str = add_path_str + '_DataAug'
if opt.random_sampling:
    add_path_str = add_path_str + '_RandomSampling'
add_path_str = add_path_str + opt.add_path_str

os.makedirs('Prediction_%s/%s' % (opt.model_name, add_path_str), exist_ok=True)

cuda = True if torch.cuda.is_available() else False

# Initialize generator and discriminator
generator = GeneratorNet(opt.channels, opt.channels)

# Losses
criterion = torch.nn.MSELoss()

if cuda:
    generator = generator.cuda()
    criterion = criterion.cuda()

# Optimizers
optimizer = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=10**-7)

# Tensor allocation
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

input_k_us  = Tensor(opt.batch_size, opt.channels, opt.im_height, opt.im_width)
input_im_fs = Tensor(opt.batch_size, opt.channels, opt.im_height, opt.im_width)
input_mask  = Tensor(opt.batch_size, opt.channels, opt.im_height, opt.im_width)

# Data loader
dataloader = DataLoader(ImageDataset(opt.dataset_name,
                                     acc_rate=opt.acc_rate,
                                     acs_num=opt.acs_num,
                                     is_testing=True,
                                     test_num=-1),
                                     batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

if __name__ == "__main__":
    total_RMSE = np.zeros(opt.n_epochs)
    
    for epoch in range(opt.epoch, opt.n_epochs):
        generator.load_state_dict(torch.load('SavedModels_%s/%s/generator_%d.pth' %
                                             (opt.model_name, add_path_str, epoch)))
        for i, test_data in enumerate(dataloader):
            test_kspaces_us = Variable(input_k_us.copy_(test_data['kspace_us']))
            test_imgs_fs    = Variable(input_im_fs.copy_(test_data['img_fs']))
            test_mask       = Variable(input_mask.copy_(test_data['mask_rev']))
            
            test_imgs_rec = generator(test_kspaces_us, test_mask)
            loss = criterion(test_imgs_rec, test_imgs_fs)
            
            print("[epoch %d/%d] [dataset %d/%d] [test RMSE: %.4f]" %
                  (epoch, opt.n_epochs, i, len(dataloader), sqrt(loss.item())))
    
            test_imgs_rec_np = Tensor.cpu(test_imgs_rec.detach()).numpy()
            test_imgs_fs_np  = Tensor.cpu(test_imgs_fs.detach()).numpy()
            img_recon = test_imgs_rec_np[:,0,:,:] + 1j * test_imgs_rec_np[:,1,:,:]
            img_fs    = test_imgs_fs_np[:,0,:,:]  + 1j * test_imgs_fs_np[:,1,:,:]
            
            if i is 0:
                imgs_recon = img_recon
                imgs_fs = img_fs
            else:
                imgs_recon = np.concatenate((imgs_recon, img_recon), axis=0)
                imgs_fs = np.concatenate((imgs_fs, img_fs), axis=0)
        
        imgs = {'recon':imgs_recon, 'fs':imgs_fs}
        RMSE = sqrt((np.abs(imgs_fs - imgs_recon)**2).mean())
        total_RMSE[epoch - 1] = RMSE
        if opt.save_results:
            sio.savemat('Prediction_%s/%s/epoch%d_RMSE%.4f.mat' % (opt.model_name, add_path_str, epoch, RMSE),imgs)
            
    total_RMSE_dic = {'RMSE': total_RMSE}
    sio.savemat('Prediction_%s/%s/log_RMSE.mat' % (opt.model_name, add_path_str),total_RMSE_dic)