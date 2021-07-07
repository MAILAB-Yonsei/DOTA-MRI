import os
from math import sqrt

import torch
from torch import unsqueeze
from torch.utils.data import DataLoader
from torch.autograd import Variable

from layer_utils import weights_init_normal
from importlib import import_module

from datasets import ImageDataset

from config import GenConfig
from utils import GenAddPathStr, GenAugSeq, save_images

from losses import loss_dict
from layer_utils import AbsLayer

opt = GenConfig()

add_path_str = GenAddPathStr(opt)
os.makedirs('SavedModels_%s/%s' % (opt.model_name, add_path_str), exist_ok=True)
os.makedirs('Validation_%s/%s' % (opt.model_name, add_path_str), exist_ok=True)

GeneratorNet = getattr(import_module('models'), opt.model_name)

# Initialize generator and criterion
generator = GeneratorNet(opt)

# criterion = torch.nn.MSELoss()
criterion_img = loss_dict['img']()
if 'DOTA' in opt.model_name and opt.dota_dc != 0:
    criterion_dc = loss_dict['dc']()

cuda = True if torch.cuda.is_available() else False
if cuda:
    generator = generator.cuda()
    criterion_img = criterion_img.cuda()
    if 'DOTA' in opt.model_name and  opt.dota_dc != 0:
        criterion_dc = criterion_dc.cuda()

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load('SavedModels_%s/%s/generator_%d.pth' % (opt.model_name, add_path_str, opt.epoch)))
else:
    # Initialize weights
    generator.apply(weights_init_normal)

# Optimizers
optimizer = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=10**-7)

# Tensor allocation
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

if opt.abs_layer == True or opt.coil_label != 'mc':
    channel_label = opt.channels
else:
    channel_label = opt.channels * opt.coil_num
    
if opt.abs_layer == True or opt.coil_input != 'mc':
    channel_input = opt.channels
else:
    channel_input = opt.channels * opt.coil_num
    
input_k_us_train   = Tensor(opt.batch_size, channel_input, opt.im_height, opt.im_width)
input_img_fs_train = Tensor(opt.batch_size, channel_label, opt.im_height, opt.im_width)
if opt.brain_masking == True:
    input_bm_train = Tensor(opt.batch_size, 1, opt.im_height, opt.im_width)

input_k_us_valid   = Tensor(1, channel_input, opt.im_height, opt.im_width)
input_img_fs_valid = Tensor(1, channel_label, opt.im_height, opt.im_width)
input_mask_valid   = Tensor(1, 1, opt.im_height, opt.im_width)

# Data augmentation
if opt.data_augment:
    aug_seq = GenAugSeq()
else:
    aug_seq = None

# Data loader
dataloader_train = DataLoader(ImageDataset(opt, aug_seq=aug_seq), batch_size=opt.batch_size, shuffle=True)
dataloader_valid = DataLoader(ImageDataset(opt, is_valid=True, test_num=1), batch_size=opt.batch_size, shuffle=False)

# --------------------------
#  Training with validation
# --------------------------

if __name__ == "__main__":
    
    for i, valid_data in enumerate(dataloader_valid):
        valid_kspaces_us = Variable(input_k_us_valid.copy_(valid_data['kspace_us']))
        valid_imgs_fs    = Variable(input_img_fs_valid.copy_(valid_data['img_fs']))
        valid_mask       = Variable(input_mask_valid.copy_(valid_data['mask']))
        train_mask       = valid_mask.repeat((opt.batch_size,1,1,1))
        
        if opt.abs_layer:
            valid_imgs_fs = AbsLayer(valid_imgs_fs)
                
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, train_data in enumerate(dataloader_train):
            if train_data['kspace_us'].size(0) != opt.batch_size:
                continue
            # Training
            train_kspaces_us  = Variable(input_k_us_train.copy_(train_data['kspace_us']))
            train_imgs_fs     = Variable(input_img_fs_train.copy_(train_data['img_fs']))
            
            if opt.brain_masking == True:
                bm = Variable(input_bm_train.copy_(train_data['bm']))
                if opt.abs_layer:
                    bm = bm[:,0,:,:]
                else:
                    bm = bm.repeat((1,channel_label,1,1))
            else:
                bm = None
                
            if opt.abs_layer:
                train_imgs_fs = AbsLayer(train_imgs_fs)
                
            optimizer.zero_grad()
            
            train_imgs_rec = generator(train_kspaces_us, train_mask)
            
            if bm != None:
                loss_train_img = criterion_img(train_imgs_rec * bm, train_imgs_fs * bm)
            else:
                loss_train_img = criterion_img(train_imgs_rec, train_imgs_fs)
            
            if 'DOTA' in opt.model_name and opt.dota_dc != 0:
                loss_train_dc = criterion_dc(train_imgs_rec, train_kspaces_us, train_mask)
                loss_train = loss_train_img + opt.dota_dc * loss_train_dc
            else:
                loss_train = loss_train_img
            
            loss_train.backward()
            optimizer.step()
            
            # Validation
            valid_imgs_rec = generator(valid_kspaces_us, valid_mask)            
                
            loss_valid_img = criterion_img(valid_imgs_rec, valid_imgs_fs)
            if 'DOTA' in opt.model_name and opt.dota_dc != 0:
                loss_valid_dc = criterion_dc(valid_imgs_rec, valid_kspaces_us, valid_mask)
                loss_valid = loss_valid_img + opt.dota_dc * loss_valid_dc
            else:
                loss_valid = loss_valid_img
            
            # Print status
            print("[Epoch %d/%d] [Batch %d/%d] [Valid loss: %.4f]" %
                  (epoch, opt.n_epochs, i, len(dataloader_train), sqrt(loss_valid.item())))
            
            # Save validation images
            batches_done = epoch * len(dataloader_train) + i
            if batches_done % opt.sample_interval == 0:
                if opt.abs_layer:
                    valid_imgs_rec_r = unsqueeze(valid_imgs_rec, 1)
                    valid_imgs_fs_r  = unsqueeze(valid_imgs_fs, 1)
                else:
                    valid_imgs_rec_r = valid_imgs_rec
                    valid_imgs_fs_r  = valid_imgs_fs
                    
                val_data = [valid_data['img_us'], valid_imgs_rec_r, valid_imgs_fs_r]
                save_images(val_data, loss_valid, epoch, batches_done, opt.model_name, add_path_str, Tensor)
                
           
        # Save model checkpoints
        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
                torch.save(generator.state_dict(), 'SavedModels_%s/%s/generator_%d.pth' % (opt.model_name, add_path_str, epoch + 1))