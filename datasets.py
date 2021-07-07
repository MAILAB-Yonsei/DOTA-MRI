from glob import glob
import numpy as np
import scipy.io as sio

from torch.utils.data import Dataset
from torch import Tensor

from utils import CartesianMask, augment_data, gen_data_from_image

class ImageDataset(Dataset):
    def __init__(self, opt, is_valid=False, is_testing=False, test_num=1, aug_seq=None):
        coil_input_name = opt.coil_input
        coil_label_name = opt.coil_label
        if coil_input_name == 'sc':
            coil_input_name += str(opt.coil_ind)
        if coil_label_name == 'sc':
            coil_label_name += str(opt.coil_ind)
        
        if is_testing:
            self.files_fs_i = sorted(glob('Data/%s/%s/Test/*.*' % (opt.dataset_name, coil_input_name)))
            self.files_fs_l = sorted(glob('Data/%s/%s/Test/*.*' % (opt.dataset_name, coil_label_name)))
            if test_num != -1:
                self.files_fs_i = self.files_fs_i[:test_num]
                self.files_fs_l = self.files_fs_l[:test_num]
        elif is_valid:
            self.files_fs_i = sorted(glob('Data/%s/%s/Valid/*.*' % (opt.dataset_name, coil_input_name)))
            self.files_fs_l = sorted(glob('Data/%s/%s/Valid/*.*' % (opt.dataset_name, coil_label_name)))
            if test_num != -1:
                self.files_fs_i = self.files_fs_i[:test_num]
                self.files_fs_l = self.files_fs_l[:test_num]
        else:
            self.files_fs_i = sorted(glob('Data/%s/%s/Train/*.*' % (opt.dataset_name, coil_input_name)))
            self.files_fs_l = sorted(glob('Data/%s/%s/Train/*.*' % (opt.dataset_name, coil_label_name)))
            self.files_bm   = sorted(glob('Data/%s/bm/*.*' % (opt.dataset_name)))
            
        self.brain_masking = False
        if opt.brain_masking == True and is_valid == False and is_testing == False:
            self.brain_masking = True
        self.opt = opt
        self.is_testing = is_testing
        
        self.aug_seq = aug_seq
        
    def __getitem__(self, index):
        if 'Challenge' in self.opt.dataset_name:
            input_name = self.opt.coil_input
            label_name = self.opt.coil_label
        else:
            input_name = label_name = 'im'
            
        img_fs_i_raw = sio.loadmat(self.files_fs_i[index % len(self.files_fs_i)])[input_name]
        img_fs_l_raw = sio.loadmat(self.files_fs_l[index % len(self.files_fs_l)])[label_name]
        
        img_shape = img_fs_l_raw.shape
        
        # load or generate sampling mask
        if not self.opt.random_sampling:
            mask = sio.loadmat('SamplingMask/mask_%.2f_%d_%d/mask%d.mat' % 
                               (self.opt.acc_rate, self.opt.acs_num, img_shape[0], self.opt.mask_index))['mask']
        else:
            mask = CartesianMask((img_shape[0], img_shape[1]), self.opt.acc_rate, self.opt.acs_num)
        
        if not self.is_testing and self.aug_seq != None:
            img_fs_i_raw = augment_data(img_fs_i_raw, img_shape, self.aug_seq)
            img_fs_l_raw = augment_data(img_fs_l_raw, img_shape, self.aug_seq)
            
        if self.opt.coil_num == 1:
            img_fs_i_raw = np.expand_dims(img_fs_i_raw, axis=2)
            
        if self.opt.coil_label != 'mc':
            img_fs_l_raw = np.expand_dims(img_fs_l_raw, axis=2)
            
        if self.brain_masking:
            bm = sio.loadmat(self.files_bm[index % len(self.files_bm)])['m']
            bm = Tensor(np.expand_dims(bm, axis=0))
                    
        kspaces_us_i, imgs_us_i, imgs_fs_i, masks_i = gen_data_from_image(img_fs_i_raw, mask)
        kspaces_us_l, imgs_us_l, imgs_fs_l, masks_l = gen_data_from_image(img_fs_l_raw, mask)
        
        if self.brain_masking:
            return {'kspace_us': kspaces_us_i, 'img_us': imgs_us_l,
                    'img_fs': imgs_fs_l, 'mask': masks_i, 'bm': bm}
        else:
            return {'kspace_us': kspaces_us_i, 'img_us': imgs_us_l,
                    'img_fs': imgs_fs_l, 'mask': masks_i}

    def __len__(self):
        return len(self.files_fs_i)