import numpy as np
import math
import torch

from imgaug import augmenters as iaa
import matplotlib.pyplot as plt

def GenAddPathStr(opt):
    add_path_str = opt.dataset_name + '_'   
    add_path_str += 'acc%.2f_acs%d_mask%d' % (opt.acc_rate, opt.acs_num, opt.mask_index)
    
    if 'K' in opt.model_name:
        add_path_str += '_K%d' % opt.k
    
    if 'Cascade_K' not in opt.model_name and opt.model_name != 'DOTA':
        add_path_str += '_I%d_iters%d' % (opt.i, opt.iters)
    
    if opt.data_augment:
        add_path_str = add_path_str + '_DataAug'
    if opt.random_sampling:
        add_path_str = add_path_str + '_RandomSampling'
    
    add_path_str = add_path_str + opt.add_path_str
    
    return add_path_str

def GenAugSeq():
    aug_seq = iaa.SomeOf((0, None), [iaa.Affine(rotate=90),
                                     iaa.Flipud(1.0),
                                     iaa.Fliplr(1.0),
                                     iaa.PiecewiseAffine(scale=0.01)])
    return aug_seq

def save_images(val_data, loss_valid, epoch, batches_done, model_name, add_path_str, Tensor):
    titles = ['Under-sampled','Reconstructed (epoch=%d, loss=%.4f)' % (epoch, math.sqrt(loss_valid)), 'Full-sampled']
    fig, axs = plt.subplots(2, 3)
    
    for col, image in enumerate([val_data[0], val_data[1], val_data[2]]):
        # Tensor to numpy, and extract magnitude
        img_np = Tensor.cpu(image.detach()).numpy()
        
        if img_np.shape[1] == 2:
            img_np = img_np[0,0,:,:] + 1j * img_np[0,1,:,:]
        elif img_np.shape[1] == 1:
            img_np = img_np[0,0,:,:]
            
        kspace_np = np.fft.fftshift(np.fft.fft2(img_np))
        kspace_abs = np.abs(kspace_np) ** 0.2
        img_abs = np.abs(img_np)
        axs[0, col].imshow(kspace_abs, cmap='gray',vmin=0,vmax=10)
        axs[0, col].set_title(titles[col],fontdict={'fontsize': 5})
        axs[0, col].axis('off')
        axs[1, col].imshow(img_abs, cmap='gray',vmin=0,vmax=1)
        axs[1, col].axis('off')
    fig.savefig("Validation_%s/%s/%d.png" % (model_name, add_path_str, batches_done), dpi = 300)
    plt.close()

def NormalPdf(length, sensitivity):
    return np.exp(-sensitivity * (np.arange(length) - length / 2)**2)

def CartesianMask(shape, acc, acs_num=10, centered=False):
    Nx, Ny = shape[-2], shape[-1]
    pdf_x = NormalPdf(Nx, 0.5/(Nx/10.)**2)
    lmda = Nx/(2.*acc)
    n_lines = int(Nx / acc)

    # add uniform distribution
    pdf_x += lmda * 1./Nx

    if acs_num:
        pdf_x[Nx//2-acs_num//2:Nx//2+acs_num//2] = 0
        pdf_x /= np.sum(pdf_x)
        n_lines -= acs_num

    mask = np.zeros((Nx, Ny))
    
    idx = np.random.choice(Nx, n_lines, False, pdf_x)
    mask[idx, :] = 1

    if acs_num:
        mask[Nx//2-acs_num//2:Nx//2+acs_num//2, :] = 1

    if not centered:
        mask = np.fft.ifftshift(mask, axes=(-1, -2))

    return mask

def augment_data(img, img_shape, aug_seq):
    img = np.reshape(img, (img_shape[0], img_shape[1], 1))
    img = aug_seq.augment_image(img)
    img = img[:,:,0]
    return img

def gen_data_from_image(img, m):
    mask = np.expand_dims(m, axis=0)
    mask = torch.Tensor(mask)
    
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis = 2)
    
    for c in range(img.shape[2]):
        img_fs = img[:,:,c]
        kspace_fs   = np.fft.fft2(img_fs)
        kspace_us   = np.fft.ifftshift(np.fft.fftshift(kspace_fs) * np.fft.fftshift(m))
        img_us      = np.fft.ifft2(kspace_us)
        
        kspace_us = np.stack((np.real(kspace_us), np.imag(kspace_us)), axis = 0)
        kspace_fs = np.stack((np.real(kspace_fs), np.imag(kspace_fs)), axis = 0)
        img_us    = np.stack((np.real(img_us),    np.imag(img_us)),    axis = 0)
        img_fs    = np.stack((np.real(img_fs),    np.imag(img_fs)),    axis = 0)
        
        kspace_us = torch.Tensor(kspace_us)
        # kspace_fs = torch.Tensor(kspace_fs)
        img_us    = torch.Tensor(img_us)
        img_fs    = torch.Tensor(img_fs)
        
        if c == 0:
            kspaces_us = kspace_us
            # kspaces_fs = kspace_fs
            imgs_us    = img_us
            imgs_fs    = img_fs
        else:
            # print(kspaces_us.shape)
            # print(kspace_us.shape)
            kspaces_us = torch.cat((kspaces_us, kspace_us), 0)
            # kspaces_fs = torch.cat((kspaces_fs, kspace_fs), 0)
            imgs_us    = torch.cat((imgs_us, img_us), 0)
            imgs_fs    = torch.cat((imgs_fs, img_fs), 0)
            
    return kspaces_us, imgs_us, imgs_fs, mask