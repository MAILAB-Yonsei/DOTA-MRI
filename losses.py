from torch import nn
from layer_utils import fft2

class dcloss(nn.Module):
    def __init__(self):
        super(dcloss, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, img_recon, k_input, mask):
        k_recon = fft2(img_recon)
        loss = self.loss(k_recon * mask, k_input)

        return loss
    
class imgloss(nn.Module):
    def __init__(self):
        super(imgloss, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, img_recon, img_label):
        loss = self.loss(img_recon, img_label)

        return loss

loss_dict = {'img': imgloss, 'dc': dcloss}