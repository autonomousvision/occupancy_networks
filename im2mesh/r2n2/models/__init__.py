import torch.nn as nn
from im2mesh.r2n2.models.decoder import Decoder


# Decoder dictionary
decoder_dict = {
    'simple': Decoder,
}


class R2N2(nn.Module):
    ''' The 3D Recurrent Reconstruction Neural Network (3D-R2N2) model.

    For details regarding the model, please see
    https://arxiv.org/abs/1604.00449

    As single-view images are used as input, we do not use the recurrent
    module.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
    '''

    def __init__(self, decoder, encoder):
        super().__init__()
        self.decoder = decoder
        self.encoder = encoder

    def forward(self, x):
        c = self.encoder(x)
        occ_hat = self.decoder(c)
        return occ_hat
