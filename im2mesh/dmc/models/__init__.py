import torch.nn as nn
from im2mesh.dmc.models import encoder, decoder


decoder_dict = {
    'unet': decoder.UNetDecoder
}

encoder_dict = {
    'pointnet_local': encoder.PointNetLocal,
}

class DMC(nn.Module):
    def __init__(self, decoder, encoder):
        super().__init__()
        self.decoder = decoder
        self.encoder = encoder

    def forward(self, x):
        c = self.encoder(x)
        offset, topology, occupancy = self.decoder(c)

        return offset, topology, occupancy
