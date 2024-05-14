import torch
import numpy as np
import torchvision
import torch.nn.functional as F

# ~ from blocks.CNN import PeriodicPadding2d, ConvUnit, UnitaryUnit


class conv_block(torch.nn.Module):

    def __init__(self, in_c, out_c, prob):
        super(conv_block, self).__init__()

        self.conv1 = torch.nn.Conv3d(in_c, out_c, kernel_size = 3, padding=1)
        self.dropout1 = torch.nn.Dropout(prob)

        self.conv2 = torch.nn.Conv3d(out_c, out_c, kernel_size = 3, padding=1)
        self.dropout2 = torch.nn.Dropout(prob)
        self.relu  = torch.nn.GELU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        return x
	
class encoder_block(torch.nn.Module):
    
    def __init__(self, in_c, out_c, prob, phi_dir=None):
        super(encoder_block, self).__init__()

        self.conv = conv_block(in_c, out_c, prob)

        if phi_dir=='phi_dir':
            self.pool = torch.nn.AvgPool3d(kernel_size = (1,1,2), stride = (1,1,2))
        else:
            self.pool = torch.nn.AvgPool3d((2,2,2))


    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p
		
class decoder_block(torch.nn.Module):

    def __init__(self, in_c, out_c, prob, phi_dir=None):
        super(decoder_block, self).__init__()

        if phi_dir=='phi_dir':
            self.up   = torch.nn.ConvTranspose3d(in_c, out_c, kernel_size = (1,1,2), stride = (1,1,2), padding = 0)
        else:
            self.up   = torch.nn.ConvTranspose3d(in_c, out_c, kernel_size = (2,2,2), stride = (2,2,2), padding = 0)

        self.conv = conv_block(out_c + out_c, out_c, prob)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        # ~ skip = self.crop(x, skip)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

    # ~ def crop(self, inputs, skip):
        # ~ _,_,H,W,D = inputs.shape
        # ~ skip    = torchvision.transforms.RandomCrop([H,W,D])(skip)
        # ~ return skip
	
class Unet(torch.nn.Module):
    def __init__(self, name):
        super(Unet, self).__init__()

        self.name = name

        self.e1 = encoder_block(41,  64,  0.0)
        self.e2 = encoder_block(64,  128,  0.0)
        self.e3 = encoder_block(128,  256,  0.6)
        self.e4 = encoder_block(256,  256,  0.0, phi_dir='phi_dir')

        self.b = conv_block(256, 512, 0.0)

        self.d1 = decoder_block(512, 256, 0.0, phi_dir='phi_dir')
        self.d2 = decoder_block(256, 256,  0.0)
        self.d3 = decoder_block(256, 128,  0.6)
        self.d4 = decoder_block(128, 64,  0.0)

        self.output = torch.nn.Conv3d(64, 1, kernel_size = 1, padding=0)

    def forward(self, inputs):
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        b = self.b(p4)

        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        outputs = self.output(d4)
        outputs = F.interpolate(outputs, size = (32, 32, 256))

        return outputs
		
