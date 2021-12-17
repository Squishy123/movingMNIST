import torch
import torch.nn as nn

from collections import OrderedDict

'''
Context Autoencoder Class
'''


class ContextAutoencoder(nn.Module):
    # accepts input for number of transforms
    def __init__(self, frame_stacks=1, channels=1):
        super(ContextAutoencoder, self).__init__()

        self.encoder = nn.Sequential(OrderedDict([
            ('encoder_conv1', nn.Conv2d(channels * frame_stacks, 16, kernel_size=3, stride=2, padding=1)),
            ('encoder_relu1', nn.ReLU()),
            ('encoder_conv2',  nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)),
            ('encoder_relu2', nn.ReLU()),
            ('encoder_conv3', nn.Conv2d(32, 64, kernel_size=7)),
            ('encoder_relu3', nn.LeakyReLU()),
        ]))

        self.bottleneck = nn.Sequential(OrderedDict([
            ('bottleneck_conv1', nn.Conv2d(64, 64, kernel_size=(1, 1))),
            ('bottleneck_relu1', nn.ReLU()),
        ]))

        self.decoder = nn.Sequential(OrderedDict([
            ('decoder_Tconv1', nn.ConvTranspose2d(64, 32, kernel_size=7)),
            ('decoder_relu1', nn.ReLU()),
            ('decoder_Tconv2', nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)),
            ('decoder_relu2', nn.ReLU()),
            ('decoder_Tconv3', nn.ConvTranspose2d(16, channels * frame_stacks, kernel_size=3, stride=2, padding=1, output_padding=1)),
            ('decoder_relu3', nn.LeakyReLU()),
        ]))

    # forward pass
    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x

'''
x = torch.randn(1000, 10, 64, 64)
model = ContextAutoencoder(10)
model(x)
torch.onnx.export(model, x, "context_autoencoder.onnx", input_names=['input'], output_names=['output'], do_constant_folding=True)
'''