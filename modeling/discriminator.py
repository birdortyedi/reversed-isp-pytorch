# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

from modeling.base import BaseNetwork
from modeling.op import FusedLeakyReLU, upfirdn2d
from modeling.layers import Blur, Downsample, EqualConv2d, EqualLinear, ScaledLeakyReLU, Flatten


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        padding=0,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
        sn=False
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2
        
        if padding:
            self.padding = padding

        if sn:
            # Not use equal conv2d when apply SN
            layers.append(
                spectral_norm(nn.Conv2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=self.padding,
                    stride=stride,
                    bias=bias and not activate,
                ))
            )
        else:
            layers.append(
                EqualConv2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=self.padding,
                    stride=stride,
                    bias=bias and not activate,
                )
            )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))
            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1], sn=False):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3, sn=sn)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True, sn=sn)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        return out


def get_haar_wavelet(in_channels):
    haar_wav_l = 1 / (2 ** 0.5) * torch.ones(1, 2)
    haar_wav_h = 1 / (2 ** 0.5) * torch.ones(1, 2)
    haar_wav_h[0, 0] = -1 * haar_wav_h[0, 0]

    haar_wav_ll = haar_wav_l.T * haar_wav_l
    haar_wav_lh = haar_wav_h.T * haar_wav_l
    haar_wav_hl = haar_wav_l.T * haar_wav_h
    haar_wav_hh = haar_wav_h.T * haar_wav_h
    
    return haar_wav_ll, haar_wav_lh, haar_wav_hl, haar_wav_hh


class HaarTransform(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        ll, lh, hl, hh = get_haar_wavelet(in_channels)
    
        self.register_buffer('ll', ll)
        self.register_buffer('lh', lh)
        self.register_buffer('hl', hl)
        self.register_buffer('hh', hh)
        
    def forward(self, input):
        ll = upfirdn2d(input, self.ll, down=2)
        lh = upfirdn2d(input, self.lh, down=2)
        hl = upfirdn2d(input, self.hl, down=2)
        hh = upfirdn2d(input, self.hh, down=2)
        
        return torch.cat((ll, lh, hl, hh), 1)
    

class InverseHaarTransform(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        ll, lh, hl, hh = get_haar_wavelet(in_channels)

        self.register_buffer('ll', ll)
        self.register_buffer('lh', -lh)
        self.register_buffer('hl', -hl)
        self.register_buffer('hh', hh)
        
    def forward(self, input):
        ll, lh, hl, hh = input.chunk(4, 1)
        ll = upfirdn2d(ll, self.ll, up=2, pad=(1, 0, 1, 0))
        lh = upfirdn2d(lh, self.lh, up=2, pad=(1, 0, 1, 0))
        hl = upfirdn2d(hl, self.hl, up=2, pad=(1, 0, 1, 0))
        hh = upfirdn2d(hh, self.hh, up=2, pad=(1, 0, 1, 0))
        
        return ll + lh + hl + hh


class FromRGB(nn.Module):
    def __init__(self, out_channel, downsample=True, blur_kernel=[1, 3, 3, 1], sn=False):
        super().__init__()

        self.downsample = downsample

        if downsample:
            self.iwt = InverseHaarTransform(4)
            self.downsample = Downsample(blur_kernel)
            self.dwt = HaarTransform(4)

        self.conv = ConvLayer(4 * 4, out_channel, 1, sn=sn)

    def forward(self, input, skip=None):
        if self.downsample:
            input = self.iwt(input)
            input = self.downsample(input)
            input = self.dwt(input)

        out = self.conv(input)

        if skip is not None:
            out = out + skip

        return input, out


class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], sn=False, ssd=False):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            252: 64 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.dwt = HaarTransform(3)

        self.from_rgbs = nn.ModuleList()
        self.convs = nn.ModuleList()

        log_size = int(math.log(size, 2)) - 1

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            self.from_rgbs.append(FromRGB(in_channel, downsample=i != log_size, sn=sn))
            self.convs.append(ConvBlock(in_channel, out_channel, blur_kernel, sn=sn))

            in_channel = out_channel

        self.from_rgbs.append(FromRGB(channels[4], sn=sn))

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3, sn=sn)
        if sn:
            self.final_linear = nn.Sequential(
                spectral_norm(nn.Linear(channels[4] * 4 * 4, channels[4])),
                FusedLeakyReLU(channels[4]),
                spectral_norm(nn.Linear(channels[4], 1)),
        )
        else:
            self.final_linear = nn.Sequential(
                EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
                EqualLinear(channels[4], 1),
            )

    def forward(self, input):
        input = self.dwt(input)
        out = None

        for from_rgb, conv in zip(self.from_rgbs, self.convs):
            input, out = from_rgb(input, out)
            out = conv(out)

        _, out = self.from_rgbs[-1](input, out)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out
    

class Discriminatorv1(BaseNetwork):
    def __init__(self, base_n_channels, in_channels=3):
        """
        img_size : (int, int, int)
            Height and width must be powers of 2.  E.g. (32, 32, 1) or
            (64, 128, 3). Last number indicates number of channels, e.g. 1 for
            grayscale or 3 for RGB
        """
        super(Discriminatorv1, self).__init__()

        self.image_to_features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, base_n_channels, 5, 2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_n_channels, 2 * base_n_channels, 5, 2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(2 * base_n_channels, 2 * base_n_channels, 5, 2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(2 * base_n_channels, 4 * base_n_channels, 5, 2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            # spectral_norm(nn.Conv2d(4 * base_n_channels, 4 * base_n_channels, 5, 2, 2)),
            # nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(4 * base_n_channels, 8 * base_n_channels, 5, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        output_size = 8 * base_n_channels * 3 * 3
        self.features_to_prob = nn.Sequential(
            spectral_norm(nn.Conv2d(8 * base_n_channels, 2 * base_n_channels, 5, 2, 1)),
            Flatten(),
            nn.Linear(output_size, 1)
        )

        self.init_weights(init_type="normal", gain=0.02)

    def forward(self, input_data):
        x = self.image_to_features(input_data)
        return self.features_to_prob(x)


class PatchDiscriminator(Discriminatorv1):
    def __init__(self, base_n_channels, in_channels=3):
        super(PatchDiscriminator, self).__init__(base_n_channels, in_channels)

        self.features_to_prob = nn.Sequential(
            spectral_norm(nn.Conv2d(8 * base_n_channels, 1, 1)),
            Flatten()
        )

    def forward(self, input_data):
        x = self.image_to_features(input_data)
        return self.features_to_prob(x)
    

if __name__ == "__main__":
    x = torch.randn(2, 4, 256, 256)
    discriminator = Discriminator(256)
    out = discriminator(x)
    print(out.size())