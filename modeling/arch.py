import torch
from torch import nn
from torch.nn.utils import spectral_norm

from modeling.base import BaseNetwork
from modeling.blocks import ConvBlock, ResBlock, DestyleGramResBlock, ResBlockv2, DestyleGramResBlockv2, DestyleGramResBlockv3
from modeling.layers import EqualConv2d
from modeling.discriminator import Discriminator, PatchDiscriminator


class IFRNetv3(BaseNetwork):
    def __init__(self, base_n_channels, out_channels=3):
        super(IFRNetv3, self).__init__()
        self.ds_res1 = DestyleGramResBlock(channels_in=3, channels_out=base_n_channels, kernel_size=5, stride=1, padding=2)
        self.ds_res2 = DestyleGramResBlock(channels_in=base_n_channels, channels_out=base_n_channels * 2, kernel_size=3, stride=2, padding=1)
        self.ds_res3 = DestyleGramResBlock(channels_in=base_n_channels * 2, channels_out=base_n_channels * 2, kernel_size=3, stride=1, padding=1)
        self.ds_res4 = DestyleGramResBlock(channels_in=base_n_channels * 2, channels_out=base_n_channels * 4, kernel_size=3, stride=2, padding=1)
        self.ds_res5 = DestyleGramResBlock(channels_in=base_n_channels * 4, channels_out=base_n_channels * 8, kernel_size=3, stride=2, padding=1, use_attention=True)
        self.ds_res6 = DestyleGramResBlock(channels_in=base_n_channels * 8, channels_out=base_n_channels * 16, kernel_size=3, stride=2, padding=1, use_attention=True)

        self.upsample = nn.PixelShuffle(upscale_factor=2)

        self.res2 = ResBlock(channels_in=base_n_channels * 16, channels_out=base_n_channels * 32, kernel_size=1, stride=1, padding=0)
        self.res3 = ResBlock(channels_in=base_n_channels * 8, channels_out=base_n_channels * 16, kernel_size=1, stride=1, padding=0)
        self.res4 = ResBlock(channels_in=base_n_channels * 4, channels_out=base_n_channels * 4, kernel_size=1, stride=1, padding=0)
        self.res5 = ConvBlock(channels_in=base_n_channels, channels_out=out_channels, kernel_size=3, stride=1, padding=0)

        self.init_weights(init_type="normal", gain=0.02)
        self._init_resblock_pixelshuffle([self.res2, self.res3, self.res4])
    
    def _init_resblock_pixelshuffle(self, layers):
        def _init_subpixel(weight, bias):
            co, ci, h, w = weight.shape
            co2 = co // 4
            # initialize sub kernel
            k = torch.empty([co2, ci, h, w])
            nn.init.kaiming_uniform_(k)
            # repeat 4 times
            k = k.repeat_interleave(4, dim=0)
            weight.data.copy_(k)
            bias.data.zero_()
        for layer in layers:
            _init_subpixel(layer.conv1.weight, layer.conv1.bias)
            _init_subpixel(layer.conv2.weight, layer.conv2.bias)
        
    def forward(self, x):        
        out = self.ds_res1(x)
        out = self.ds_res2(out)
        out = self.ds_res3(out)
        out = self.ds_res4(out)
        out = self.ds_res5(out)
        out = self.ds_res6(out)
        
        out = self.res2(out)
        out = self.upsample(out)
        out = self.res3(out)
        out = self.upsample(out)
        out = self.res4(out)
        out = self.upsample(out)
        out = self.res5(out)
        
        return out


class IFRNetv4P20(BaseNetwork):
    def __init__(self, base_n_channels, out_channels=3):
        super(IFRNetv4P20, self).__init__()
        self.ds_res1 = DestyleGramResBlockv2(channels_in=3, channels_out=base_n_channels, ds_num_layers=2, kernel_size=5, stride=1, padding=0)
        self.ds_res2 = DestyleGramResBlockv2(channels_in=base_n_channels, channels_out=base_n_channels * 2, ds_num_layers=2, kernel_size=3, stride=2, padding=0)
        self.ds_res3 = DestyleGramResBlockv2(channels_in=base_n_channels * 2, channels_out=base_n_channels * 2, ds_num_layers=2, kernel_size=3, stride=1, padding=0)
        self.ds_res4 = DestyleGramResBlockv2(channels_in=base_n_channels * 2, channels_out=base_n_channels * 4, ds_num_layers=2, kernel_size=3, stride=2, padding=0)
        self.ds_res5 = DestyleGramResBlockv2(channels_in=base_n_channels * 4, channels_out=base_n_channels * 8, ds_num_layers=2, kernel_size=3, stride=2, padding=0, use_attention=True)
        self.ds_res6 = DestyleGramResBlockv2(channels_in=base_n_channels * 8, channels_out=base_n_channels * 16, ds_num_layers=2, kernel_size=3, stride=2, padding=0, use_attention=True)  # change padding for s7 (1) and p20 (0)

        self.upsample = nn.PixelShuffle(upscale_factor=2)

        self.res2 = ResBlockv2(channels_in=base_n_channels * 16, channels_out=base_n_channels * 32, kernel_size=1, stride=1, padding=0)
        self.res3 = ResBlockv2(channels_in=base_n_channels * 8, channels_out=base_n_channels * 16, kernel_size=1, stride=1, padding=0)
        self.res4 = ResBlockv2(channels_in=base_n_channels * 4, channels_out=base_n_channels * 4, kernel_size=1, stride=1, padding=0)
        self.res5 = EqualConv2d(base_n_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False) 
        # self.res6 = EqualConv2d(base_n_channels, out_channels, kernel_size=3, padding=0, stride=1, bias=False)  # change padding for s7 (0) and p20 (1)

        self.init_weights(init_type="normal", gain=0.02)
        # self._init_resblock_pixelshuffle([self.res2, self.res3, self.res4])
    
    def _init_resblock_pixelshuffle(self, layers):
        def _init_subpixel(weight, bias):
            co, ci, h, w = weight.shape
            co2 = co // 4
            # initialize sub kernel
            k = torch.empty([co2, ci, h, w])
            nn.init.kaiming_uniform_(k)
            # repeat 4 times
            k = k.repeat_interleave(4, dim=0)
            weight.data.copy_(k)
            bias.data.zero_()
        for layer in layers:
            _init_subpixel(layer.conv1.weight, layer.conv1.bias)
            _init_subpixel(layer.conv2.weight, layer.conv2.bias)
        
    def forward(self, x):        
        out = self.ds_res1(x)
        out = self.ds_res2(out)
        out = self.ds_res3(out)
        out = self.ds_res4(out)
        out = self.ds_res5(out)
        out = self.ds_res6(out)
        
        out = self.res2(out)
        out = self.upsample(out)
        out = self.res3(out)
        out = self.upsample(out)
        out = self.res4(out)
        out = self.upsample(out)
        out = self.res5(out)
        # out = self.res5_2(out)
        # out = self.res6(out)
        
        return out
    

class IFRNetv4S7(BaseNetwork):
    def __init__(self, base_n_channels, out_channels=3):
        super(IFRNetv4S7, self).__init__()
        self.ds_res1 = DestyleGramResBlockv2(channels_in=3, channels_out=base_n_channels, ds_num_layers=2, kernel_size=5, stride=1, padding=0)
        self.ds_res2 = DestyleGramResBlockv2(channels_in=base_n_channels, channels_out=base_n_channels * 2, ds_num_layers=2, kernel_size=3, stride=2, padding=0)
        self.ds_res3 = DestyleGramResBlockv2(channels_in=base_n_channels * 2, channels_out=base_n_channels * 2, ds_num_layers=2, kernel_size=3, stride=1, padding=0)
        self.ds_res4 = DestyleGramResBlockv2(channels_in=base_n_channels * 2, channels_out=base_n_channels * 4, ds_num_layers=2, kernel_size=3, stride=2, padding=0)
        self.ds_res5 = DestyleGramResBlockv2(channels_in=base_n_channels * 4, channels_out=base_n_channels * 8, ds_num_layers=2, kernel_size=3, stride=2, padding=0, use_attention=True)
        self.ds_res6 = DestyleGramResBlockv2(channels_in=base_n_channels * 8, channels_out=base_n_channels * 16, ds_num_layers=2, kernel_size=3, stride=2, padding=1, use_attention=True)  # change padding for s7 (1) and p20 (0)

        self.upsample = nn.PixelShuffle(upscale_factor=2)

        self.res2 = ResBlockv2(channels_in=base_n_channels * 16, channels_out=base_n_channels * 32, kernel_size=1, stride=1, padding=0)
        self.res3 = ResBlockv2(channels_in=base_n_channels * 8, channels_out=base_n_channels * 16, kernel_size=1, stride=1, padding=0)
        self.res4 = ResBlockv2(channels_in=base_n_channels * 4, channels_out=base_n_channels * 4, kernel_size=1, stride=1, padding=0)
        self.res5 = EqualConv2d(base_n_channels, base_n_channels, kernel_size=3, padding=0, stride=1, bias=False) 
        self.res6 = EqualConv2d(base_n_channels, out_channels, kernel_size=3, padding=0, stride=1, bias=False)  # change padding for s7 (0) and p20 (1)

        self.init_weights(init_type="normal", gain=0.02)
        # self._init_resblock_pixelshuffle([self.res2, self.res3, self.res4])
    
    def _init_resblock_pixelshuffle(self, layers):
        def _init_subpixel(weight, bias):
            co, ci, h, w = weight.shape
            co2 = co // 4
            # initialize sub kernel
            k = torch.empty([co2, ci, h, w])
            nn.init.kaiming_uniform_(k)
            # repeat 4 times
            k = k.repeat_interleave(4, dim=0)
            weight.data.copy_(k)
            bias.data.zero_()
        for layer in layers:
            _init_subpixel(layer.conv1.weight, layer.conv1.bias)
            _init_subpixel(layer.conv2.weight, layer.conv2.bias)
        
    def forward(self, x):        
        out = self.ds_res1(x)
        out = self.ds_res2(out)
        out = self.ds_res3(out)
        out = self.ds_res4(out)
        out = self.ds_res5(out)
        out = self.ds_res6(out)
        
        out = self.res2(out)
        out = self.upsample(out)
        out = self.res3(out)
        out = self.upsample(out)
        out = self.res4(out)
        out = self.upsample(out)
        out = self.res5(out)
        # out = self.res5_2(out)
        out = self.res6(out)
        
        return out
    
    
class IFRNetv5(BaseNetwork):
    def __init__(self, base_n_channels, out_channels=3):
        super(IFRNetv5, self).__init__()
        # self.res1 = ResBlockv2(channels_in=3, channels_out=base_n_channels, kernel_size=5, downsample=True, use_dropout=True, use_attention=False)
        self.res1 = DestyleGramResBlockv3(channels_in=3, channels_out=base_n_channels, ds_num_layers=5, 
                                          kernel_size=3, stride=2, padding=0, use_attention=True) 
        # self.res2 = ResBlockv2(channels_in=base_n_channels, channels_out=base_n_channels * 2, kernel_size=3, downsample=True, use_dropout=True, use_attention=False)
        self.res2 = DestyleGramResBlockv3(channels_in=base_n_channels, channels_out=base_n_channels * 2, ds_num_layers=5,
                                          kernel_size=3, stride=2, padding=0, use_attention=False) 
        # self.res3 = ResBlockv2(channels_in=base_n_channels * 2, channels_out=base_n_channels * 4, kernel_size=3, downsample=True, use_dropout=True, use_attention=True)
        self.res3 = DestyleGramResBlockv3(channels_in=base_n_channels * 2, channels_out=base_n_channels * 4, ds_num_layers=5,
                                          kernel_size=3, stride=2, padding=0, use_attention=True) 
        # self.res4 = ResBlockv2(channels_in=base_n_channels * 4, channels_out=base_n_channels * 8, kernel_size=3, downsample=True, use_dropout=True, use_attention=True)
        self.res4 = DestyleGramResBlockv3(channels_in=base_n_channels * 4, channels_out=base_n_channels * 8, ds_num_layers=5, 
                                          kernel_size=3, stride=2, padding=0, use_attention=True) 
        self.res5 = DestyleGramResBlockv3(channels_in=base_n_channels * 8, channels_out=base_n_channels * 16, ds_num_layers=5, 
                                                   kernel_size=3, stride=1, padding=0, use_attention=True)  # change padding for s7 (1) and p20 (0)

        self.upsample = nn.PixelShuffle(upscale_factor=2)
        self.pad = nn.ReflectionPad2d((0, 1, 1, 0))

        self.up_res2 = ResBlockv2(channels_in=base_n_channels * 16, channels_out=base_n_channels * 32, kernel_size=1)
        self.up_res3 = ResBlockv2(channels_in=base_n_channels * 8, channels_out=base_n_channels * 16, kernel_size=1)
        self.up_res4 = ResBlockv2(channels_in=base_n_channels * 4, channels_out=base_n_channels * 4, kernel_size=1)
        self.up_conv1 = EqualConv2d(base_n_channels, base_n_channels, kernel_size=3, padding=0, stride=1, bias=False)  # change padding for s7 (0) and p20 (1)
        self.up_conv2 = EqualConv2d(base_n_channels, out_channels, kernel_size=3, padding=0, stride=1, bias=False) 

        self.init_weights(init_type="normal", gain=0.02)
        
    def forward(self, x):  
        out = self.res1(x)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        out = self.res5(out)
        
        out = self.up_res2(out)
        out = self.pad(out)
        out = self.upsample(out)
        out = self.up_res3(out)
        out = self.upsample(out)
        out = self.up_res4(out)
        out = self.upsample(out)
        out = self.up_conv1(out)
        out = self.up_conv2(out)
        
        return out
        
    


if __name__ == '__main__':
    # from torchvision.models.vgg import vgg16
    # vgg_feats = vgg16(pretrained=True).features.eval().cuda()
    # vgg_feats = nn.Sequential(*[module for module in vgg_feats][:35])
    ifrnet = IFRNetv4P20(32, 4)
    x = torch.rand((4, 3, 496, 496))
    output = ifrnet(x)
    print(output.size())
    ifrnet.print_network()

    # disc = Discriminator(32, 4).cuda()
    # d_out = disc(output)
    # print(d_out.size())

    # patch_disc = PatchDiscriminator(32, 4).cuda()
    # p_d_out = patch_disc(output)
    # print(p_d_out.size())