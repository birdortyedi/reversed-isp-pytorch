from turtle import down, forward
from numpy import pad
import torch
from torch import nn

from modeling.normalization import EFDM, AdaIN
from modeling.discriminator import ConvLayer
from modeling.op import FusedLeakyReLU


class DestyleResBlock(nn.Module):
    def __init__(self, channels_out, kernel_size, channels_in=None, stride=1, dilation=1, padding=1, use_dropout=False, style_proj_n_ch=128):
        super(DestyleResBlock, self).__init__()

        # uses 1x1 convolutions for downsampling
        if not channels_in or channels_in == channels_out:
            channels_in = channels_out
            self.projection = None
        else:
            self.projection = nn.Conv2d(channels_in, channels_out, kernel_size=1, stride=stride, dilation=1)
        self.use_dropout = use_dropout

        self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels_out, channels_out, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation)
        self.adain = AdaIN()
        self.style_projector = nn.Linear(style_proj_n_ch, channels_out * 2)
        if self.use_dropout:
            self.dropout = nn.Dropout()
        self.lrelu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, feat):
        residual = x
        out = self.conv1(x)
        out = self.lrelu1(out)
        out = self.conv2(out)
        style_proj = self.style_projector(feat)
        out = self.adain(out, style_proj)
        if self.use_dropout:
            out = self.dropout(out)
        if self.projection:
            residual = self.projection(x)
        out = out + residual
        out = self.lrelu2(out)
        return out
    

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    

class DestyleGramResBlock(nn.Module):
    def __init__(self, channels_out, kernel_size, channels_in=None, stride=1, dilation=1, padding=1, use_dropout=False, use_attention=False) -> None:
        super(DestyleGramResBlock, self).__init__()
        # uses 1x1 convolutions for downsampling
        if not channels_in or channels_in == channels_out:
            channels_in = channels_out
            self.projection = None
        else:
            self.projection = nn.Conv2d(channels_in, channels_out, kernel_size=1, stride=stride, dilation=1)
        self.use_dropout = use_dropout
        self.use_attention = use_attention

        self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels_out, channels_out, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation)
        self.adain = AdaIN()
        self.style_projector = Destyler(channels_in ** 2, channels_out * 2)
        self.ca = ChannelAttention(channels_out)
        self.sa = SpatialAttention()
        if self.use_dropout:
            self.dropout = nn.Dropout()
        self.lrelu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
    def forward(self, x):
        residual = x
        
        G_feat = self.gram_matrix(x)
        style_proj = self.style_projector(G_feat)
        
        out = self.conv1(x)
        out = self.lrelu1(out)
        out = self.conv2(out)
        
        out = self.adain(out, style_proj)
        
        if self.use_attention:
            out = self.ca(out) * out
            out = self.sa(out) * out
        if self.use_dropout:
            out = self.dropout(out)
        if self.projection:
            residual = self.projection(x)
        out = out + residual
        out = self.lrelu2(out)
        return out
    
    def gram_matrix(self, x):
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        G = torch.bmm(features, features.transpose(1, 2))
        G = G / (h * w)
        return G.view(b, c * c)
    
    

class DestyleGramResBlockv2(DestyleGramResBlock):
    def __init__(self, channels_out, kernel_size, ds_num_layers, channels_in=None, stride=1, dilation=1, padding=1, use_dropout=False, use_attention=False) -> None:
        super(DestyleGramResBlockv2, self).__init__(channels_out, kernel_size, channels_in, stride, dilation, padding, use_dropout, use_attention)
        self.conv1 = ConvLayer(channels_in, channels_out, kernel_size)
        self.conv2 = ConvLayer(channels_out, channels_out, kernel_size, downsample=stride!=1, padding=padding)
        self.style_projector = Destylerv2(channels_in ** 2, channels_out * 2, num_layers=ds_num_layers)
        self.lrelu2 = FusedLeakyReLU(channels_out)
        
    def forward(self, x):
        residual = x
        
        G_feat = self.gram_matrix(x)
        style_proj = self.style_projector(G_feat)
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.adain(out, style_proj)
        
        if self.use_attention:
            out = self.ca(out) * out
            out = self.sa(out) * out
        if self.use_dropout:
            out = self.dropout(out)
        if self.projection:
            residual = self.projection(x)
        out = out + residual
        out = self.lrelu2(out)
        return out
    
    
class DestyleGramResBlockv3(DestyleGramResBlockv2):
    def __init__(self, channels_out, kernel_size, ds_num_layers, channels_in=None, stride=1, dilation=1, padding=1, use_dropout=False, use_attention=False) -> None:
        super().__init__(channels_out, kernel_size, ds_num_layers, channels_in, stride, dilation, padding, use_dropout, use_attention)
        # uses 1x1 convolutions for downsampling
        if not channels_in or channels_in == channels_out:
            channels_in = channels_out
            self.projection = None
        else:
            self.projection = ConvLayer(channels_in, channels_out, kernel_size=1, downsample=stride!=1)
        self.efdm = EFDM()
        self.style_projector = Destylerv3(channels_in, channels_out, num_layers=ds_num_layers, stride_first=stride, padding_first=padding)
    
    def forward(self, x):
        residual = x
        
        style_proj = self.style_projector(x)
        
        out = self.conv1(x)
        out = self.conv2(out)
        if self.use_attention:
            out = self.ca(out) * out
            out = self.sa(out) * out
        if self.use_dropout:
            out = self.dropout(out)
            
        out = self.efdm(out, style_proj)
        
        if self.projection:
            residual = self.projection(x)
        out = out + residual
        out = self.lrelu2(out)
        return out


class Destylerv3(nn.Module):
    def __init__(self, in_channel, out_channel, num_layers=5, stride_first=1, padding_first=0) -> None:
        super().__init__()
        layers = [ConvLayer(in_channel, out_channel, kernel_size=3, downsample=stride_first!=1, padding=padding_first, activate=False),]
        for _ in range(num_layers-1):
            layers.append(ConvLayer(out_channel, out_channel, kernel_size=3, padding=1, activate=False))
        self.destyler = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.destyler(x)
    


class ResBlock(nn.Module):
    def __init__(self, channels_out, kernel_size, channels_in=None, stride=1, dilation=1, padding=1, use_dropout=False):
        super(ResBlock, self).__init__()

        # uses 1x1 convolutions for downsampling
        if not channels_in or channels_in == channels_out:
            channels_in = channels_out
            self.projection = None
        else:
            self.projection = nn.Conv2d(channels_in, channels_out, kernel_size=1, stride=stride, dilation=1)
        self.use_dropout = use_dropout

        self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels_out, channels_out, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation)
        self.n2 = nn.BatchNorm2d(channels_out)
        if self.use_dropout:
            self.dropout = nn.Dropout()
        self.lrelu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.lrelu1(out)
        out = self.conv2(out)
        # out = self.n2(out)
        if self.use_dropout:
            out = self.dropout(out)
        if self.projection:
            residual = self.projection(x)
        out = out + residual
        out = self.lrelu2(out)
        return out
    

class ResBlockv2(nn.Module):
    def __init__(self, channels_out, kernel_size, channels_in=None, stride=1, padding=0, dilation=1, use_dropout=False, use_attention=False):
        super(ResBlockv2, self).__init__()
        # uses 1x1 convolutions for downsampling
        if (not channels_in or channels_in == channels_out):
            channels_in = channels_out
            self.projection = None
        else:
            self.projection = nn.Conv2d(channels_in, channels_out, kernel_size=1, stride=stride, dilation=1)
            
        self.use_dropout = use_dropout
        self.use_attention = use_attention
        
        self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels_out, channels_out, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation)
        self.n2 = nn.BatchNorm2d(channels_out)
        if self.use_dropout:
            self.dropout = nn.Dropout()
        self.lrelu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        if self.use_attention:
            self.ca = ChannelAttention(channels_out)
            self.sa = SpatialAttention()
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.use_dropout:
            out = self.dropout(out)
        if self.projection:
            residual = self.projection(x)
        if self.use_attention:
            out = self.ca(out) * out
            out = self.sa(out) * out
        out = out + residual
        out = self.lrelu2(out)
        return out
    

class ConvBlock(nn.Module):
    def __init__(self, channels_out, kernel_size, channels_in=None, stride=1, dilation=1, padding=1, use_dropout=False):
        super(ConvBlock, self).__init__()

        # uses 1x1 convolutions for downsampling
        if not channels_in or channels_in == channels_out:
            channels_in = channels_out
            self.projection = None
        else:
            self.projection = nn.Conv2d(channels_in, channels_out, kernel_size=1, stride=stride, dilation=1)
        self.use_dropout = use_dropout

        self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels_out, channels_out, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation)
        self.n2 = nn.BatchNorm2d(channels_out)
        if self.use_dropout:
            self.dropout = nn.Dropout()
        self.lrelu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.lrelu1(out)
        out = self.conv2(out)
        # out = self.n2(out)
        if self.use_dropout:
            out = self.dropout(out)
        if self.projection:
            residual = self.projection(x)
        out = self.lrelu2(out)
        return out

class Destyler(nn.Module):
    def __init__(self, in_features, num_features):
        super(Destyler, self).__init__()
        self.fc1 = nn.Linear(in_features, num_features)
        self.fc2 = nn.Linear(num_features, num_features)
        self.fc3 = nn.Linear(num_features, num_features)
        self.fc4 = nn.Linear(num_features, num_features)
        self.fc5 = nn.Linear(num_features, num_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x


class Destylerv2(nn.Module):
    def __init__(self, in_features, num_features, num_layers=5) -> None:
        super(Destylerv2, self).__init__()
        layers = [nn.Linear(in_features, num_features),]
        for _ in range(num_layers-1):
            layers.append(nn.Linear(num_features, num_features))
        self.fc = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.fc(x)