from turtle import mode
import torch
from torch import nn
from torchvision.models import vgg16

from modeling.arch import IFRNetv3, IFRNetv4S7, IFRNetv4P20, IFRNetv5 # Discriminator, PatchDiscriminator
from modeling.discriminator import Discriminator


# def build_model(input_size, n_channels):
#     vgg_feats = vgg16(pretrained=False).features
#     vgg_feats_layers = VGG16FeatLayer(vgg_feats)
#     vgg_feats_model = nn.Sequential(*[module for module in vgg_feats][:35]).eval()
#     for param in vgg_feats_model.parameters():
#         param.requires_grad = False
#     net = IFRNetv2(vgg_feats=vgg_feats_model, input_size=input_size, base_n_channels=n_channels)
#     return net, vgg_feats_layers


def build_model(n_channels, out_channels=3, model_type="s7"):
    if model_type == "s7":
        return IFRNetv4S7(base_n_channels=n_channels, out_channels=out_channels)
    elif model_type == "p20":
        return IFRNetv4P20(base_n_channels=n_channels, out_channels=out_channels)
    else:
        raise ValueError("Unknown model type: {}".format(model_type))
        


# def build_discriminators(n_channels, in_channels=3):
#     return Discriminator(base_n_channels=n_channels, in_channels=in_channels), PatchDiscriminator(base_n_channels=n_channels, in_channels=in_channels)


def build_discriminators(size, use_sn=True):
    return Discriminator(size=size, sn=use_sn)