import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from utils.path_hyperparameter import ph
import cv2
from torchvision import transforms as T
from pathlib import Path
from DCNv2.dcn_v2 import DCN


class BITCBAM(nn.Module):

    def __init__(self, in_channel):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.k = kernel_size(in_channel)
        self.channel_conv1 = nn.Conv1d(2, 1, kernel_size=self.k, padding=self.k // 2)
        self.channel_conv2 = nn.Conv1d(2, 1, kernel_size=self.k, padding=self.k // 2)
        self.spatial_conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.spatial_conv2 = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.softmax = nn.Softmax(0)

    def forward(self, t1, t2, module_name=None,
                img_name=None):
        t1_channel_avg_pool = self.avg_pool(t1)
        t1_channel_max_pool = self.max_pool(t1)
        t2_channel_avg_pool = self.avg_pool(t2)
        t2_channel_max_pool = self.max_pool(t2)

        channel_pool_t1 = torch.cat([t1_channel_avg_pool, t2_channel_max_pool],
                                 dim=2).squeeze(-1).transpose(1, 2)
        channel_pool_t2 = torch.cat([t2_channel_avg_pool, t1_channel_max_pool],
                                 dim=2).squeeze(-1).transpose(1, 2)
        
        t1_channel_attention = self.channel_conv1(channel_pool_t1)
        t2_channel_attention = self.channel_conv2(channel_pool_t2)
        channel_stack = torch.stack([t1_channel_attention, t2_channel_attention],
                                    dim=0)
        channel_stack = self.softmax(channel_stack).transpose(-1, -2).unsqueeze(-1)

        t1_spatial_avg_pool = torch.mean(t1, dim=1, keepdim=True)
        t1_spatial_max_pool = torch.max(t1, dim=1, keepdim=True)[0]
        t2_spatial_avg_pool = torch.mean(t2, dim=1, keepdim=True)
        t2_spatial_max_pool = torch.max(t2, dim=1, keepdim=True)[0]
        spatial_pool_t1 = torch.cat([t1_spatial_avg_pool, t2_spatial_max_pool], dim=1)
        spatial_pool_t2 = torch.cat([t2_spatial_avg_pool, t1_spatial_max_pool], dim=1)
        t1_spatial_attention = self.spatial_conv1(spatial_pool_t1)
        t2_spatial_attention = self.spatial_conv2(spatial_pool_t2)
        spatial_stack = torch.stack([t1_spatial_attention, t2_spatial_attention], dim=0)
        spatial_stack = self.softmax(spatial_stack)

        stack_attention = channel_stack + spatial_stack + 1
        fuse = stack_attention[0] * t1 + stack_attention[1] * t2


        return fuse

class Conv_BN_ReLU(nn.Module):

    def __init__(self, in_channel, out_channel, kernel, stride):
        super().__init__()
        self.conv_bn_relu = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=kernel,
                                                    padding=kernel // 2, bias=False, stride=stride),
                                          nn.BatchNorm2d(out_channel),
                                          nn.ReLU(inplace=True),
                                          )

    def forward(self, x):
        output = self.conv_bn_relu(x)

        return output


class CGSU(nn.Module):

    def __init__(self, in_channel):
        super().__init__()

        mid_channel = in_channel // 2

        self.conv1 = nn.Sequential(nn.Conv2d(mid_channel, mid_channel, kernel_size=3, padding=1,
                                             bias=False),
                                   nn.BatchNorm2d(mid_channel),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(p=ph.dropout_p)
                                   )

    def forward(self, x):
        x1, x2 = channel_split(x)
        x1 = self.conv1(x1)
        output = torch.cat([x1, x2], dim=1)

        return output


class CGSU_DOWN(nn.Module):

    def __init__(self, in_channel):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1,
                                             stride=2, bias=False),
                                   nn.BatchNorm2d(in_channel),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(p=ph.dropout_p)
                                   )
        self.conv_res = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        
        output1 = self.conv1(x)

        output2 = self.conv_res(x)

        output = torch.cat([output1, output2], dim=1)

        return output


class Changer_channel_exchange(nn.Module):

    def __init__(self, p=2):
        super().__init__()
        self.p = p

    def forward(self, x1, x2):
        N, C, H, W = x1.shape
        exchange_mask = torch.arange(C) % self.p == 0
        exchange_mask1 = exchange_mask.cuda().int().expand((N, C)).unsqueeze(-1).unsqueeze(-1)
        exchange_mask2 = 1 - exchange_mask1
        out_x1 = exchange_mask1 * x1 + exchange_mask2 * x2
        out_x2 = exchange_mask1 * x2 + exchange_mask2 * x1

        return out_x1, out_x2


class Encoder_Block(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()

        assert out_channel == in_channel * 2, 'the out_channel is not in_channel*2 in encoder block'
        self.conv1 = nn.Sequential(
            CGSU_DOWN(in_channel=in_channel),
            CGSU(in_channel=out_channel),
            CGSU(in_channel=out_channel)
        )
        self.conv3 = Conv_BN_ReLU(in_channel=out_channel, out_channel=out_channel, kernel=3, stride=1)

    def forward(self, x, module_name=None, img_name=None):
        x = self.conv1(x)
        x = self.conv3(x)
        output = x

        return output


class Decoder_Block(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()

        assert out_channel == in_channel // 2, 'the out_channel is not in_channel//2 in decoder block'
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.fuse = nn.Sequential(nn.Conv2d(in_channels=in_channel + out_channel, out_channels=out_channel,
                                            kernel_size=1, padding=0, bias=False),
                                  nn.BatchNorm2d(out_channel),
                                  nn.ReLU(inplace=True),
                                  CGSU(in_channel=out_channel),
                                  Conv_BN_ReLU(in_channel=out_channel, out_channel=out_channel, kernel=3, stride=1)
                                  )

    def forward(self, de, en):
        de = self.up(de)
        output = torch.cat([de, en], dim=1)
        output = self.fuse(output)

        return output

class Decoder_Edge_Block(nn.Module):

    def __init__(self, de_in_channel, en_in_channel, out_channel):
        super().__init__()

        assert out_channel == de_in_channel // 2, 'the out_channel is not in_channel//2 in decoder block'
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.fuse = nn.Sequential(nn.Conv2d(in_channels=de_in_channel + en_in_channel + en_in_channel, out_channels=out_channel,
                                            kernel_size=1, padding=0, bias=False),
                                  nn.BatchNorm2d(out_channel),
                                  nn.ReLU(inplace=True),
                                  CGSU(in_channel=out_channel),
                                  Conv_BN_ReLU(in_channel=out_channel, out_channel=out_channel, kernel=3, stride=1)
                                  )

    def forward(self, de, en1, en2):
        de = self.up(de)
        output = torch.cat([de, en1, en2], dim=1)
        output = self.fuse(output)

        return output


def kernel_size(in_channel):
    k = int((math.log2(in_channel) + 1) // 2)
    if k % 2 == 0:
        return k + 1
    else:
        return k


def channel_split(x):
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % 4 == 0)
    x = x.reshape(batchsize * num_channels // 2, 2, height * width)
    x = x.permute(1, 0, 2)
    x = x.reshape(2, -1, num_channels // 2, height, width)
    return x[0], x[1]


class Deformable_convolution(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()         
        self.dcn = DCN(in_channel, out_channel, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2)

    def forward(self, x):
        output = self.dcn(x.float())

        return output
    
class DCN_BN_ReLU(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.dcn_bn_relu = nn.Sequential(Deformable_convolution(in_channel, out_channel),
                                          nn.BatchNorm2d(out_channel),
                                          nn.ReLU(inplace=True),
                                          )

    def forward(self, x):
        output = self.dcn_bn_relu(x)

        return output

class Changer_spatial_exchange(nn.Module):
    
    def __init__(self, p=2):
        super().__init__()
        self.p = p

    def forward(self, x1, x2):
        N, c, h, w = x1.shape
        exchange_mask = torch.arange(w) % self.p == 0
 
        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[..., ~exchange_mask] = x1[..., ~exchange_mask]
        out_x2[..., ~exchange_mask] = x2[..., ~exchange_mask]
        out_x1[..., exchange_mask] = x2[..., exchange_mask]
        out_x2[..., exchange_mask] = x1[..., exchange_mask]
        
        return out_x1, out_x2

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels,in_channels//ratio,1,bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels//ratio, in_channels,1,bias=False)
        self.sigmod = nn.Sigmoid()
    def forward(self,x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmod(out)
