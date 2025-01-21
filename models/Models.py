import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from models.eaddn_parts import (Conv_BN_ReLU, DCN_BN_ReLU, CGSU, Encoder_Block, BITCBAM, Decoder_Block, Decoder_Edge_Block, ChannelAttention,
                               Changer_channel_exchange, Changer_spatial_exchange)


class EADDN(nn.Module):

    def __init__(self):
        super().__init__()

        channel_list = [32, 64, 128, 256, 512]
        # encoder
        self.en_block1 = nn.Sequential(Conv_BN_ReLU(in_channel=3, out_channel=channel_list[0], kernel=3, stride=1),
                                       DCN_BN_ReLU(in_channel=channel_list[0], out_channel=channel_list[0]),
                                       CGSU(in_channel=channel_list[0]),
                                       CGSU(in_channel=channel_list[0]),
                                       )
        self.en_block2 = Encoder_Block(in_channel=channel_list[0], out_channel=channel_list[1])
        self.en_block3 = Encoder_Block(in_channel=channel_list[1], out_channel=channel_list[2])
        self.en_block4 = Encoder_Block(in_channel=channel_list[2], out_channel=channel_list[3])
        self.en_block5 = Encoder_Block(in_channel=channel_list[3], out_channel=channel_list[4])

        self.spatial_exchange2 = Changer_spatial_exchange()
        self.channel_exchange4 = Changer_channel_exchange()

        # decoder
        self.de_block1_1 = Decoder_Block(in_channel=channel_list[4], out_channel=channel_list[3])
        self.de_block2_1 = Decoder_Block(in_channel=channel_list[3], out_channel=channel_list[2])
        self.de_block3_1 = Decoder_Block(in_channel=channel_list[2], out_channel=channel_list[1])
        
        self.de_block1_2 = Decoder_Block(in_channel=channel_list[4], out_channel=channel_list[3])
        self.de_block2_2 = Decoder_Block(in_channel=channel_list[3], out_channel=channel_list[2])
        self.de_block3_2 = Decoder_Block(in_channel=channel_list[2], out_channel=channel_list[1])

        self.upsample_x2_1 = nn.Sequential(
            nn.Conv2d(channel_list[1] + channel_list[0], channel_list[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel_list[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_list[0], 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

        self.upsample_x2_2 = nn.Sequential(
            nn.Conv2d(channel_list[1] + channel_list[0], channel_list[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel_list[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_list[0], 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

        self.conv_out_change_1 = nn.Conv2d(8, 1, kernel_size=7, stride=1, padding=3)
        self.conv_out_change_2 = nn.Conv2d(8, 1, kernel_size=7, stride=1, padding=3)

        # 边缘分支：
        self.dpfa3 = DPFA(in_channel=channel_list[3])
        self.edge_de_block2 = Decoder_Edge_Block(de_in_channel = channel_list[2], en_in_channel = channel_list[1], out_channel = channel_list[1])
        self.edge_de_block1 = Decoder_Edge_Block(de_in_channel = channel_list[1], en_in_channel = channel_list[0], out_channel = channel_list[0])
        self.edge_conv = nn.Sequential(
            nn.Conv2d(channel_list[0], channel_list[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel_list[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_list[0], 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        self.conv_out_edge = nn.Conv2d(8, 1, kernel_size=7, stride=1, padding=3)
        self.conv_edge_skip = Conv_BN_ReLU(in_channel=channel_list[0], out_channel=channel_list[0], kernel=3, stride=1)

        # 总和分支
        self.dpfa_sum4 = DPFA(in_channel=channel_list[3])
        self.dpfa_sum3 = DPFA(in_channel=channel_list[2])
        self.dpfa_sum2 = DPFA(in_channel=channel_list[1])
        self.dpfa_sum1 = DPFA(in_channel=8)

        self.up_sum4 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.conv_bn_relu_sum4 = Conv_BN_ReLU(in_channel=channel_list[3], out_channel=8, kernel=3, stride=1)
        self.up_sum3 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.conv_bn_relu_sum3 = Conv_BN_ReLU(in_channel=channel_list[2], out_channel=8, kernel=3, stride=1)
        self.up_sum2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv_bn_relu_sum2 = Conv_BN_ReLU(in_channel=channel_list[1], out_channel=8, kernel=3, stride=1)
        self.conv_bn_relu_sum1 = Conv_BN_ReLU(in_channel=8, out_channel=8, kernel=3, stride=1)
        self.ca = ChannelAttention(8 * 4, ratio=16)
        self.ca1 = ChannelAttention(8, ratio=16 // 4)

        self.conv_final = nn.Conv2d(8 * 4, 1, kernel_size=1)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, t1, t2, img_name=None):
    
        # encoder
        t1_1 = self.en_block1(t1)  # 32x256x256
        t2_1 = self.en_block1(t2)

        t1_2 = self.en_block2(t1_1)  # 64x128x128
        t2_2 = self.en_block2(t2_1)
        t1_2, t2_2 = self.spatial_exchange2(t1_2, t2_2)

        t1_3 = self.en_block3(t1_2)  # 128x64x64
        t2_3 = self.en_block3(t2_2)

        t1_4 = self.en_block4(t1_3)  # 256x32x32
        t2_4 = self.en_block4(t2_3)
        t1_4, t2_4 = self.channel_exchange4(t1_4, t2_4)

        t1_5 = self.en_block5(t1_4)  # 512x16x16
        t2_5 = self.en_block5(t2_4)
        
        # 边缘分支:
        edg1_3 = t1_3
        edg2_3 = t2_3
        edg3 = self.dpfa3(edg1_3, edg2_3)  # 128x64x64

        edg2 = self.edge_de_block2(edg3, t1_2, t2_2)  # 64x128x128

        edg1 = self.edge_de_block1(edg2, t1_1, t2_1)  # 32x256x256
        edg = self.edge_conv(edg1)
        edg = self.conv_out_edge(edg)
        edg_skip = self.conv_edge_skip(edg1)
        
        # decoder
        de1_5 = t1_5
        de2_5 = t2_5

        de1_4 = self.de_block1_1(de1_5, t1_4)
        de2_4 = self.de_block1_2(de2_5, t2_4)

        de1_3 = self.de_block2_1(de1_4, t1_3)
        de2_3 = self.de_block2_2(de2_4, t2_3)

        de1_2 = self.de_block3_1(de1_3, t1_2)
        de2_2 = self.de_block3_2(de2_3, t2_2)

        de1_2_skip = torch.cat([de1_2, F.interpolate(edg_skip, t1_2.size()[2:], mode='bilinear', align_corners=True)], dim=1)
        de2_2_skip = torch.cat([de2_2, F.interpolate(edg_skip, t2_2.size()[2:], mode='bilinear', align_corners=True)], dim=1)

        seg_out1_1 = self.upsample_x2_1(de1_2_skip)
        seg_out2_1 = self.upsample_x2_2(de2_2_skip)
        seg_out1 = self.conv_out_change_1(seg_out1_1)
        seg_out2 = self.conv_out_change_2(seg_out2_1)

        # decoder_zong
        sum_de4 = self.dpfa_sum4(de1_4, de2_4)

        sum_de3 = self.dpfa_sum3(de1_3, de2_3)

        sum_de2 = self.dpfa_sum2(de1_2, de2_2)

        sum_de1 = self.dpfa_sum1(seg_out1_1, seg_out2_1)

        sum_4 = self.conv_bn_relu_sum4(self.up_sum4(sum_de4))
        sum_3 = self.conv_bn_relu_sum3(self.up_sum3(sum_de3))
        sum_2 = self.conv_bn_relu_sum2(self.up_sum2(sum_de2))
        sum_1 = self.conv_bn_relu_sum1(sum_de1)

        out = torch.cat([sum_4, sum_3, sum_2, sum_1], 1)

        intra = torch.sum(torch.stack((sum_4, sum_3, sum_2, sum_1)), dim=0)
        ca1 = self.ca1(intra)
        out = self.ca(out) * (out + ca1.repeat(1, 4, 1, 1))
        sum_out = self.conv_final(out)


        return sum_out, seg_out1, seg_out2, edg     #seg_out1是change map, seg_out2是change map, edg是edge.
