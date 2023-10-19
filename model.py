"""
Copyright (c) 2019 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
def get_wav(in_channels, pool=True):
    """wavelet decomposition using conv2d"""

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d

    LL = net(in_channels, in_channels,
             kernel_size=4, stride=2, padding=1, bias=False,
             groups=in_channels)
    LH = net(in_channels, in_channels,
             kernel_size=4, stride=2, padding=1, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels,
             kernel_size=4, stride=2, padding=1, bias=False,
             groups=in_channels)

    HH = net(in_channels, in_channels,
              kernel_size=4, stride=2, padding=1, bias=False,
              groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = (torch.tensor([
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]
    ]).unsqueeze(0).expand(in_channels, -1, -1, -1))/16

    LH.weight.data = (torch.tensor([
        [-1., -1., 1., 1.],
        [-1., -1., 1., 1.],
        [-1., -1., 1., 1.],
        [-1., -1., 1., 1.]
    ]).unsqueeze(0).expand(in_channels, -1, -1, -1))/8

    HL.weight.data = (torch.tensor([
        [1.,  1.,  1.,   1.],
        [1.,  1.,  1.,   1.],
        [-1., -1., -1., -1.],
        [-1., -1., -1., -1.]
    ]).unsqueeze(0).expand(in_channels, -1, -1, -1))/8

    HH.weight.data = (torch.tensor([
        # [-1., -1., -1., 0.],
        # [-1., -1., 0., 1.],
        # [-1., 0., 1., 1.],
        # [0., 1., 1., 1.]

        [1., 0., 0., -1.],
        [0., 1., -1., 0.],
        [0., -1., 1., 0.],
        [-1., 0., 0., 1.]
    ]).unsqueeze(0).expand(in_channels, -1, -1, -1)) / 4


    return LL, LH, HL, HH


class WavePool(nn.Module):
    def __init__(self, in_channels):
        super(WavePool, self).__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels)

    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)


class WaveUnpool(nn.Module):
    def __init__(self, in_channels, option_unpool='sum'):
        super(WaveUnpool, self).__init__()
        self.in_channels = in_channels
        self.option_unpool = option_unpool
        self.LL, self.LH, self.HL, self.HH = get_wav(self.in_channels, pool=False)

    def forward(self, LL, LH, HL, HH, original=None):
        if self.option_unpool == 'sum':
            shape_x1 = LL.size()
            shape_x2 = LH.size()
            left = 0
            right = 0
            top = 0
            bot = 0
            if shape_x1[3] != shape_x2[3]:
                lef_right = abs(shape_x2[3] - shape_x1[3])
                if lef_right % 2 is 0.0:
                    left = int(lef_right / 2)
                    right = int(lef_right / 2)
                else:
                    left = int(lef_right / 2)
                    right = int(lef_right - left)

            if shape_x1[2] != shape_x2[2]:
                top_bot = abs(shape_x1[2] - shape_x2[2])
                if top_bot % 2 is 0.0:
                    top = int(top_bot / 2)
                    bot = int(top_bot / 2)
                else:
                    top = int(top_bot / 2)
                    bot = int(top_bot - top)
            reflection_padding = [left, right, top, bot]
            reflection_pad = nn.ReflectionPad2d(reflection_padding)
            LL = reflection_pad(LL)
            return self.LL(LL) + self.LH(LH) + self.HL(HL) + self.HH(HH)
        elif self.option_unpool == 'cat5' and original is not None:
            return torch.cat([self.LL(LL), self.LH(LH), self.HL(HL), self.HH(HH), original], dim=1)
        else:
            raise NotImplementedError



class WaveEncoder(nn.Module):
    def __init__(self, option_unpool):
        super(WaveEncoder, self).__init__()
        self.option_unpool = option_unpool

        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(inplace=True)

        self.conv0 = nn.Conv2d(1, 1, 1, 1, 0)
        self.conv1_1 = nn.Conv2d(1, 64, 3, 1, 0)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
        self.pool1 = WavePool(64)

        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 0)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 0)
        self.pool2 = WavePool(128)

        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 0)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 0)
        self.pool3 = WavePool(256)

        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 0)

    def forward(self, x):
        skips = {}

        for level in [1, 2, 3, 4]:
            x = self.encode(x, skips, level)
        return x,skips

    def encode(self, x, skips, level):
        assert level in {1, 2, 3, 4}
        if self.option_unpool == 'sum':
            if level == 1:
                out = self.conv0(x)
                out = self.relu(self.conv1_1(self.pad(out)))
                out = self.relu(self.conv1_2(self.pad(out)))
                skips['conv1_2'] = out
                LL, LH, HL, HH = self.pool1(out)
                skips['pool1'] = [LH, HL, HH]
                return LL
            elif level == 2:
                out = self.relu(self.conv2_1(self.pad(x)))
                out = self.relu(self.conv2_2(self.pad(out)))
                skips['conv2_2'] = out
                LL, LH, HL, HH = self.pool2(out)
                skips['pool2'] = [LH, HL, HH]
                return LL
            elif level == 3:
                out = self.relu(self.conv3_1(self.pad(x)))
                out = self.relu(self.conv3_2(self.pad(out)))
                out = self.relu(self.conv3_3(self.pad(out)))
                out = self.relu(self.conv3_4(self.pad(out)))
                skips['conv3_4'] = out
                LL, LH, HL, HH = self.pool3(out)
                skips['pool3'] = [LH, HL, HH]
                return LL
            else:
                return self.relu(self.conv4_1(self.pad(x)))

        elif self.option_unpool == 'cat5':
            if level == 1:
                out = self.conv0(x)
                out = self.relu(self.conv1_1(self.pad(out)))
                return out

            elif level == 2:
                out = self.relu(self.conv1_2(self.pad(x)))
                skips['conv1_2'] = out
                LL, LH, HL, HH = self.pool1(out)
                skips['pool1'] = [LH, HL, HH]
                out = self.relu(self.conv2_1(self.pad(LL)))
                return out

            elif level == 3:
                out = self.relu(self.conv2_2(self.pad(x)))
                skips['conv2_2'] = out
                LL, LH, HL, HH = self.pool2(out)
                skips['pool2'] = [LH, HL, HH]
                out = self.relu(self.conv3_1(self.pad(LL)))
                return out

            else:
                out = self.relu(self.conv3_2(self.pad(x)))
                out = self.relu(self.conv3_3(self.pad(out)))
                out = self.relu(self.conv3_4(self.pad(out)))
                skips['conv3_4'] = out
                LL, LH, HL, HH = self.pool3(out)
                skips['pool3'] = [LH, HL, HH]
                out = self.relu(self.conv4_1(self.pad(LL)))
                return out
        else:
            raise NotImplementedError


class WaveDecoder(nn.Module):
    def __init__(self, option_unpool):
        super(WaveDecoder, self).__init__()
        self.option_unpool = option_unpool

        if option_unpool == 'sum':
            multiply_in = 1
        elif option_unpool == 'cat5':
            multiply_in = 5
        else:
            raise NotImplementedError

        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.conv4_1 = nn.Conv2d(512, 256, 3, 1, 0)
        self.sigmoid=nn.Sigmoid()

        self.recon_block3 = WaveUnpool(256, option_unpool)
        if option_unpool == 'sum':
            self.conv3_4 = nn.Conv2d(256*multiply_in, 256, 3, 1, 0)
        else:
            self.conv3_4_2 = nn.Conv2d(256*multiply_in, 256, 3, 1, 0)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_1 = nn.Conv2d(256, 128, 3, 1, 0)
        self.recon_block2 = WaveUnpool(128, option_unpool)
        if option_unpool == 'sum':
            self.conv2_2 = nn.Conv2d(128*multiply_in, 128, 3, 1, 0)
        else:
            self.conv2_2_2 = nn.Conv2d(128*multiply_in, 128, 3, 1, 0)
        self.conv2_1 = nn.Conv2d(128, 64, 3, 1, 0)

        self.recon_block1 = WaveUnpool(64, option_unpool)
        if option_unpool == 'sum':
            self.conv1_2 = nn.Conv2d(64*multiply_in, 64, 3, 1, 0)
        else:
            self.conv1_2_2 = nn.Conv2d(64*multiply_in, 64, 3, 1, 0)
        self.conv1_1 = nn.Conv2d(64, 1, 3, 1, 0)

    def forward(self, x, skips):
        for level in [4, 3, 2, 1]:
            x = self.decode(x, skips, level)
        return x

    def decode(self, x, skips, level):
        assert level in {4, 3, 2, 1}
        if level == 4:
            out = self.relu(self.conv4_1(self.pad(x)))
            LH, HL, HH = skips['pool3']
            original = skips['conv3_4'] if 'conv3_4' in skips.keys() else None
            out = self.recon_block3(out, LH, HL, HH, original)
            _conv3_4 = self.conv3_4 if self.option_unpool == 'sum' else self.conv3_4_2
            out = self.relu(_conv3_4(self.pad(out)))
            out = self.relu(self.conv3_3(self.pad(out)))
            return self.relu(self.conv3_2(self.pad(out)))
        elif level == 3:
            out = self.relu(self.conv3_1(self.pad(x)))
            LH, HL, HH = skips['pool2']
            original = skips['conv2_2'] if 'conv2_2' in skips.keys() else None
            out = self.recon_block2(out, LH, HL, HH, original)
            _conv2_2 = self.conv2_2 if self.option_unpool == 'sum' else self.conv2_2_2
            return self.relu(_conv2_2(self.pad(out)))
        elif level == 2:
            out = self.relu(self.conv2_1(self.pad(x)))
            LH, HL, HH = skips['pool1']
            original = skips['conv1_2'] if 'conv1_2' in skips.keys() else None
            out = self.recon_block1(out, LH, HL, HH, original)
            _conv1_2 = self.conv1_2 if self.option_unpool == 'sum' else self.conv1_2_2
            return self.relu(_conv1_2(self.pad(out)))
        else:
            return [self.relu(self.conv1_1(self.pad(x)))]


# class Reconstruct(nn.Module):
#     def __init__(self):
#         super(Reconstruct, self).__init__()
#         self.pad = nn.ReflectionPad2d(1)
#         self.relu = nn.LeakyReLU(inplace=True)
#
#         self.conv1 = nn.Conv2d(1, 1, 1, 1, 0)
#         self.conv1 = nn.Conv2d(1, 1, 1, 1, 0)
#         self.conv1 = nn.Conv2d(1, 1, 1, 1, 0)
#         self.conv1 = nn.Conv2d(1, 1, 1, 1, 0)
#         self.conv1 = nn.Conv2d(1, 1, 1, 1, 0)
#
#
#     def deco(self, x):
#         activate = nn.LeakyReLU(inplace=True)
#
#
# class WT(nn.Module):
#     def __init__(self):
#         super(WT, self).__init__()
#         self.encoder = WaveEncoder(option_unpool='sum')
#         self.decoder = WaveDecoder(option_unpool='sum')
#         self.re = Reconstruct()
#     def encode(self, x, skips, level):
#         return self.encoder.encode(x, skips, level)
#
#     def decode(self, x, skips, level):
#         return self.decoder.decode(x, skips, level)
#
#     def reconstruct(self, x):
#         return self.re.deco(x)
#
#     def get_all_feature(self, ir_image,vis_y_image):
#         skips = {}
#         content_skips={}
#         style_skips={}
#         feats={'encoder': {}, 'decoder': {}}
#         content_feats = {'encoder': {}, 'decoder': {}}
#         style_feats = {'encoder': {}, 'decoder': {}}
#
#         for level in [1, 2, 3, 4]:
#             ir_image = self.encode(ir_image, content_skips, level)
#             vis_y_image = self.encode(vis_y_image, style_skips, level)
#
#
#
#             content_feats['encoder'][level] = ir_image
#             style_feats['encoder'][level] = vis_y_image
#
#         return skips,content_skips,style_skips,content_feats,style_feats
#
#     def transfer(self, vis_y_image, ir_image):
#         # style=vis, conten=ir
#         # content_feat, content_skips = self.get_all_feature(ir_image)
#         skips, content_skips, style_skips, content_feats, style_feats = self.get_all_feature(ir_image, vis_y_image)
#         # style_feats, style_skips = self.get_all_feature(vis_y_image,level_features)
#
#         fusion_feat = {'encoder': {}, 'decoder': {}}
#         fusion_skips = {}
#
#         wct2_dec_level = [1, 2, 3]
#         wct_skips=[1,2,3]
#         encode = ['', 'conv1_2', 'conv2_2', 'conv3_4', '']
#         wct2_skip_level = ['pool1', 'pool2', 'pool3']
#         fusion_skips['pool1'] = [0, 0, 0]
#         fusion_skips['pool2'] = [0, 0, 0]
#         fusion_skips['pool3'] = [0, 0, 0]
#         fusion = torch.tensor(1)
#         for level in [1, 2, 3, 4]:
#             # skip=wct2_skip_level[level]
#
#             fusion_feat['encoder'][level] = spatial_fusion(torch.abs(style_feats['encoder'][level]),
#                                                                     torch.abs(content_feats['encoder'][level]))
#             # if level in wct2_dec_level:
#             #     fusion_skips[encode[level]] = (torch.abs(content_skips[encode[level]]) + torch.abs(style_skips[encode[level]]))/2
#
#             if level in wct_skips:
#                 skip_level=wct2_skip_level[level-1]
#             # for skip_level in wct2_skip_level:
#                 for component in [0, 1, 2]:  # component: [LH, HL, HH]
#                     fusion_skips[skip_level][component] =(content_skips[skip_level][component] +
#                                                            style_skips[skip_level][component])
#                     # fusion = torch.sum(fusion_skips[skip_level][component], dim=0)
#                     # fusion = torch.sum(fusion, dim=0)
#                     # fusion = fusion / fusion_skips[skip_level][component].size()[1]
#                     # # rgb_fused_image = YCrCb2RGB(fusion, cb[0], cr[0])
#                     # rgb_fused_image = transforms.ToPILImage()(fusion)
#                     # ss=0
#                     # rgb_fused_image.save('weight/1.png')
#             # imge_feat['encoder'][level]=(content_feat + style_feats['encoder'][level])/2
#
#         for level in [4, 3, 2, 1]:
#             # if level in style_feats['decoder'] and level in wct2_dec_level:
#             # content_feat = feature_wct(content_feat, style_feats['encoder'][level])  # WCT提取特征
#             # content_feat = attention_fusion_weight(style_feats['decoder'][level], content_feat)
#             # content_feat = (style_feats['decoder'][level] + content_feat)/2
#             # imge_feat['decoder'][level] = (content_feat + style_feats['decoder'][level])/2
#             if level == 4:
#                 fusion = self.decode(fusion_feat['encoder'][level], fusion_skips, level)
#
#                 # fusion=torch.sum(fusion,dim=0)
#                 # fusion = torch.sum(fusion, dim=0)
#                 # fusion=fusion/256
#                 # # rgb_fused_image = YCrCb2RGB(fusion, cb[0], cr[0])
#                 # rgb_fused_image = transforms.ToPILImage()(fusion)
#                 # rgb_fused_image.save('weight/1.png')
#             if level == 3:
#                 # fusion_3 = self.deco3(fusion_feat['encoder'][level])
#                 # fusion = (fusion_3 + fusion)/2
#                 fusion = self.decode(fusion, fusion_skips, level)
#
#             if level == 2:
#                 # fusion_2 = self.deco2(fusion_feat['encoder'][level])
#                 # fusion = fusion * fusion_2 + fusion
#                 fusion = self.decode(fusion, fusion_skips, level)
#
#             if level == 1:
#                 # fusion_1 = self.deco1(fusion_feat['encoder'][level])
#                 # fusion = fusion * fusion_1 + fusion
#                 fusion = self.decode(fusion, fusion_skips, level)
#         return fusion
# class PIAFusion(nn.Module):
#     def __init__(self):
#         super(PIAFusion, self).__init__()
#         # self.encoder = Encoder()
#         # self.decoder = Decoder()
#         self.Tran = WT()
#
#     def forward(self, y_vi_image, ir_image):
#         # vi_encoder_out, ir_encoder_out = self.encoder(y_vi_image, ir_image)#特征提取器
#         # encoder_out = Fusion(vi_encoder_out, ir_encoder_out, vis_image)#torch.cat
#         fea_WT=self.Tran.transfer(y_vi_image, ir_image)
#         # encoder_out = Fusion_layer(vi_encoder_out, ir_encoder_out, fea_WT)
#         # decoder_weight=Fusion_weight(vis_image)
#         # 将从红外和可见光图像中提取的深层特征连接起来，作为图像重建器的输入
#         # fused_image = self.decoder(encoder_out)#图像重建器
#         return fea_WT
#
# from loader.msrs_data import MSRS_data
# from torch import nn
# import torch
# from common import gradient, clamp, attention_fusion_weight, spatial_fusion
#
# data='/data/Disk_A/yongbiao/USERPROG/PIAFusion/test_data/MSRS/'
# # data='/data/Disk_B/datasets/imagefusion/KAIST-database'
# train_dataset = MSRS_data(data)
# train_loader = DataLoader(
#         train_dataset, batch_size=1, shuffle=True,
#         num_workers=1, pin_memory=True)
# # train_tqdm = tqdm(train_loader, total=len(train_loader))
# model = PIAFusion()
# for vis_image, vis_y_image, _, _, inf_image, name,_ in train_loader:
#     fused_image = model(vis_y_image,inf_image)
# # 强制约束范围在[0,1], 以免越界值使得最终生成的图像产生异常斑点
#     fused_image = clamp(fused_image[0])
#     print(fused_image.shape, name[0])