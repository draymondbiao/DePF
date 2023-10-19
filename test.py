"""测试融合网络"""
import argparse
import os
import random

import numpy as np

from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from loader.msrs_data import MSRS_data
from loader.common import YCrCb2RGB, RGB2YCrCb, clamp, spatial_fusion, attention_fusion_weight,gradient
from model import WaveEncoder, WaveDecoder
from PIL import Image
from torch.utils import data
from torchvision import transforms
import loader.fusion_strategy
import torch
from time import time

torch.cuda.set_device(0)

def CMDAF(vi_feature, ir_feature):#由卷积层从红外和可见光图像中提取的特征
    sigmoid = nn.Sigmoid()
    gap = nn.AdaptiveAvgPool2d(1)
    batch_size, channels, _, _ = vi_feature.size()

    sub_vi_ir = vi_feature - ir_feature
    # print(gap(sub_vi_ir))
    vi_ir_div = sub_vi_ir * sigmoid((sub_vi_ir))

    sub_ir_vi = ir_feature - vi_feature
    ir_vi_div = sub_ir_vi * sigmoid((sub_ir_vi))

    # 特征加上各自的带有简易通道注意力机制的互补特征
    vi_feature += ir_vi_div
    ir_feature += vi_ir_div

    return vi_feature, ir_feature
def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)



class WCT2:
    def __init__(self, option_unpool='sum', verbose=True):

        self.verbose = verbose

        self.encoder = WaveEncoder(option_unpool).cuda()
        self.decoder = WaveDecoder(option_unpool).cuda()
        self.encoder.load_state_dict(torch.load('./models/Final_Encoder_epoch_4.model',
            map_location=torch.device('cpu')))
        self.decoder.load_state_dict(torch.load('./models/Final_Decoder_epoch_4.model',
            map_location=torch.device('cpu')))
        # print(self.encoder)
        total = sum([params.nelement() for params in self.encoder.parameters()])
        print("Number of params Encoder: {%.2f M}" % (total / 1e6))

        total = sum([params.nelement() for params in self.decoder.parameters()])
        print("Number of params Encoder: {%.2f M}" % (total / 1e6))
        self.encoder.eval()
        self.decoder.eval()



    def print_(self, msg):
        if self.verbose:
            print(msg)

    def encode(self, x, skips, level):
        return self.encoder.encode(x, skips, level)

    def decode(self, x, skips, level):
        return self.decoder.decode(x, skips, level)



    def get_all_feature(self, ir_image,vis_y_image):
        skips = {}
        ir_skips={}
        vis_skips={}
        feats={'encoder': {}, 'decoder': {}}
        ir_feats = {'encoder': {}, 'decoder': {}}
        vis_feats = {'encoder': {}, 'decoder': {}}

        for level in [1, 2, 3, 4]:
            ir_image = self.encode(ir_image, ir_skips, level)
            vis_y_image = self.encode(vis_y_image, vis_skips, level)

            fusion = torch.sum(vis_y_image,dim=0)
            fusion = torch.sum(fusion, dim=0)
            fusion = fusion/vis_y_image.size()[1]
            # rgb_fused_image = YCrCb2RGB(fusion, cb[0], cr[0])
            rgb_fused_image = transforms.ToPILImage()(fusion)
            # rgb_fused_image.save('weight/1.png')


            ir_feats['encoder'][level] = ir_image
            vis_feats['encoder'][level] = vis_y_image

        return skips,ir_skips,vis_skips,ir_feats,vis_feats

    def transfer(self, vis_y_image, ir_image):
        # vis=vis, conten=ir
        # ir_feat, ir_skips = self.get_all_feature(ir_image)
        skips, ir_skips, vis_skips, ir_feats, vis_feats = self.get_all_feature(ir_image, vis_y_image)
        # vis_feats, vis_skips = self.get_all_feature(vis_y_image,level_features)

        fusion_feat = {'encoder': {}, 'decoder': {}}
        fusion_skips = {}

        wct2_dec_level = [1, 2, 3]
        wct_skips=[1,2,3]
        encode = ['', 'conv1_2', 'conv2_2', 'conv3_4', '']
        wct2_skip_level = ['pool1', 'pool2', 'pool3']
        fusion_skips['pool1'] = [0, 0, 0]
        fusion_skips['pool2'] = [0, 0, 0]
        fusion_skips['pool3'] = [0, 0, 0]
        fusion = torch.tensor(1)
        for level in [1, 2, 3, 4]:
            # skip=wct2_skip_level[level]

            fusion_feat['encoder'][level] = spatial_fusion(torch.abs(vis_feats['encoder'][level]),
                                                                    torch.abs(ir_feats['encoder'][level]))

            if level in wct_skips:
                skip_level = wct2_skip_level[level-1]
            # for skip_level in wct2_skip_level:
                for component in [0, 1, 2]:  # component: [LH, HL, HH]
                    fusion_skips[skip_level][component] = (ir_skips[skip_level][component] +
                                                           vis_skips[skip_level][component])
                    # fusion = torch.sum(fusion_skips[skip_level][component], dim=0)
                    # fusion = torch.sum(fusion, dim=0)
                    # fusion = fusion / fusion_skips[skip_level][component].size()[1]
                    # rgb_fused_image = transforms.ToPILImage()(fusion)
                    # rgb_fused_image.save('weight/1.png')

        for level in [4, 3, 2, 1]:
            if level == 4:
                fusion = self.decode(fusion_feat['encoder'][level], fusion_skips, level)

            if level == 3:
                fusion = self.decode(fusion, fusion_skips, level)

            if level == 2:
                fusion = self.decode(fusion, fusion_skips, level)

            if level == 1:
                fusion = self.decode(fusion, fusion_skips, level)
        return fusion


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch PIAFusion')
    #/data/Disk_A/yongbiao/USERPROG/PIAFusion/test_data/MSRS-main/
    parser.add_argument('--dataset_path', metavar='DIR', default='/data/Disk_A/yongbiao/USERPROG/PIAFusion/test_data/TNO2/',
                        help='path to dataset (default: imagenet)')# 测试数据存放位置
    parser.add_argument('-a', '--arch', metavar='ARCH', default='fusion_model',
                        choices=['fusion_model'])
    parser.add_argument('--save_path', default='results/DePF_TNO2')# 融合结果存放位置

    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--cuda', default=True, type=bool,
                        help='use GPU or not.')

    args = parser.parse_args()

    # init_seeds(args.seed)


    test_dataset = MSRS_data(args.dataset_path)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)




    # 如果是融合网络
    if args.arch == 'fusion_model':
        test_tqdm = tqdm(test_loader, total=len(test_loader))
        list_no=[]
        with torch.no_grad():
            wct2 = WCT2()
            sum=0

            for vis_image, vis_y_image, cb, cr, inf_image, name,size in test_tqdm:
                vis_image = vis_image.cuda()
                vis_y_image = vis_y_image.cuda()
                cb = cb.cuda()
                cr = cr.cuda()
                inf_image = inf_image.cuda()

                # try:
                start_time = time()
                fused_image = wct2.transfer(vis_y_image,inf_image)
                end_time = time()
                elapsed = end_time - start_time
                sum+=elapsed
                # print(name[0],elapsed)
                fused_image = clamp(fused_image[0][0])
                # 格式转换，因为tensor不能直接保存成图片

                # fused_image=fused_image.reshape([])
                fused_image = YCrCb2RGB(fused_image, cb[0], cr[0])
                fused_image = transforms.ToPILImage()(fused_image)
                # fused_image=fused_image.resize((size[0],size[1]),Image.BILINEAR)
                fused_image.save(f'{args.save_path}/{name[0]}')
            print(sum)
            #     except:
            #         list_no.append(name[0].split('.')[0])
            # print(list_no)