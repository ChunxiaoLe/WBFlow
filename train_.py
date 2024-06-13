import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from PIL import Image
from PIL import ImageFile
from torchvision import transforms
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
import time
from PIL import Image, ImageOps, ImageEnhance, ImageFile
from glow_wb import calc_mean_std

import net
from sampler import InfiniteSamplerWrapper
from dataset_camera import BasicDataset_
from dataset_msa import BasicDataset
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

from math import log, sqrt, pi

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated


class mae_loss():

    @staticmethod
    def compute(output, target):
        loss = torch.sum(torch.abs(output - target)) / output.size(0)
        return loss


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


def val_camera(glow, test_loader, name, camera_name, i, tf_writer):
    glow.eval()
    loss = 0.
    count = 0
    pred = []

    iter_num = len(test_loader)
    for batch in test_loader:
        # for i, (x_t, y_t, camera_t) in enumerate(test_loader):
        x_t = batch['image']
        y_t = batch['gt-AWB']
        camera_t = batch['label']
        x_t = x_t.float().to(device)
        y_t = y_t.float().to(device)
        camera_t = camera_t.float().to(device)
        loss_this, pred_this = forward_aux_loss_camera(glow, x_t, y_t,camera_t)
        loss = loss_this.cpu().numpy() + loss

        if (count + 1) % 5 == 0:
            # mapname = camera_name + '_' + str(i)
            tf_writer.add_images(f'{name}/{camera_name}_{str(count)}/', pred_this.cpu().permute(0, 2, 3, 1).numpy()[0],
                                 i, dataformats="HWC")

        count = count + 1
    loss_ = loss / count
    # print(f'{camera_name}---- %d Loss = %.6f ---' % (iter_num, loss_))
    tf_writer.add_scalar(f'{name}/{camera_name}', loss_.item(), i)
    return loss_


def forward_aux_loss_camera(glow, x, y,cam):
    x = x.to(device)
    y = y.to(device)
    cam = cam.to(device)

    z_c,z_cam = glow(x,cam, forward=True)
    # reverse
    pred, pred_cam = glow(z_c,z_cam, forward=False)

    loss = mae_loss.compute(pred, y)
    return loss, pred


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save(glow, optimizer, filename, total_it):
    state_dict = glow.module.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].to(torch.device('cpu'))

    state = {'iter': total_it, 'state_dict': state_dict, 'optimizer': optimizer.state_dict()}
    torch.save(state, filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--content_dir', type=str, default='/dataset/colorconstancy/set1_all1/',
                        help='Directory path to a batch of content images')
    parser.add_argument('--style_dir', type=str,
                        default='/dataset/colorconstancy/Chinese-Landscape-Painting-Dataset-main/',
                        help='Directory path to a batch of style images')

    # training options

    parser.add_argument('--log_dir', default='./logs',
                        help='Directory to save the log')
    parser.add_argument('--lr', type=float, default=1e-4) # 1e-4
    parser.add_argument('--lr_decay', type=float, default=5e-5) # 1e-5
    "160000, 320000"
    parser.add_argument('--max_iter', type=int, default=320000)
    parser.add_argument('--batch_size', type=int, default=4)

    parser.add_argument('--mse_weight', type=float, default=0)
    parser.add_argument('--style_weight', type=float, default=1)
    parser.add_argument('--content_weight', type=float, default=0.1)

    # save options
    parser.add_argument('--n_threads', type=int, default=8)
    "200"
    parser.add_argument('--print_interval', type=int, default=200)
    "5000"
    parser.add_argument('--save_model_interval', type=int, default=5000)
    parser.add_argument('--start_iter', type=int, default=0, help='starting iteration')
    "glow_clp.pth"
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # glow parameters
    parser.add_argument('--n_flow', default=8, type=int, help='number of flows in each block')  # 32
    parser.add_argument('--n_block', default=2, type=int, help='number of blocks')  # 4
    parser.add_argument('--no_lu', action='store_true', help='use plain convolution instead of LU decomposed version')
    parser.add_argument('--affine', default=False, type=bool, help='use affine coupling instead of additive')
    parser.add_argument('--operator', type=str, default='wb',
                        help='style feature transfer operator')
    "GLOW_WB_noconv_feat_nosigma GLOW_WB_CONS_1"
    parser.add_argument('--pre_name', default='GLOW_WB_CAM', type=str, help='')
    parser.add_argument('--name', default='GLOW_WB_CAM', type=str, help='')
    parser.add_argument('--save_dir', default='./output', type=str, help='')

    args = parser.parse_args()

    device = torch.device('cuda:0')


    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    print(f"--------Fewshot--{args.name}----------")

    # l1 loss
    model_dir = '%s/model/%s' % (args.save_dir, args.name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    tf_dir = '%s/log/%s' % (args.save_dir, args.name)
    if not os.path.exists(tf_dir):
        os.makedirs(tf_dir)

    tf_writer = SummaryWriter(log_dir=tf_dir)
    # mseloss = nn.MSELoss()

    # ---------------------------glow------------------------------
    from glow_wb import Glow,Camera_Glow

    glow_single = Camera_Glow(3, 12, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu)

    # print("--------loading checkpoint----------")
    # print("=> loading checkpoint '{}'".format(args.resume))
    # checkpoint = torch.load(args.resume)
    # # args.start_iter = checkpoint['iter']
    # glow_single.load_state_dict(checkpoint['state_dict'])

    "加载backbone"
    # model_dict = glow_single.state_dict()
    # pretrained_dict = torch.load(args.resume)['state_dict']
    #
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #
    # model_dict.update(pretrained_dict)
    # glow_single.load_state_dict(model_dict)

    glow_single = glow_single.to(device)

    # for k, v in glow_single.named_parameters():
    #     print('{}: {}'.format(k, v.requires_grad))

    "冻结层"
    ## 冻参数
    # print('--------fixed feature projection--------')
    # frozen_layers = [glow_single.blocks,glow_single.wb.conv_std,glow_single.wb.conv_mean]
    #
    # for layer in frozen_layers:
    #     for name, value in layer.named_parameters():
    #         value.requires_grad = False
    #
    # for k, v in glow_single.named_parameters():
    #     print('{}: {}'.format(k, v.requires_grad))

    glow = nn.DataParallel(glow_single, device_ids=[0])
    glow.train()

    patch_size = 256
    K = 20

    # -------------------------------------------------------------
    "下载数据：input, GT"
    "训练10次求平均，因为训练数据是随机选的"
    # train_dataset = BasicDataset_(name='cam8_ct_5', patch_size=patch_size, K=K, patch_num_per_image=1, type='train')
    # # train_iter = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    # train_iter = iter(DataLoader(
    #     train_dataset, batch_size=args.batch_size,
    #     sampler=InfiniteSamplerWrapper(train_dataset),
    #     num_workers=args.n_threads))

    train_dataset = BasicDataset(name='all_5000_1', patch_size=patch_size, patch_num_per_image=1, type='train')
    # train_iter = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    train_iter = iter(DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(train_dataset),
        num_workers=args.n_threads))

    # train_ = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    "can replace the val dataset as set1-val or others"
    val_sam_dataset = BasicDataset(name='SamsungNX2000_ct_5', patch_size=patch_size, patch_num_per_image=1, type='val')
    val_sam = DataLoader(val_sam_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    val_pan_dataset = BasicDataset(name='PanasonicGX1_ct_5', patch_size=patch_size, patch_num_per_image=1, type='val')
    val_pan = DataLoader(val_pan_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    val_sony_dataset = BasicDataset(name='SonyA57_ct_5', patch_size=patch_size, patch_num_per_image=1, type='val')
    val_sony = DataLoader(val_sony_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    val_oly_dataset = BasicDataset(name='OlympusEPL6_ct_5', patch_size=patch_size, patch_num_per_image=1, type='val')
    val_oly = DataLoader(val_oly_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    "optimizer"
    optimizer = torch.optim.Adam(glow.module.parameters(), lr=args.lr)
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         optimizer.load_state_dict(checkpoint['optimizer'])

    # Time = time.time()
    # -----------------------training------------------------
    "training_epoch"
    # log_ = []
    max_acc = 100000
    for total_it in range(args.start_iter, args.max_iter):
        glow.train()
        adjust_learning_rate(optimizer, iteration_count=total_it)
        train_images = next(train_iter)
        # content_images = next(content_iter).to(device)
        # style_images = next(style_iter).to(device)
        ".to(device)"
        input_img = train_images['image'].float().to(device)
        gt_img = train_images['gt-AWB'].float().to(device)
        cam = train_images['label'].float().to(device)

        # glow forward: real -> z_real, style -> z_style
        if total_it == args.start_iter:
            with torch.no_grad():
                _,_ = glow.module(input_img,cam, forward=True)
                continue

        # (log_p, logdet, z_outs) = glow()
        z_c,z_cam = glow(input_img,cam, forward=True)
        # reverse
        correct_img, correct_cam = glow(z_c,z_cam, forward=False)

        "image loss"
        loss_img = mae_loss.compute(correct_img, gt_img)
        "cam loss"
        loss_cam = mae_loss.compute(correct_cam,cam)


        loss = loss_img + loss_cam

        # optimizer update
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(glow.module.parameters(), 5)
        optimizer.step()

        print(f'{args.name}---- %d Loss_img = %.6f --- Loss_cam = %.6f---' % (total_it, loss_img, loss_cam))
        tf_writer.add_scalar(f'{args.name}/loss_img', loss_img.item(), total_it)
        tf_writer.add_scalar(f'{args.name}/loss_cam', loss_cam.item(), total_it)

        # update loss log
        # log_.append(loss_mae.item())

        "save image + val"
        if total_it % args.print_interval == 0:
            glow.eval()
            kk = total_it
            with torch.no_grad():
                "can replace the val dataset as set1-val or others"
                acc_sam = val_camera(glow, val_sam, args.name, 'SamsungNX2000', kk, tf_writer)
                print(f'{args.name}-SamsungNX2000- %d Loss = %.3f ---' % (total_it, acc_sam))
                acc_pan = val_camera(glow, val_pan, args.name, 'PanasonicGX1', kk, tf_writer)
                print(f'{args.name}-PanasonicGX1- %d Loss = %.3f ---' % (total_it, acc_pan))
                acc_oly = val_camera(glow, val_oly, args.name, 'OlympusEPL6', kk, tf_writer)
                print(f'{args.name}-OlympusEPL6- %d Loss = %.3f ---' % (total_it, acc_oly))
                acc_sony = val_camera(glow, val_sony, args.name, 'SonyA57', kk, tf_writer)
                print(f'{args.name}-SonyA57- %d Loss = %.3f ---' % (total_it, acc_sony))
                acc = (acc_sam + acc_pan + acc_oly + acc_sony) / 4
                print(f'{args.name}-ALL-CAM- %d Loss = %.3f ---' % (total_it, acc))

                if acc < max_acc:
                    print("best model! save...")
                    max_acc = acc
                    outfile = os.path.join(model_dir, 'best_model.tar')
                    save(glow, optimizer, outfile, total_it)

                else:
                    print('GG!! best accuracy {:f}'.format(max_acc))

            glow.train()

        if (total_it + 1) % args.save_model_interval == 0 or (total_it + 1) == args.max_iter:
            outfile = os.path.join(model_dir, '{:d}.tar'.format(total_it + 1))
            save(glow, optimizer, outfile, total_it)
            print('Done')



