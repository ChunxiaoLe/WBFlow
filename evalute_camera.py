import argparse
import os
import torch
from evaluation.evaluate_cc import evaluate_cc
import seaborn as sns
sns.set()
import numpy as np
from glow_wb import Camera_Glow_norev_re
from dataset_msa import BasicDataset
from torch.utils.data import DataLoader



class Evaluator:
    def __init__(self, args, CAM, device):
        self.args = args
        self.cam_list = ["Canon1DsMkIII", "Canon600D", "FujifilmXM1", "NikonD40", "NikonD5200", "OlympusEPL6",
                         "PanasonicGX1",
                         "SamsungNX2000", "SonyA57"]
        # self.config = config
        self.device = device
        self.global_step = 0
        self.bs = 1
        self.CAM = CAM

        # networks OlympusEPL6
        self.type = args.train_cam
        print(self.type)

        self.glow = Camera_Glow_norev_re(3, 15, args.n_flow, args.n_block, affine=args.affine,
                                           conv_lu=not args.no_lu).to(device)


        print("--------loading checkpoint----------")
        checkpoint = torch.load('./model/'+self.type+'/'+'best_model.tar')
        args.start_iter = checkpoint['iter']
        self.glow.load_state_dict(checkpoint['state_dict'])

        self.pz = 256

        val_can1_dataset = BasicDataset(name='Canon1DsMkIII_ct_5', patch_size=256, patch_num_per_image=1, type='val')
        self.val_can1 = DataLoader(val_can1_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        val_can6_dataset = BasicDataset(name='Canon600D_ct_5', patch_size=256, patch_num_per_image=1, type='val')
        self.val_can6 = DataLoader(val_can6_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        val_fuj_dataset = BasicDataset(name='FujifilmXM1_ct_5', patch_size=256, patch_num_per_image=1, type='val')
        self.val_fuj = DataLoader(val_fuj_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        val_nik_dataset = BasicDataset(name='NikonD5200_ct_5', patch_size=256, patch_num_per_image=1, type='val')
        self.val_nik = DataLoader(val_nik_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

        val_sam_dataset = BasicDataset(name='SamsungNX2000_ct_5', patch_size=256, patch_num_per_image=1, type='val')
        self.val_sam = DataLoader(val_sam_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        val_pan_dataset = BasicDataset(name='PanasonicGX1_ct_5', patch_size=256, patch_num_per_image=1, type='val')
        self.val_pan = DataLoader(val_pan_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        val_sony_dataset = BasicDataset(name='SonyA57_ct_5', patch_size=256, patch_num_per_image=1, type='val')
        self.val_sony = DataLoader(val_sony_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        val_oly_dataset = BasicDataset(name='OlympusEPL6_ct_5', patch_size=256, patch_num_per_image=1, type='val')
        self.val_oly = DataLoader(val_oly_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

        "txt"
        self.error_txt = './error/' + self.type+'_Deltae1.txt'

        try:
            f = open(self.error_txt, 'r')
            f.close()
        except IOError:
            f = open(self.error_txt, 'w')


    def error_evaluation(self,error_list):
        es = np.array(error_list)
        es.sort()
        ae = np.array(es).astype(np.float32)

        x, y, z = np.percentile(ae, [25, 50, 75])
        Mean = np.mean(ae)

        print("Mean\tQ1\tQ2\tQ3")
        print("{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(Mean, x, y, z))

    def save_e(self,val_loader_s_c,type,CAM):
        D_sc, MS_sc, MA_sc,F_avg, F_std,F_avg_avg,F_std_avg = self.do_eval(val_loader_s_c, self.device,CAM)
        print(type)
        self.error_evaluation(D_sc)
        self.error_evaluation(MS_sc)
        self.error_evaluation(MA_sc)

        SC = {}
        SC['DE'] = D_sc
        SC['MS'] = MS_sc
        SC['MA'] = MA_sc
        np.save('./error/' + self.type + '_'+ type, SC)

        return F_avg,F_std,F_avg_avg,F_std_avg



    def do_eval(self, loader,device,CAM=True):
        D,MS,MA = [],[],[]
        F_avg, F_std = [],[]

        for batch in loader:
            "image"
            gt = batch['gt-AWB'].to(device=device, dtype=torch.float32)
            imgs = batch['image'].to(device=device, dtype=torch.float32)
            "label"
            name = batch['name']
            cam = batch['label'].to(device=device, dtype=torch.float32)

            name_ = name[0][0]
            cam_name = name_.split('_')[0]
            visdir_ = './output/vis1/'+cam_name+'/'
            if not(os.path.exists(visdir_)):
                os.makedirs(visdir_)

            z_c = self.glow(imgs, cam, forward=True)
            out_img = self.glow(z_c, cam, forward=False)

            ""
            imgs_ = imgs.cpu().detach().numpy()[0].transpose(1,2,0)
            gt_ = gt.cpu().detach().numpy()[0].transpose(1,2,0)
            out_img_ =out_img.cpu().detach().numpy()[0].transpose(1,2,0)

            deltaE00, MSE, MAE = evaluate_cc(out_img_ * 255, gt_ * 255, 0, opt=3)

            error_save = open(self.error_txt, mode='a')
            error_save.write(
                '\n' + 'Name:' + str(name_)  + '  DELTA E:' + str(round(deltaE00,3))+ '  MSE:' + str(round(MSE[0],3))+ '  MAE:' + str(round(MAE,3)))
            # 关闭文件
            error_save.close()
            D.append(deltaE00)
            MS.append(MSE[0])
            MA.append(MAE)

        D = np.array(D)
        MS = np.array(MS)
        MA = np.array(MA)
        F_avg = np.array(F_avg)
        F_std = np.array(F_std)
        F_avg_avg = np.average(F_avg,axis=0)
        F_std_avg = np.average(F_std,axis=0)

        return D,MS,MA,F_avg,F_std,F_avg_avg,F_std_avg


    def do_testing(self):
        self.glow.eval()

        with torch.no_grad():
            print('--------------------------------')
            F_avg_fuj, F_std_fuj, F_avg_avg_fuj, F_std_avg_fuj = self.save_e(self.val_fuj, type='FujifilmXM1',
                                                                             CAM=self.CAM)
            print('--------------------------------')
            F_avg_can1,F_std_can1,F_avg_avg_can1,F_std_avg_can1 \
                = self.save_e(self.val_can1, type='Canon1DsMkIII',CAM=self.CAM)
            print('--------------------------------')
            F_avg_can6,F_std_can6,F_avg_avg_can6,F_std_avg_can6 \
                = self.save_e(self.val_can6, type='Canon600D',CAM=self.CAM)
            print('--------------------------------')
            F_avg_nik,F_std_nik,F_avg_avg_nik,F_std_avg_nik = self.save_e(self.val_nik, type='NikonD5200',CAM=self.CAM)
            print('--------------------------------')
            F_avg_oly,F_std_oly,F_avg_avg_oly,F_std_avg_oly = self.save_e(self.val_oly, type='OlympusEPL6',CAM=self.CAM)
            print('--------------------------------')
            F_avg_pan,F_std_pan,F_avg_avg_pan,F_std_avg_pan = self.save_e(self.val_pan, type='PanasonicGX1',CAM=self.CAM)
            print('--------------------------------')
            F_avg_sam,F_std_sam,F_avg_avg_sam,F_std_avg_sam = self.save_e(self.val_sam, type='SamsungNX2000',CAM=self.CAM)
            print('--------------------------------')
            F_avg_sony,F_std_sony,F_avg_avg_sony,F_std_avg_sony = self.save_e(self.val_sony, type='SonyA57',CAM=self.CAM)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--size', type=int, default=256,
                        help='New size for the content and style images, \
                        keeping the original size if set to 0')
    parser.add_argument('--crop', action='store_true',
                        help='do center crop to create squared image')
    parser.add_argument('--save_ext', default='.jpg',
                        help='The extension name of the output image')
    parser.add_argument('--output', type=str, default='output',
                        help='Directory to save the output image(s)')

    # glow parameters
    parser.add_argument('--operator', type=str, default='wct',
                        help='style feature transfer operator')
    parser.add_argument('--n_flow', default=8, type=int, help='number of flows in each block')  # 32
    parser.add_argument('--n_block', default=2, type=int, help='number of blocks')  # 4
    parser.add_argument('--no_lu', action='store_true', help='use plain convolution instead of LU decomposed version')
    parser.add_argument('--affine', default=False, type=bool, help='use affine coupling instead of additive')

    parser.add_argument('--train_cam',
                        default='GLOW_WB_RE_12000_15', type=str, help='')

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('------------Camera Testing---------')

    evaluator = Evaluator(args,True, device)
    evaluator.do_testing()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()



