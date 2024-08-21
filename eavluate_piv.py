import csv
import sys

sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
import flowiz
import torch.nn.functional as F
from core.raft_GMA import RAFT_GMA
from core import datasets
from core.PAFT import PAFT
DEVICE = 'cuda'
from matplotlib import pyplot as plt
import flowiz as fz

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    if len(img.shape) == 2:
        img = np.tile(img[..., None], (1, 1, 3))
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imshow('image', img_flo[:, :, [2, 1, 0]] / 255.0)
    cv2.waitKey()

@torch.no_grad()
def validate_PIV(model, iter):
    """ Peform validation using the KITTI-2015 (train) split """

    val_dataset = datasets.PIV(split='training')

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        b,c,h,w = image1.size()
        flow_low, flow_pr = model(image1, image2, iters=iter, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()
        fig, axarr = plt.subplots(1, 2, figsize=(16, 8))
        u = flow[0, :, :].detach().numpy()
        v = flow[1, :, :].detach().numpy()
        flow_vis = np.concatenate((np.expand_dims(u, axis=2), np.expand_dims(v, axis=2)), 2)
        # flow_vis = np.concatenate((u.view(h, w, 1), v.view(h, w, 1)), 2)
        axarr[1].imshow(fz.convert_from_flow(flow_vis))
        # axarr[1].title("predict")
        # Control arrow density
        X = np.arange(0, h, 8)
        Y = np.arange(0, w, 8)
        xx, yy = np.meshgrid(X, Y)
        U = u[xx.T, yy.T]
        V = v[xx.T, yy.T]
        # Draw velocity direction
        axarr[1].quiver(yy.T, xx.T, U, -V)
        axarr[1].axis('off')

        u = flow_gt[0].detach()
        v = flow_gt[1].detach()

        color_data_label = np.concatenate((u.view(h, w, 1), v.view(h, w, 1)), 2)
        u = u.numpy()
        v = v.numpy()
        # Draw velocity magnitude
        axarr[0].imshow(fz.convert_from_flow(color_data_label))
        # axarr[0].title("GT")
        # Control arrow density
        X = np.arange(0, h, 8)
        Y = np.arange(0, w, 8)
        xx, yy = np.meshgrid(X, Y)
        U = u[xx.T, yy.T]
        V = v[xx.T, yy.T]

        # Draw velocity direction
        axarr[0].quiver(yy.T, xx.T, U, -V)
        axarr[0].axis('off')
        fig.savefig("./output/RAFTGMA/frame_%d" % val_id)

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5
        print("epe:",epe[val].mean().item())

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)
    with open("./output/RAFTGMA/EPE.csv") as f:
        writer = csv.writer(f)
        writer.writerow(epe_list)

    print("Validation PIV: %f, %f" % (epe, f1))
    # return {'PIV-epe': epe, 'piv-f1': f1}


def test(args):


    with torch.no_grad():
        images1_f = glob.glob(os.path.join(args.path, '*_img1.tif'))
        images2_f = glob.glob(os.path.join(args.path, '*_img2.tif'))

        # images = sorted(images)
        count = 0
        for imfile1, imfile2 in zip(images1_f, images2_f):
            count += 1
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            h_org, w_org = image1.shape[-2], image1.shape[-1]
            save_to_disk = True
            resize = True
            if image1.shape[-1] == 256 & image1.shape[-2] == 256 & image2.shape[-1] == 256 & image2.shape[-2] == 256:
                resize = False
            else:
                resize = True

            if resize:
                image1 = F.interpolate(image1.view(-1, 3, h_org, w_org), (256, 256),
                                       mode='bilinear',
                                       align_corners=False)
                image2 = F.interpolate(image2.view(-1, 3, h_org, w_org), (256, 256),
                                       mode='bilinear',
                                       align_corners=False)
            image1 = torch.squeeze(image1)
            image2 = torch.squeeze(image2)

            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            # padder = InputPadder(image1.shape, mode='kitti')
            # image1, image2 = padder.pad(image1, image2)

            b, _, h, w = image1.size()
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            flow = flow_up[0].permute(1, 2, 0).cpu().numpy()
            # viz(image1, flow_up)

            # visualization
            fig, axarr = plt.subplots(1, 1, figsize=(16, 8))
            resize_ratio_u = h_org / 256
            resize_ratio_v = w_org / 256
            u = flow[:, :, 0] * resize_ratio_u
            v = flow[:, :, 1] * resize_ratio_v
            # u = flow[0, :, :].detach()
            # v = flow[1, :, :].detach()

            U_len = u.shape[0] * u.shape[1]
            V_len = v.shape[0] * v.shape[1]

            color_data_pre = np.stack((u, v), axis=2)


            f = flowiz.convert_from_flow(color_data_pre)
            plt.imshow(f)
            plt.show()
            axarr.imshow(f)

            # Control arrow density
            X = np.arange(0, h, 8)
            Y = np.arange(0, w, 8)
            xx, yy = np.meshgrid(X, Y)
            U = u[xx.T, yy.T]
            V = v[xx.T, yy.T]
            # Draw velocity direction
            axarr.quiver(yy.T, xx.T, U, -V, scale=1, scale_units="xy")
            # axarr.axis('off')
            # plt.show()
            color_data_pre_unliteflownet = color_data_pre
            if save_to_disk:
                fig.savefig('./output/frame_%d.png' % count, bbox_inches='tight')
                print("output % d" % count)
                plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint", default="./checkpoints/GMA_80000_MECAraft-PIV.pth")
    parser.add_argument('--path', help="dataset for evaluation", default="/home/asus/yangzhipeng/datasets/pivdata/test_data")
    parser.add_argument('--small', action='store_true', help='use small model', default=True)
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision', default=True)
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    model = RAFT_GMA(args)
    model.load_state_dict(torch.load(args.model))

    # model = model.module
    model.to(DEVICE)
    model.eval()

    validate_PIV(model,iter=12)

