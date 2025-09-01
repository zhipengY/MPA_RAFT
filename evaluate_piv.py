import csv
import sys
import time

# import matplotlib.cm
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from core.mpa_raft import MPA_RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder
# import flowiz
import torch.nn.functional as F
from core import PIV_dataset, datasets
from thop import profile
from core.mpa_raft_quarter import MPA_RAFT_quarter

DEVICE = 'cuda'
from matplotlib import pyplot as plt
# import flowiz as fz
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'backend:cudaMallocAsync'

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

    """ Peform validation using the PIV-datasets (train) split """
    path_list = os.listdir(model.args.path)
    for path1 in path_list:
        data_path = model.args.path + "/" + path1 + "/test_data/"    # tiff files
        # data_path = model.args.path + path1 + '/'         # mat files

        val_dataset = PIV_dataset.PIV(split='training', root=data_path)

        out_list, epe_list = [], []
        rmse_list, mse_list, R2_list,rMS_gt, rMS_pre = [], [], [], [], []
        inf_time_avg, name = [], []
        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, valid_gt, extra_info = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape, mode='kitti')
            image1, image2 = padder.pad(image1, image2)

            b, c, h, w = image1.size()
            s = time.time()
            flow_low, flow_pr = model(image1, image2, iters=iter, test_mode=True)

            # print("infurence time:", time.time() - s)
            inf_time_avg.append(time.time() - s)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()

            # mse = torch.mean(torch.sum((flow - flow_gt)**2, axis=0).view(-1))
            mse = torch.mean((flow - flow_gt)**2)

            rmse = torch.sqrt(mse)


            R2 = 1 - torch.sum((flow - flow_gt)**2) / torch.sum((flow_gt - torch.mean(flow_gt))**2)

            rms_gt = torch.sqrt(torch.mean(flow_gt**2))
            rms_pre = torch.sqrt(torch.mean(flow**2))

            mag = torch.sqrt(torch.sum(flow_gt**2, dim=0))

            mag = torch.sum(flow_gt ** 2, dim=0).sqrt()

            epe = epe.view(-1)
            mag = mag.view(-1)
            val = valid_gt.view(-1) >= 0.5

            print("epe:", epe[val].mean().item())

            fig, axarr = plt.subplots(1, 2, figsize=(16, 8))
            u = flow[0, :, :].detach().numpy()
            v = flow[1, :, :].detach().numpy()


            mappable2 = axarr[1].imshow(np.sqrt(v ** 2 + u ** 2), cmap="jet",)
            # axarr[1].title("predict")

            # Control arrow density
            X = np.arange(0, h, 8)
            Y = np.arange(0, w, 8)
            xx, yy = np.meshgrid(X, Y)
            U = u[xx.T, yy.T]
            V = v[xx.T, yy.T]

            # Draw velocity direction
            # axarr[1].quiver(yy.T, xx.T, U, -V)
            axarr[1].axis('off')
            axarr[1].set_title(path1 + "& epe:" + str(round(epe[val].mean().item(), 4)),
                            fontdict={"fontsize": 16})  # results title

            u = flow_gt[0].detach().numpy()
            v = flow_gt[1].detach().numpy()


            # Draw velocity magnitude
            mappable1 = axarr[0].imshow(np.sqrt(v**2 + u ** 2), cmap="jet",)

            # axarr[0].title("GT")
            # Control arrow density
            X = np.arange(0, h, 8)
            Y = np.arange(0, w, 8)
            xx, yy = np.meshgrid(X, Y)
            U = u[xx.T, yy.T]
            V = v[xx.T, yy.T]

            # Draw velocity direction
            # axarr[0].quiver(yy.T, xx.T, U, -V)
            axarr[0].axis('off')
            axarr[0].set_title("Ground Truth", fontdict={"fontsize": 16})

            cax1 = fig.add_axes(
                [axarr[0].get_position().x1 + 0.01, axarr[0].get_position().y0, 0.02, axarr[0].get_position().height])
            cax2 = fig.add_axes(
                [axarr[1].get_position().x1 + 0.01, axarr[1].get_position().y0, 0.02, axarr[1].get_position().height])
            cb1 = fig.colorbar(mappable1, cax=cax1)
            cb2 = fig.colorbar(mappable2, cax=cax2)
            cb1.ax.tick_params(labelsize=16)
            cb2.ax.tick_params(labelsize=16)
            cb1.set_label('pixel displacement', fontsize=16)
            cb2.set_label('pixel displacement', fontsize=16)


            # plt.show()
            if not os.path.exists(f'./output/{args.dataset_name}/{path1}'):
                os.makedirs(f'./output/{args.dataset_name}/{path1}')
            image_name = extra_info[0]
            np.save(f'./output/{args.dataset_name}/' + path1 + '/' + image_name + 'frame_%d.npy' % val_id, flow)
            # np.save('./output/04_all_50000_GMA/' + path1 + '/frame_%d.npy' % val_id, flow)
            fig.savefig(f"./output/{args.dataset_name}/{path1}/{image_name}frame_{val_id}")

            plt.close()

            out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
            epe_list.append(epe[val].mean().item())
            out_list.append(out[val].cpu().numpy())
            rmse_list.append(rmse)
            mse_list.append(mse)
            R2_list.append(R2)
            rMS_gt.append(rms_gt)
            rMS_pre.append(rms_pre)
            name.append(extra_info[0])

        epe_list = np.array(epe_list)
        out_list = np.concatenate(out_list)
        rmse_list = np.array(rmse_list)
        mse_list = np.array(mse_list)
        R2_list = np.array(R2_list)
        rMS_gt = np.array(rMS_gt)
        rMS_pre = np.array(rMS_pre)

        epe = np.mean(epe_list)
        f1 = 100 * np.mean(out_list)

        print(path1, "RMSE:", np.mean(rmse_list))

        import pandas as pd
        data_frame = pd.DataFrame({'name':name,"epe": epe_list, "rmse": rmse_list, "mse": mse_list, "R2": R2_list, "rMS_gt": rMS_gt, 
                                   "rMS_pre": rMS_pre,'infer time': inf_time_avg})
        data_frame.to_csv(f"./output/{args.dataset_name}/{path1}/result.csv", index=False)


@torch.no_grad()
def validate_PIV_single(model, iter):
    """ Peform validation using the PIV-datasets (train) split """
    val_dataset = PIV_dataset.PIV(split='training', root=model.args.path)

    out_list, epe_list = [], []
    rmse_list, mse_list, R2_list,rMS_gt, rMS_pre = [], [], [], [], []
    inf_time_avg, name = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt, extra_info = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        # b, c, h, w = image1.size()
        _, h, w = flow_gt.size()
        s = time.time()
        flow_low, flow_pr = model(image1, image2, iters=iter, test_mode=True)

        print("infurence time:", time.time() - s)
        inf_time_avg.append(time.time() - s)
        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()

        mse = torch.mean((flow - flow_gt)**2)

        rmse = torch.sqrt(mse)

        R2 = 1 - torch.sum((flow - flow_gt)**2) / torch.sum((flow_gt - torch.mean(flow_gt))**2)

        rms_gt = torch.sqrt(torch.mean(flow_gt**2))
        rms_pre = torch.sqrt(torch.mean(flow**2))

        mag = torch.sum(flow_gt ** 2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.contiguous().view(-1) >= 0.5

        print("epe:", epe[val].mean().item())

        fig, axarr = plt.subplots(1, 2, figsize=(16, 8))
        u = flow[0, :, :].detach().numpy()
        v = flow[1, :, :].detach().numpy()

        mappable2 = axarr[1].imshow(np.sqrt(v**2 + u ** 2), cmap="jet",)

        X = np.arange(0, h, 8)
        Y = np.arange(0, w, 8)
        xx, yy = np.meshgrid(X, Y)
        U = u[xx.T, yy.T]
        V = v[xx.T, yy.T]
        # Draw velocity direction
        axarr[1].quiver(yy.T, xx.T, U, -V)
        axarr[1].axis('off')
        axarr[1].set_title(model.args.dataset_name + "& epe:" + str(round(epe[val].mean().item(), 4)),
                        fontdict={"fontsize": 16})  # results title

        u = flow_gt[0].detach().numpy()
        v = flow_gt[1].detach().numpy()

        # Draw velocity magnitude
        mappable1 = axarr[0].imshow(np.sqrt(v ** 2 + u ** 2), cmap="jet",)

        X = np.arange(0, h, 8)
        Y = np.arange(0, w, 8)
        xx, yy = np.meshgrid(X, Y)
        U = u[xx.T, yy.T]
        V = v[xx.T, yy.T]

        # Draw velocity direction
        axarr[0].quiver(yy.T, xx.T, U, -V)
        axarr[0].axis('off')
        axarr[0].set_title("Ground Truth", fontdict={"fontsize": 16})

        cax1 = fig.add_axes(
            [axarr[0].get_position().x1 + 0.01, axarr[0].get_position().y0, 0.02, axarr[0].get_position().height])
        cax2 = fig.add_axes(
            [axarr[1].get_position().x1 + 0.01, axarr[1].get_position().y0, 0.02, axarr[1].get_position().height])
        cb1 = fig.colorbar(mappable1, cax=cax1)
        cb2 = fig.colorbar(mappable2, cax=cax2)
        cb1.ax.tick_params(labelsize=16)
        cb2.ax.tick_params(labelsize=16)
        # cb1.set_label('pixel displacement', fontsize=16)
        cb2.set_label('pixel displacement', fontsize=16)


        plt.show()
        if not os.path.exists(f'./output/{args.dataset_name}'):
            os.makedirs(f'./output/{args.dataset_name}')
        image_name = extra_info[0]
        np.save(f'./output/{args.dataset_name}/{image_name}frame_{val_id}.npy', flow.detach().numpy())
        fig.savefig(f"./output/{args.dataset_name}/{image_name}frame_{val_id}")

        plt.close()

        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())
        rmse_list.append(rmse)
        mse_list.append(mse)
        R2_list.append(R2)
        rMS_gt.append(rms_gt)
        rMS_pre.append(rms_pre)
        name.append(extra_info[0])

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)
    rmse_list = np.array(rmse_list)
    mse_list = np.array(mse_list)
    R2_list = np.array(R2_list)
    rMS_gt = np.array(rMS_gt)
    rMS_pre = np.array(rMS_pre)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation PIV: %f, %f" % (epe, f1))
    print("Average inference time %s Seconds", np.mean(inf_time_avg))
    import pandas as pd
    data_frame = pd.DataFrame({'name':name,"epe": epe_list, "rmse": rmse_list, "mse": mse_list, "R2": R2_list, "rMS_gt": rMS_gt,
                                "rMS_pre": rMS_pre,'infer time': inf_time_avg})
    data_frame.to_csv(f"./output/{args.dataset_name}/result.csv", index=False)


@torch.no_grad()
def validate_PIV_exp(model, iter):
    """ Peform validation using the PIV-datasets (train) split """
    val_dataset = PIV_dataset.PIV(split='testing', root=model.args.path)

    inf_time_avg, name = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, extra_info = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        b, c, h, w = image1.size()
        s = time.time()
        flow_low, flow_pr = model(image1, image2, iters=iter, test_mode=True)

        print("infurence time:", time.time() - s)
        inf_time_avg.append(time.time() - s)
        flow = padder.unpad(flow_pr[0]).cpu()

        u = flow[0, :, :].detach().numpy()
        v = flow[1, :, :].detach().numpy()

        _, _, h, w = image1.size()
        
        mp = plt.imshow(np.sqrt(u ** 2 + v ** 2), cmap='RdYlBu_r', vmin=0, vmax=2)
        X = np.arange(0, h, 8)
        Y = np.arange(0, w, 8)
        xx, yy = np.meshgrid(X, Y)
        U = u[xx.T, yy.T]
        V = v[xx.T, yy.T]
        # Draw velocity direction
        plt.quiver(yy.T, xx.T, U, -V)
        plt.title(extra_info[0])
        plt.axis("off")
        plt.colorbar(mp)
        # plt.show()

        # cb1.set_label('pixel displacement', fontsize=16)
        # plt.set_label('pixel displacement', fontsize=16)

        # plt.show()
        if not os.path.exists(f'./output/{args.dataset_name}'):
            os.makedirs(f'./output/{args.dataset_name}')

        np.save(f'./output/{args.dataset_name}/{extra_info[0]}_flow.npy', flow)
        plt.savefig(f"./output/{args.dataset_name}/{extra_info[0]}")

        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint",
                        default="")  # checkpoint path
    parser.add_argument('--path', help="dataset for evaluation",
                        default="")   # test data path
    parser.add_argument('--dataset_name', help="save file name", default="")
    parser.add_argument('--small', action='store_true', help='use small model', default=True)
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision', default=True)
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    model = MPA_RAFT(args)
    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()

    validate_PIV_single(model, iter=12)


