# Data loading based on https://github.com/NVIDIA/flownet2-pytorch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from os.path import *
import os
import math
import random
from glob import glob
import os.path as osp

from core.utils import frame_utils
from core.utils.augmentor import FlowAugmentor, SparseFlowAugmentor

class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):

        if self.is_test:
            if splitext(self.image_list[index][0])[-1] == ".mat":
                img1,img2, flow = frame_utils.read_gen(self.image_list[index][0])
            else:
                img1 = np.array(frame_utils.read_gen(self.image_list[index][0]))
                img2 = np.array(frame_utils.read_gen(self.image_list[index][1]))

                if len(img1.shape) == 2:
                    img1 = np.tile(img1[..., None], (1, 1, 3))
                    img2 = np.tile(img2[..., None], (1, 1, 3))
                img1 = np.array(img1).astype(np.uint8)[..., :3]
                img2 = np.array(img2).astype(np.uint8)[..., :3]
                # img1 = np.resize(np.array(img1).astype(np.uint8)[..., :3],(512,512,3))
                # img2 = np.resize(np.array(img2).astype(np.uint8)[..., :3],(512,152,3))
                img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
                img2 = torch.from_numpy(img2).permute(2, 0, 1).float()

            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None

        if splitext(self.image_list[index][0])[-1] == ".mat":
            img1, img2, flow = frame_utils.read_gen(self.image_list[index][0])

        else:
            if self.sparse:
                flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
            else:
                flow = frame_utils.read_gen(self.flow_list[index])
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # flow = np.resize(np.array(flow).astype(np.float32), (256, 256, 2))
        # img1 = np.resize(np.array(img1).astype(np.uint8), (256, 256))
        # img2 = np.resize(np.array(img2).astype(np.uint8), (256, 256))

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        # flow = torch.from_numpy(flow).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        return img1, img2, flow, valid.float(), self.extra_info[index]
        # return img1, img2, flow, valid.float()

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self

    def __len__(self):
        return len(self.image_list)

class PIV(FlowDataset):
    def __init__(self, aug_params=None, split='training', root=''):
        super(PIV, self).__init__(aug_params, sparse=False)
        if split == 'testing':
            self.is_test = True

        # ***********************************************
        """ root / type / training(testing) """
        # root = osp.join(root, split)
        # self.flow_list = []
        # images1 = []
        # images2 = []
        # for scene in os.listdir(root):
            # imgs1 = sorted(glob(osp.join(root, scene + "/test_data/", '*img1.jpg')))
            # imgs2 = sorted(glob(osp.join(root, scene + "/test_data/", '*img2.jpg')))
            # flow_ = sorted(glob(osp.join(root, scene + "/test_data/", '*.npz')))

            # imgs1 = sorted(glob(osp.join(root, scene, '*img1.jpg')))
            # imgs2 = sorted(glob(osp.join(root, scene, '*img2.jpg')))
            # flow_ = sorted(glob(osp.join(root, scene, '*.npz')))

            # imgs1 = sorted(glob(osp.join(root, '*img1.jpg')))
            # imgs2 = sorted(glob(osp.join(root, '*img2.jpg')))
            # flow_ = sorted(glob(osp.join(root, '*.npz')))
            #
            # self.flow_list = self.flow_list + flow_
            # images1 = images1 + imgs1
            # images2 = images2 + imgs2
            # del imgs1, imgs2, flow_

        # if split == 'test':
        #     self.is_test = True

        # ***********************************************

        # root = osp.join(root, split)

        # images1 = sorted(glob(osp.join(root, '*image1.tif')))
        # images2 = sorted(glob(osp.join(root, '*image2.tif')))

        # images1 = sorted(glob(osp.join(root, '*1.tif')))
        # images2 = sorted(glob(osp.join(root, '*2.tif')))

        # images1 = sorted(glob(osp.join(root, '*img1.tif')))
        # images2 = sorted(glob(osp.join(root, '*img2.tif')))

        images1 = sorted(glob(osp.join(root, '*image1.tif')))
        images2 = sorted(glob(osp.join(root, '*image2.tif')))

        for img1, img2 in zip(images1, images2):
            frame_id_ = img1.split('/')[-1]
            frame_id = frame_id_.split('.')[0]
            # frame_id = img1.split('/')[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]

        if split == 'training':
            # self.flow_list = sorted(glob(osp.join(root, '*.flo')))
            # self.flow_list = sorted(glob(osp.join(root, '*.npy')))
            self.flow_list = sorted(glob(osp.join('')))

class PIV_MAT(FlowDataset):
    def __init__(self, aug_params=None, split='training', root=''):
        super(PIV_MAT, self).__init__(aug_params)
        # flow_root = osp.join(root, split, 'flow')
        # image_root = osp.join(root, split)   # datasets name / training
        image_root = root

        if split == 'testing':
            self.is_test = True

        # for scene in os.listdir(image_root):
            # image_list = sorted(glob(osp.join(image_root, scene,'*.mat')))
        image_list = sorted(glob(osp.join(image_root, '*.mat')))       # val
        
        for i in range(len(image_list) - 1):
            scene = image_list[i].split('/')[-1]       # single
            self.image_list += [[image_list[i], image_list[i + 1]]]
            self.extra_info += [(scene, i)]  # scene and frame_id

            # if split != 'test':
            #     self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))

