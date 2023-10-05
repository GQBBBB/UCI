import os
import os.path as osp
from re import X
import numpy as np
import random
import collections
import torch
import torchvision
from torch.utils import data
import matplotlib.pyplot as plt
import nibabel as nib
from skimage.transform import resize
import SimpleITK as sitk
import math
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
#from batchgenerators.transforms import Compose
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform, \
    BrightnessTransform, ContrastAugmentationTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
import pickle
from batchgenerators.dataloading.data_loader import DataLoaderBase, SlimDataLoaderBase, DataLoader
from batchgenerators.utilities.file_and_folder_operations import subfiles


class DataSet3D_su(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(64, 192, 192)):
        self.root = root
        self.list_path = list_path
        self.crop_d, self.crop_h, self.crop_w = crop_size
        self.img_ids = [i_id.strip().split() for i_id in open(self.root + self.list_path)]

        print("Start preprocessing....")
        print(self.img_ids)

        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        for item in self.img_ids:
            # print(item)
            img_identifier = item[0]
            preprocessed_data_path = os.environ['nnUNet_preprocessed']
            img_gt_file = osp.join(preprocessed_data_path, 'nnUNetData_plans_v2.1_stage1', img_identifier + '.npy')
            with open(os.path.join(preprocessed_data_path, 'nnUNetData_plans_v2.1_stage1', img_identifier + '.pkl'), 'rb') as f:
                properties = pickle.load(f)
            for key in list(properties["class_locations"].keys()):
                if len(properties["class_locations"][key])==0:
                    del properties["class_locations"][key]
            self.files.append({
                "img_lab": img_gt_file,
                "name": img_identifier,
                "properties": properties
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

        self.tr_transforms = get_train_transform(patch_size=crop_size)

    def __len__(self):
        return len(self.files)

    def truncate(self, CT):

        min_HU = -958
        max_HU = 327
        subtract = 82.92
        divide = 136.97

        # truncate
        CT[np.where(CT <= min_HU)] = min_HU
        CT[np.where(CT >= max_HU)] = max_HU
        CT = CT - subtract
        CT = CT / divide
        return CT

    def locate_bbx(self, label, class_locs):

        img_d, img_h, img_w = label.shape

        if random.random() < 0.5:
            selected_class = np.random.choice(len(class_locs)) + 1
            if selected_class in class_locs:
                if len(class_locs[selected_class]) == 0:
                    # if no foreground found, then randomly select
                    d0 = random.randint(0, img_d - self.crop_d)
                    h0 = random.randint(0, img_h - self.crop_h)
                    w0 = random.randint(0, img_w - self.crop_w)
                    d1 = d0 + self.crop_d
                    h1 = h0 + self.crop_h
                    w1 = w0 + self.crop_w
                else:
                    selected_voxel = class_locs[selected_class][np.random.choice(len(class_locs[selected_class]))]
                    center_d, center_h, center_w = selected_voxel

                    d0 = center_d - self.crop_d // 2
                    d1 = center_d + self.crop_d // 2
                    h0 = center_h - self.crop_h // 2
                    h1 = center_h + self.crop_h // 2
                    w0 = center_w - self.crop_w // 2
                    w1 = center_w + self.crop_w // 2

                    if h0 < 0:
                        delta = h0 - 0
                        h0 = 0
                        h1 = h1 - delta
                    if h1 > img_h:
                        delta = h1 - img_h
                        h0 = h0 - delta
                        h1 = img_h
                    if w0 < 0:
                        delta = w0 - 0
                        w0 = 0
                        w1 = w1 - delta
                    if w1 > img_w:
                        delta = w1 - img_w
                        w0 = w0 - delta
                        w1 = img_w
                    if d0 < 0:
                        delta = d0 - 0
                        d0 = 0
                        d1 = d1 - delta
                    if d1 > img_d:
                        delta = d1 - img_d
                        d0 = d0 - delta
                        d1 = img_d
            else:
                d0 = random.randint(0, img_d - self.crop_d)
                h0 = random.randint(0, img_h - self.crop_h)
                w0 = random.randint(0, img_w - self.crop_w)
                d1 = d0 + self.crop_d
                h1 = h0 + self.crop_h
                w1 = w0 + self.crop_w

        else:
            d0 = random.randint(0, img_d - self.crop_d)
            h0 = random.randint(0, img_h - self.crop_h)
            w0 = random.randint(0, img_w - self.crop_w)
            d1 = d0 + self.crop_d
            h1 = h0 + self.crop_h
            w1 = w0 + self.crop_w

        d0 = np.max([d0, 0])
        d1 = np.min([d1, img_d])
        h0 = np.max([h0, 0])
        h1 = np.min([h1, img_h])
        w0 = np.max([w0, 0])
        w1 = np.min([w1, img_w])

        return [d0, d1, h0, h1, w0, w1]

    def pad_image(self, img, target_size):
        """Pad an image up to the target size."""
        rows_missing = math.ceil(target_size[0] - img.shape[0])
        cols_missing = math.ceil(target_size[1] - img.shape[1])
        dept_missing = math.ceil(target_size[2] - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img


    def __getitem__(self, index):
        
        datafiles = self.files[index]
        # read npy file
        img_lab = np.load(datafiles["img_lab"])#['data']
        image = img_lab[0]
        label = img_lab[1]
        name = datafiles["name"]
        class_locs = datafiles["properties"]["class_locations"]

        image = self.pad_image(image, [self.crop_d, self.crop_h, self.crop_w])
        label = self.pad_image(label, [self.crop_d, self.crop_h, self.crop_w])

        [d0, d1, h0, h1, w0, w1] = self.locate_bbx(label, class_locs)

        image = image[d0: d1, h0: h1, w0: w1]
        label = label[d0: d1, h0: h1, w0: w1]

        image = image[np.newaxis, :]
        label = label[np.newaxis, :]
        label[label<0]=0.

        image = image.astype(np.float32)
        label = label.astype(np.float32)

        data_dict = {'image': image[np.newaxis, :], 'label': label[np.newaxis, :], 'name': name}
        data_dict = self.tr_transforms(**data_dict)

        return data_dict


class ValDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(64, 192, 192), mean=(128, 128, 128), scale=True,
                 mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_d, self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.img_ids = [i_id.strip().split() for i_id in open(self.root + self.list_path)]
        print(self.img_ids)

        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        for item in self.img_ids:
            print(item)
            img_identifier = item[0]
            preprocessed_data_path = os.environ['nnUNet_preprocessed']
            img_gt_file = osp.join(preprocessed_data_path, 'nnUNetData_plans_v2.1_stage1', img_identifier + '.npy')
            with open(os.path.join(preprocessed_data_path, 'nnUNetData_plans_v2.1_stage1', img_identifier + '.pkl'), 'rb') as f:
                properties = pickle.load(f)
            for key in list(properties["class_locations"].keys()):
                if len(properties["class_locations"][key])==0:
                    del properties["class_locations"][key]
            self.files.append({
                "img_lab": img_gt_file,
                "name": img_identifier,
                "properties": properties
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

    def __len__(self):
        return len(self.files)

    def truncate(self, CT):

        min_HU = -1024
        max_HU = 325
        subtract = 158.58
        divide = 324.70

        # truncate
        CT[np.where(CT <= min_HU)] = min_HU
        CT[np.where(CT >= max_HU)] = max_HU
        CT = CT - subtract
        CT = CT / divide
        return CT

    def pad_image(self, img, target_size):
        """Pad an image up to the target size."""
        rows_missing = math.ceil(target_size[0] - img.shape[0])
        cols_missing = math.ceil(target_size[1] - img.shape[1])
        dept_missing = math.ceil(target_size[2] - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img

    def __getitem__(self, index):
        datafiles = self.files[index]
        # read npz file
        img_lab = np.load(datafiles["img_lab"])#['data']
        image = img_lab[0]
        label = img_lab[1]
        name = datafiles["name"]
        properties = datafiles["properties"]

        image = self.pad_image(image, [self.crop_d, self.crop_h, self.crop_w])
        label = self.pad_image(label, [self.crop_d, self.crop_h, self.crop_w])

        # image = self.truncate(image)
        # label = self.id2trainId(label)

        image = image[np.newaxis, :]
        label = label[np.newaxis, :]
        label = (label > 0)

        image = image.astype(np.float32)
        label = label.astype(np.float32)

        return image.copy(), label.copy(), name, properties


def get_train_transform(patch_size):
    tr_transforms = []

    tr_transforms.append(RenameTransform('image', 'data', True))
    tr_transforms.append(RenameTransform('label', 'seg', True))

    tr_transforms.append(
        SpatialTransform(
            patch_size, patch_center_dist_from_border=[i // 2 for i in patch_size],
            do_elastic_deform=False, alpha=(0., 900.), sigma=(9., 13.),
            do_rotation=True,
            angle_x=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            angle_y=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            angle_z=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            do_scale=True, scale=(0.85, 1.25),
            border_mode_data='constant', border_cval_data=0,
            order_data=3, border_mode_seg="constant", border_cval_seg=-1,
            order_seg=1,
            random_crop=True,
            p_el_per_sample=0.2, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
            independent_scale_for_each_axis=False)
    )
    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
    tr_transforms.append(
        GaussianBlurTransform(blur_sigma=(0.5, 1.), different_sigma_per_channel=True, p_per_channel=0.5,
                              p_per_sample=0.2))
    tr_transforms.append(BrightnessMultiplicativeTransform((0.75, 1.25), p_per_sample=0.15))
    tr_transforms.append(BrightnessTransform(0.0, 0.1, True, p_per_sample=0.15, p_per_channel=0.5))
    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
    tr_transforms.append(
        SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True, p_per_channel=0.5, order_downsample=0,
                                       order_upsample=3, p_per_sample=0.25, ignore_axes=None))
    tr_transforms.append(GammaTransform(gamma_range=(0.7, 1.5), invert_image=False, per_channel=True, retain_stats=True,
                                        p_per_sample=0.15))

    tr_transforms.append(MirrorTransform(axes=(0, 1, 2)))

    tr_transforms.append(RenameTransform('data', 'image', True))
    tr_transforms.append(RenameTransform('seg', 'label', True))

    # now we compose these transforms together
    tr_transforms = Compose(tr_transforms)
    return tr_transforms





