import argparse
import os, sys
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

sys.path.append("..")

import torch
import numpy as np
import torch.backends.cudnn as cudnn
from scipy.ndimage.filters import gaussian_filter
import math
from tqdm import tqdm
from mydataset3D import ValDataSet
import timeit
from utils.ParaFlop import print_model_parm_nums
import nibabel as nib
from math import ceil
from skimage.measure import label as LAB
import SimpleITK as sitk
from batchgenerators.augmentations.utils import resize_segmentation
from nnunet.preprocessing.preprocessing import get_lowres_axis, get_do_separate_z, resample_data_or_seg
from medpy.metric import hd95
from collections import OrderedDict
import json


from models.UCI import MiTNET


start = timeit.default_timer()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="UCI")
    parser.add_argument("--gpu", type=str, default='0')

    parser.add_argument("--save_path", type=str, default='outputs/withChestXray14/')

    parser.add_argument("--reload_from_checkpoint", type=str2bool, default=True)
    parser.add_argument("--checkpoint_path", type=str, default='snapshots/withChestXray14/')
    parser.add_argument("--is_proj1", type=str2bool, default=False)

    # 3D
    parser.add_argument("--data_dir3D", type=str, default='./data_list/')
    parser.add_argument("--val_list3D", type=str, default='COVIDSegChallenge/val.txt') 
    parser.add_argument("--nnUNet_preprocessed", type=str, default='../nnUNet/nnUNet_preprocessed/Task115_COVIDSegChallenge')
    parser.add_argument("--input_size3D", type=str, default='32,256,256')
    parser.add_argument("--num_classes", type=int, default=2)
    
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("--FP16", type=str2bool, default=True)

    parser.add_argument("--isHD", type=str2bool, default=False)
    parser.add_argument("--arch", type=str, default='mediumv7')
    parser.add_argument("--type", type=str, default='model')

    return parser


def dice_score(preds, labels):
    preds = preds[np.newaxis, :]
    labels = labels[np.newaxis, :]
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    predict = preds.view().reshape(preds.shape[0], -1)
    target = labels.view().reshape(labels.shape[0], -1)

    num = np.sum(np.multiply(predict, target), axis=1)
    den = np.sum(predict, axis=1) + np.sum(target, axis=1) + 1

    dice = 2 * num / den

    return dice.mean()


def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.

    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape).cuda()
    result = result.scatter_(1, input, 1)

    return result



def compute_dice_score(preds, labels):

    preds = torch.softmax(preds, 1)

    pred_pa = preds[:, 0, :, :, :]
    label_pa = labels[:, 0, :, :, :]
    dice_pa = dice_score(pred_pa, label_pa)

    return dice_pa


def assert_shape(test, reference):

    assert test.shape == reference.shape, "Shape mismatch: {} and {}".format(
        test.shape, reference.shape)


class ConfusionMatrix:

    def __init__(self, test=None, reference=None):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.reference_empty = None
        self.reference_full = None
        self.test_empty = None
        self.test_full = None
        self.set_reference(reference)
        self.set_test(test)

    def set_test(self, test):

        self.test = test
        self.reset()

    def set_reference(self, reference):

        self.reference = reference
        self.reset()

    def reset(self):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.test_empty = None
        self.test_full = None
        self.reference_empty = None
        self.reference_full = None

    def compute(self):

        if self.test is None or self.reference is None:
            raise ValueError("'test' and 'reference' must both be set to compute confusion matrix.")

        assert_shape(self.test, self.reference)

        self.tp = int(((self.test != 0) * (self.reference != 0)).sum())
        self.fp = int(((self.test != 0) * (self.reference == 0)).sum())
        self.tn = int(((self.test == 0) * (self.reference == 0)).sum())
        self.fn = int(((self.test == 0) * (self.reference != 0)).sum())
        self.size = int(np.prod(self.reference.shape, dtype=np.int64))
        self.test_empty = not np.any(self.test)
        self.test_full = np.all(self.test)
        self.reference_empty = not np.any(self.reference)
        self.reference_full = np.all(self.reference)

    def get_matrix(self):

        for entry in (self.tp, self.fp, self.tn, self.fn):
            if entry is None:
                self.compute()
                break

        return self.tp, self.fp, self.tn, self.fn

    def get_size(self):

        if self.size is None:
            self.compute()
        return self.size

    def get_existence(self):

        for case in (self.test_empty, self.test_full, self.reference_empty, self.reference_full):
            if case is None:
                self.compute()
                break

        return self.test_empty, self.test_full, self.reference_empty, self.reference_full


def compute_HD95(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, voxel_spacing=None, connectivity=1, **kwargs):

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty or test_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0

    test, reference = confusion_matrix.test, confusion_matrix.reference

    return hd95(test, reference, voxel_spacing, connectivity)



def _get_gaussian(patch_size, sigma_scale=1. / 8) -> np.ndarray:
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i * sigma_scale for i in patch_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(np.float32)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map


def multi_net(net_list, img, proto2D, proto3D):
    # img = torch.from_numpy(img).cuda()
    #print(len(net_list), proto2D.sum(), proto3D.sum())
    padded_prediction, _ = net_list[0](img, proto3D, proto2D, "3D_val")

    if len(padded_prediction) == 2:
        padded_prediction = torch.softmax(padded_prediction[0][0], 1)
    else:
        padded_prediction = torch.softmax(padded_prediction[0], 1)

    for i in range(1, len(net_list)):
        padded_prediction_i = net_list[i](img)
        padded_prediction_i = torch.softmax(padded_prediction_i, 1)
        padded_prediction += padded_prediction_i
    padded_prediction /= len(net_list)
    return padded_prediction  # .cpu().data.numpy()


def predict_sliding(args, net_list, image, tile_size, classes, gaussian_importance_map, proto2D, proto3D):  # tile_size:32x256x256
    # gaussian_importance_map = _get_gaussian(tile_size, sigma_scale=1. / 8)

    # padding or not?
    flag_padding = False
    dept_missing = math.ceil(tile_size[0] - image.shape[2])
    rows_missing = math.ceil(tile_size[1] - image.shape[3])
    cols_missing = math.ceil(tile_size[2] - image.shape[4])
    if rows_missing < 0:
        rows_missing = 0
    if cols_missing < 0:
        cols_missing = 0
    if dept_missing < 0:
        dept_missing = 0
    # image = np.pad(image, ((0, 0), (0, 0), (0, dept_missing), (0, rows_missing), (0, cols_missing)), 'constant')
    image = np.pad(image, ((0, 0), (0, 0), (0, dept_missing), (0, rows_missing), (0, cols_missing)), constant_values = (-1,-1))

    image_size = image.shape
    overlap = 0.5

    strideHW = ceil(tile_size[1] * (1 - overlap))
    strideD = ceil(tile_size[0] * (1 - overlap))
    tile_deps = int(ceil((image_size[2] - tile_size[0]) / strideD) + 1)
    tile_rows = int(ceil((image_size[3] - tile_size[1]) / strideHW) + 1)  # strided convolution formula
    tile_cols = int(ceil((image_size[4] - tile_size[2]) / strideHW) + 1)
    # print("Need %i x %i x %i prediction tiles @ stride %i x %i px" % (tile_deps, tile_cols, tile_rows, strideD, strideHW))
    full_probs = np.zeros(
        (image_size[0], classes, image_size[2], image_size[3], image_size[4]))  # .astype(np.float32)  # 1x4x155x240x240
    count_predictions = np.zeros(
        (image_size[0], classes, image_size[2], image_size[3], image_size[4]))  # .astype(np.float32)
    full_probs = torch.from_numpy(full_probs)
    count_predictions = torch.from_numpy(count_predictions)

    for dep in tqdm(range(tile_deps)):
        for row in range(tile_rows):
            for col in range(tile_cols):
                d1 = int(dep * strideD)
                x1 = int(col * strideHW)
                y1 = int(row * strideHW)
                d2 = min(d1 + tile_size[0], image_size[2])
                x2 = min(x1 + tile_size[2], image_size[4])
                y2 = min(y1 + tile_size[1], image_size[3])
                d1 = max(int(d2 - tile_size[0]), 0)
                x1 = max(int(x2 - tile_size[2]), 0)  # for portrait images the x1 underflows sometimes
                y1 = max(int(y2 - tile_size[1]), 0)  # for very few rows y1 underflows

                img = image[:, :, d1:d2, y1:y2, x1:x2]
                img = torch.from_numpy(img).cuda()

                # prediction1 = multi_net(net_list, img)
                # prediction2 = multi_net(net_list, img[:, :, :, :, ::-1].copy())[:, :, :, :, ::-1]
                # prediction3 = multi_net(net_list, img[:, :, :, ::-1, :].copy())[:, :, :, ::-1, :]
                # prediction4 = multi_net(net_list, img[:, :, ::-1, :, :].copy())[:, :, ::-1, :, :]
                # prediction = (prediction1 + prediction2 + prediction3 + prediction4) / 4.
                # prediction = torch.from_numpy(prediction)
                
                prediction = multi_net(net_list, img, proto2D, proto3D)
                prediction2 = torch.flip(multi_net(net_list, torch.flip(img, [2]), proto2D, proto3D), [2])
                prediction3 = torch.flip(multi_net(net_list, torch.flip(img, [3]), proto2D, proto3D), [3])
                prediction4 = torch.flip(multi_net(net_list, torch.flip(img, [4]), proto2D, proto3D), [4])
                prediction5 = torch.flip(multi_net(net_list, torch.flip(img, [2, 3]), proto2D, proto3D), [2, 3])
                prediction6 = torch.flip(multi_net(net_list, torch.flip(img, [2, 4]), proto2D, proto3D), [2, 4])
                prediction7 = torch.flip(multi_net(net_list, torch.flip(img, [3, 4]), proto2D, proto3D), [3, 4])
                prediction8 = torch.flip(multi_net(net_list, torch.flip(img, [2, 3, 4]), proto2D, proto3D), [2, 3, 4])
                prediction = (prediction + prediction2 + prediction3 + prediction4 + prediction5 + prediction6 + prediction7 + prediction8) / 8.
                # prediction = prediction1
                prediction = prediction.cpu()

                prediction[:, :] *= gaussian_importance_map

                if isinstance(prediction, list):
                    shape = np.array(prediction[0].shape)
                    shape[0] = prediction[0].shape[0] * len(prediction)
                    shape = tuple(shape)
                    preds = torch.zeros(shape).cuda()
                    bs_singlegpu = prediction[0].shape[0]
                    for i in range(len(prediction)):
                        preds[i * bs_singlegpu: (i + 1) * bs_singlegpu] = prediction[i]
                    count_predictions[:, :, d1:d2, y1:y2, x1:x2] += 1
                    full_probs[:, :, d1:d2, y1:y2, x1:x2] += preds

                else:
                    count_predictions[:, :, d1:d2, y1:y2, x1:x2] += gaussian_importance_map
                    full_probs[:, :, d1:d2, y1:y2, x1:x2] += prediction

    # average the predictions in the overlapping regions
    full_probs /= count_predictions
    # visualize normalization Weights
    # plt.imshow(np.mean(count_predictions, axis=2))
    # plt.show()
    return full_probs[:,:,:(image_size[2]-dept_missing), :(image_size[3]-rows_missing), :(image_size[4]-cols_missing)]


def save_nii(args, pred, name, properties):  # bs, c, WHD

    # segmentation = pred.transpose((1, 2, 0))  # bsx240x240x155
    segmentation = pred

    # save
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    current_shape = segmentation.shape
    # shape_original_after_cropping = np.array(properties.get('size_after_cropping'))[[0,2,1]]
    # shape_original_before_cropping = properties.get('original_size_of_raw_data')[0].data.numpy()[[0,2,1]]

    shape_original_after_cropping = np.array(properties.get('size_after_cropping'), dtype='int')[[0,2,1]]
    shape_original_before_cropping = properties.get('original_size_of_raw_data')[0].data.numpy()[[0,2,1]]

    order = 1
    force_separate_z = None

    if np.any(np.array(current_shape) != np.array(shape_original_after_cropping)):
        if order == 0:
            seg_old_spacing = resize_segmentation(segmentation, shape_original_after_cropping, 0)
        else:
            if force_separate_z is None:
                if get_do_separate_z(properties.get('original_spacing').data.numpy()[0]):
                    do_separate_z = True
                    lowres_axis = get_lowres_axis(properties.get('original_spacing').data.numpy()[0])
                elif get_do_separate_z(properties.get('spacing_after_resampling').data.numpy()):
                    do_separate_z = True
                    lowres_axis = get_lowres_axis(properties.get('spacing_after_resampling').data.numpy()[0])
                else:
                    do_separate_z = False
                    lowres_axis = None
            else:
                do_separate_z = force_separate_z
                if do_separate_z:
                    lowres_axis = get_lowres_axis(properties.get('original_spacing').data.numpy()[0])
                else:
                    lowres_axis = None

            print("separate z:", do_separate_z, "lowres axis", lowres_axis)
            seg_old_spacing = resample_data_or_seg(segmentation[None], shape_original_after_cropping, is_seg=True,
                                                   axis=lowres_axis, order=order, do_separate_z=do_separate_z,
                                                   order_z=0)[0] # gqb
    else:
        seg_old_spacing = segmentation

    bbox = properties.get('crop_bbox')

    if bbox is not None:
        seg_old_size = np.zeros(shape_original_before_cropping)
        for c in range(3):
            bbox[c][1] = np.min((bbox[c][0] + seg_old_spacing.shape[c], shape_original_before_cropping[c]))
        seg_old_size[bbox[0][0]:bbox[0][1],
        bbox[1][0]:bbox[1][1],
        bbox[2][0]:bbox[2][1]] = seg_old_spacing
    else:
        seg_old_size = seg_old_spacing

    if name[0][:4] == 'kidn' or name[0][:4] == 'case':
        seg_old_size = np.rot90(seg_old_size[:, ::-1, :], 1, [1, 2])
        seg_old_size = seg_old_size.transpose([1, 2, 0])
        name[0] = name[0].replace("kidney","case")

    seg_resized_itk = sitk.GetImageFromArray(seg_old_size.astype(np.uint8))
    seg_resized_itk.SetSpacing(np.array(properties['itk_spacing']).astype(np.float64))
    seg_resized_itk.SetOrigin(np.array(properties['itk_origin']).astype(np.float64))
    seg_resized_itk.SetDirection(np.array(properties['itk_direction']).astype(np.float64))
    sitk.WriteImage(seg_resized_itk, args.save_path + name[0]+'.nii.gz')

    return None

def validate(args, input_size, model, ValLoader, num_classes, json_dict, proto2D, proto3D):

    for index, batch in enumerate(ValLoader):
        # print('%d processd' % (index))
        image, label, name, properties = batch

        print("Processing %s" % name[0])
        with torch.no_grad():
            gaussian_importance_map = _get_gaussian(input_size, sigma_scale=1. / 8)
            pred = predict_sliding(args, model, image.numpy(), input_size, num_classes, gaussian_importance_map, proto2D, proto3D)
            size_after_resampling = np.array(properties["size_after_resampling"])
            pred = pred[:,:,0:size_after_resampling[0], 0:size_after_resampling[1], 0:size_after_resampling[2]]
        
            pred_tumor = np.asarray(np.argmax(pred, 1))
        
            # save
            save_nii(args, pred_tumor[0], name, properties)
            print("Saving done.")


    # evaluate metrics
    print("Start to evaluate...")

    val_Dice = np.zeros(shape=(1,args.num_classes-1))
    val_HD = np.zeros(shape=(1,args.num_classes-1))
    count_Dice = 0.
    count_HD = 0.

    for root, dirs, files in os.walk(args.save_path):
        for i in sorted(files):
            if i[-6:]!='nii.gz':
                continue
            i_file = os.path.join(root, i)
            i2_file = os.path.join(os.environ["nnUNet_preprocessed"], 'gt_segmentations', i)
            predNII = nib.load(i_file)
            labelNII = nib.load(i2_file)
            pred = predNII.get_data()
            label = labelNII.get_data()

            voxel_spacing = np.array(sitk.ReadImage(i_file).GetSpacing())[::-1]

            for cls_i in range(1, args.num_classes):
                dice_i = dice_score(pred==cls_i, label==cls_i)

                if args.isHD:
                    HD_i = compute_HD95(test=(pred==cls_i), reference=(label==cls_i), voxel_spacing=voxel_spacing)
                else:
                    HD_i = 999.

                val_Dice[0, cls_i-1] += dice_i
                val_HD[0, cls_i-1] += HD_i if HD_i!=999. else 0

                log_i = ('Processing {} Score{} = [Dice-{:.4}; HD-{:.4}]'.format(i[0:-7], cls_i, dice_i, HD_i))
                print("%s: %s" % (i[0:-7], log_i))
                json_dict[i]=log_i

            count_Dice += 1
            count_HD += 1

    count_Dice = count_Dice  if count_Dice!=0 else 1
    count_HD = count_HD  if count_HD!=0 else 1
    val_Dice = val_Dice / count_Dice
    val_HD = val_HD / count_HD

    for cls_j in range(args.num_classes-1):
        print('Average score{}: {:.4}'.format(cls_j + 1, val_Dice[0, cls_j]))

    print('Average score: {:.4}'.format(np.mean(val_Dice[0])))

    return val_Dice, val_HD
    


def main():
    """Create the model and start the training."""
    parser = get_arguments()
    print(parser)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    os.environ['nnUNet_preprocessed'] = args.nnUNet_preprocessed
    if args.num_gpus > 1:
        torch.cuda.set_device(args.local_rank)

    d, h, w = map(int, args.input_size3D.split(','))
    input_size3D = (d, h, w)

    cudnn.benchmark = True
    seed = 1234
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Create network.
    model = MiTNET(args.arch, norm2D='IN2', norm3D='IN3', act='LeakyReLU', img_size2D=(224, 224), img_size3D=input_size3D, num_classes2D=14, num_classes3D=args.num_classes, pretrain=False, pretrain_path=None, modal_type="MM").cuda()

    print_model_parm_nums(model)

    model.eval()

    device = torch.device('cuda:{}'.format(args.local_rank))
    model.to(device)

    # load checkpoint...
    if args.reload_from_checkpoint:
        print('loading from checkpoint: {}'.format(args.checkpoint_path))
        if os.path.exists(args.checkpoint_path):
            checkpoint = torch.load(os.path.join(args.checkpoint_path, "checkpoint.pth"), map_location=torch.device('cpu'))
            pre_dict = checkpoint['model']
            model.load_state_dict(pre_dict)#, strict=False
            proto2D = checkpoint['proto2D'].cuda()
            proto3D = checkpoint['proto3D'].cuda()
            
            print('length of pre layers: %.f' % (len(pre_dict)))
            print('length of model layers: %.f' % (len(model.state_dict())))
            
        else:
            print('File not exists in the reload path: {}'.format(args.checkpoint_path))

    valloader = torch.utils.data.DataLoader(
        ValDataSet(args.data_dir3D, args.val_list3D, crop_size=input_size3D),
        batch_size=1,
        pin_memory=True,
    )

    json_dict = OrderedDict()
    json_dict['name'] = "Covid"
    json_dict["meanDice"] = OrderedDict()
    json_dict["meanHD"] = OrderedDict()

    print('validate ...')
    val_Dice, val_HD = validate(args, input_size3D, [model], valloader, args.num_classes, json_dict, proto2D, proto3D)

    json_dict["meanDice"]["Dice_all"] = str(val_Dice[0])
    json_dict["meanDice"]["Dice_average"] = str(np.nanmean(val_Dice[0]))
    json_dict["meanHD"]["HD_all"] = str(val_HD[0])
    json_dict["meanHD"]["HD_average"] = str(np.nanmean(val_HD[0]))

    print(json_dict["meanDice"])
    print(json_dict["meanHD"])

    with open(os.path.join(args.save_path, "summary.json"), 'w') as f:
        json.dump(json_dict, f, indent=4, sort_keys=True)

    end = timeit.default_timer()
    print(end - start, 'seconds')


if __name__ == '__main__':
    main()
