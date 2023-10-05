import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import sys
sys.path.append("..")

import torch
import numpy as np
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import os.path as osp

from mydataset3D import DataSet3D_su
from CXR_Covid_dataset import CXR_train, CXR_val
import timeit, time
from tensorboardX import SummaryWriter
import loss_nnU as loss
from utils.ParaFlop import print_model_parm_nums
from math import ceil
from utils.dino_utils import fix_random_seeds, load_pretrained_weights
import torch.nn.functional as F
import torch.distributed as dist
from apex import amp
from torch.cuda.amp import autocast
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from collections import OrderedDict
import json
import shutil

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
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("--arch", type=str, default='mediumv7')
    parser.add_argument("--is_proj1", type=str2bool, default=False)
    
    parser.add_argument("--snapshot_dir", type=str, default='snapshots/WithCXR/')
    
    parser.add_argument("--reload_from_checkpoint", type=str2bool, default=False)
    parser.add_argument("--reload_snapshot_dir", type=str, default='snapshots/No')
    parser.add_argument("--reload_epoch", type=str, default='2w')
    
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--FP16", type=str2bool, default=True)
    parser.add_argument("--power", type=float, default=0.9)

    # 3D
    parser.add_argument("--num_epochs", type=str, default='1000')
    parser.add_argument("--data_dir3D", type=str, default='./data_list/')
    parser.add_argument("--train_list3D", type=str, default='COVIDSegChallenge/train.txt') 
    parser.add_argument("--nnUNet_preprocessed", type=str, default='../nnUNet/nnUNet_preprocessed/Task115_COVIDSegChallenge')
    parser.add_argument("--input_size3D", type=str, default='32,256,256')
    parser.add_argument("--batch_size3D", type=int, default=2)
    parser.add_argument("--itrs_each_epoch", type=int, default=250)
    parser.add_argument("--num_classes3D", type=int, default=2)
    
    # 2D
    parser.add_argument("--data_dir2D", type=str, default='../CXR_Covid-19_Challenge/')
    parser.add_argument("--label_path2D", type=str, default='./data_list/CXR_Covid-19_Challenge/')
    parser.add_argument("--input_size2D", type=str, default='224,224')
    parser.add_argument("--batch_size2D", type=int, default=32)
    parser.add_argument("--num_classes2D", type=int, default=3)

    return parser


def lr_poly(base_lr, it, max_it, power):
    return base_lr * ((1 - float(it) / max_it) ** (power))


def adjust_learning_rate(optimizer, i_it, lr, num_stemps, power):
    lr = lr_poly(lr, i_it, num_stemps, power)
    optimizer.param_groups[0]['lr'] = lr
    return lr


def main():
    """Create the model and start the training."""
    parser = get_arguments()
    print(parser)
    args = parser.parse_args()
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ['nnUNet_preprocessed'] = args.nnUNet_preprocessed

    # dino.utils.init_distributed_mode(args)
    fix_random_seeds(args.random_seed)

    if args.num_gpus > 1:
        torch.cuda.set_device(args.local_rank)

    if not os.path.isdir(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    writer_out = os.path.join(args.snapshot_dir, 'output.txt')
    writer = SummaryWriter(args.snapshot_dir)

    h, w = map(int, args.input_size2D.split(','))
    input_size2D = (h, w)
    
    d, h, w = map(int, args.input_size3D.split(','))
    input_size3D = (d, h, w)

    cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # Create network.
    model = MiTNET(args.arch, norm2D='IN2', norm3D='IN3', act='LeakyReLU', img_size2D=input_size2D, img_size3D=input_size3D, 
                   num_classes2D=args.num_classes2D, num_classes3D=args.num_classes3D, pretrain=True, pretrain_path="../pretrain_ED/checkpoint0400.pth", modal_type="MM").cuda()

    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    #print(optimizer)

    if args.FP16:
        print("Note: Using FP16 (torch.cuda.amp) during training")
        scaler = torch.cuda.amp.GradScaler()

    # Prototype
    # N and N
    proto2D = torch.zeros(256, 320).cuda()
    proto3D = torch.zeros(256, 320).cuda()
    
    # load checkpoint...
    if args.reload_from_checkpoint:
        print('loading from checkpoint: {}'.format(args.snapshot_dir))
        if os.path.exists(args.snapshot_dir):
            if args.FP16:
                checkpoint = torch.load(osp.join(args.snapshot_dir, "checkpoint.pth"), map_location=torch.device('cpu'))
                pre_dict = checkpoint['model']
                model.load_state_dict(pre_dict)
                proto2D = checkpoint['proto2D'].cuda()
                proto3D = checkpoint['proto3D'].cuda()
                optimizer.load_state_dict(checkpoint['optimizer'])
                args.start_epoch = checkpoint['epoch'] + 1
                max_mean_auc = checkpoint['max_mean_auc']
                scaler.load_state_dict(checkpoint['scaler'])
                print('length of pre layers: %.f' % (len(pre_dict)))
                print('length of model layers: %.f' % (len(model.state_dict())))
                
            else:
                # TODO
                pass
        else:
            print('File not exists in the reload path: {}'.format(args.checkpoint_path))
    else:
        max_mean_auc = 0
    print("Now max mean auc", max_mean_auc)
    
    net_numpool = 5
    weights = np.array([1 / (2 ** i) for i in range(net_numpool)])
    mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
    weights[~mask] = 0
    weights = weights / weights.sum()
    
    loss_cls = torch.nn.CrossEntropyLoss().cuda()
    loss_seg = loss.MultipleOutputLoss(weight_factors=weights).cuda()

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    # data loader
    train_set2D = CXR_train(img_dir=args.data_dir2D, label_file=osp.join(args.label_path2D, "train.txt"))
    trainloader2D = torch.utils.data.DataLoader(train_set2D, batch_size=args.batch_size2D, shuffle=True, num_workers=8, pin_memory=True)#, pin_memory=True, persistent_workers=True
    print(f"2D Training Data loaded: there are {len(train_set2D)} images.")
    val_set2D = CXR_val(img_dir=args.data_dir2D, label_file=osp.join(args.label_path2D, "val.txt"))
    valloader2D = torch.utils.data.DataLoader(val_set2D, batch_size=1, num_workers=4, pin_memory=True)#, pin_memory=True
    print(f"2D Validation Data loaded: there are {len(val_set2D)} images.")
  
    train_set3D = DataSet3D_su(args.data_dir3D, args.train_list3D, max_iters=args.itrs_each_epoch * args.batch_size3D, crop_size=input_size3D)
    trainloader3D = torch.utils.data.DataLoader(train_set3D, batch_size=args.batch_size3D, shuffle=True, num_workers=8, pin_memory=True)#, pin_memory=True
    print(f"3D Training Data loaded: there are {len(train_set3D)} images.")
       
    Iterations = 250000
    iterations_per_epoch = args.itrs_each_epoch
    real_num_epochs = int(Iterations / iterations_per_epoch)
    
    for epoch in range(real_num_epochs):
        if epoch < args.start_epoch:
            continue
        
        epoch_loss_cls = []
        epoch_loss_seg = []
        epoch_loss = []
    
        for it, (batch2D, batch3D) in tqdm(enumerate(zip(trainloader2D, trainloader3D))):
        
            if it >= iterations_per_epoch:  # each epoch contains 250 iterations
                break
            
            step = iterations_per_epoch * epoch + it
            adjust_learning_rate(optimizer, step, args.learning_rate, Iterations, args.power)
        
            model.train()
        
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
        
                # ------------------------2D------------------------ # 
                images2D, labels2D = batch2D
                images2D = images2D.cuda()
                labels2D = labels2D.cuda()
        
                preds2D, proto2D = model(images2D, proto2D, proto3D, '2D', epoch)
                loss_su2D = loss_cls(preds2D, labels2D)
                epoch_loss_cls.append(float(loss_su2D))
        
                # ------------------------3D------------------------ #
                images3D = batch3D['image'][:, 0].cuda()
                labels3D = batch3D['label'][:, 0].cuda()

                preds3D, proto3D = model(images3D, proto3D, proto2D, '3D', epoch)
                loss_su3D = loss_seg(preds3D, labels3D)
                epoch_loss_seg.append(float(loss_su3D))
                
                term_all = loss_su2D + loss_su3D
                epoch_loss.append(float(term_all))
            
            if args.FP16:
                scaler.scale(term_all).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
                scaler.step(optimizer)
                scaler.update()
            else:
                assert 0
                term_all.backward()
                optimizer.step()
        
        # -------------------------log------------------------- #  
        epoch_loss_cls = np.mean(epoch_loss_cls)
        epoch_loss_seg = np.mean(epoch_loss_seg)
        epoch_loss = np.mean(epoch_loss)
        

        if (args.local_rank == 0):
            line_train = 'Epoch {}: lr = {:.4}, loss_cls = {:.4}, loss_seg = {:.4}, loss_all = {:.4}\n'.format(epoch,
                                                                                        optimizer.param_groups[0]['lr'],
                                                                                        epoch_loss_cls.item(),    
                                                                                        epoch_loss_seg.item(),    
                                                                                        epoch_loss.item())
                

            print(line_train)
            with open(writer_out, "a") as f:
                f.write(line_train)

            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('Train_clsloss', epoch_loss_cls.item(), epoch)
            writer.add_scalar('Train_segloss', epoch_loss_seg.item(), epoch)
            writer.add_scalar('Train_loss', epoch_loss.item(), epoch)
            
        # -------------------------val------------------------- #    
        if (epoch + 1) % 10 == 0:
            [val_acc, val_auc, val_AP] = val_mode_cls(valloader2D, model, proto2D, proto3D, epoch)
            line_val = "val epoch %d: vacc=%f, vauc=%f, vAP=%f \n" % (epoch, val_acc, val_auc, val_AP)
            print(line_val)
            writer.add_scalar('Vaild_auc', val_auc, epoch)
            if val_auc > max_mean_auc:
                max_mean_auc = val_auc
                best2D = osp.join(args.snapshot_dir, '2D_best_auc.pth')
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'proto2D': proto2D,
                    'proto3D': proto3D,
                    'max_mean_auc': max_mean_auc,
                    'args': args,
                    'scaler': scaler.state_dict(),
                }
                torch.save(checkpoint, best2D)
                

        # -------------------------save------------------------- #
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'proto2D': proto2D,
            'proto3D': proto3D,
            'max_mean_auc': max_mean_auc,
            'args': args,
            'scaler': scaler.state_dict(),
        }
        name_ = osp.join(args.snapshot_dir, 'checkpoint.pth')
        torch.save(checkpoint, name_)
            
        if (epoch + 1) % 40 == 0:# 40
            it_ = ((epoch + 1) * 250) / 10000
            new_name = osp.join(args.snapshot_dir, 'checkpoint_' + str(int(it_)) + 'w.pth')
            shutil.copyfile(name_, new_name)
            

    end = timeit.default_timer()
    print("%.2f h" % ((end - start)/3600))



def val_mode_cls(dataloader, model, proto2D, proto3D, epoch):
    # valiadation
    pro_score = []
    pro_index = []
    label_val = []
    for index, batch in tqdm(enumerate(dataloader)):
        data, label = batch
        data = data.cuda()

        model.eval()
        with torch.no_grad():
            #pred = model(data)
            pred, _ = model(data, proto2D, proto3D, '2D_val', epoch)
        pred = torch.softmax(pred, 1)
        pro_score.append(pred.cpu().data.numpy())
        pred = torch.argmax(pred, 1)
        pro_index.append(pred.cpu().data.numpy())
        label_val.append(label.data.numpy())

    pro_score = np.concatenate(pro_score, 0)
    pro_index = np.concatenate(pro_index, 0)
    label_val = np.concatenate(label_val, 0)
    label_val_onehot = label_binarize(label_val, classes=np.arange(3)) 
    print(metrics.roc_auc_score(label_val, pro_score, multi_class='ovr'), metrics.roc_auc_score(label_val_onehot, pro_score))

    val_acc = metrics.accuracy_score(label_val, pro_index)
    val_auc = metrics.roc_auc_score(label_val, pro_score, multi_class='ovr')

    return val_acc, val_auc, 0.0


if __name__ == '__main__':
    main()
