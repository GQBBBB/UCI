from numpy import False_
import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from utils.dino_utils import trunc_normal_
from scipy import ndimage
import numpy as np
from models.utils import init_proto, update_proto


logger = logging.getLogger(__name__)


class SeparableConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1):
        super(SeparableConv3d, self).__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels)
        self.pointwise = nn.Conv3d(in_channels, out_channels, 1, 1, 0, 1, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


def Norm_layer(norm_cfg, inplanes):
    if norm_cfg == 'BN1':
        out = nn.BatchNorm1d(inplanes)
    elif norm_cfg == 'BN3':
        out = nn.BatchNorm3d(inplanes)
    elif norm_cfg == 'SyncBN':
        out = nn.SyncBatchNorm(inplanes)
    elif norm_cfg == 'GN':
        out = nn.GroupNorm(16, inplanes)
    elif norm_cfg == 'IN2':
        out = nn.InstanceNorm2d(inplanes, affine=True)
    elif norm_cfg == 'IN3':
        out = nn.InstanceNorm3d(inplanes, affine=True)
    elif norm_cfg == 'LN':
        out = nn.LayerNorm(inplanes, eps=1e-6)

    return out


def Activation_layer(activation_cfg, inplace=True):
    if activation_cfg == 'ReLU':
        out = nn.ReLU(inplace=inplace)
    elif activation_cfg == 'LeakyReLU':
        out = nn.LeakyReLU(negative_slope=1e-2, inplace=inplace)

    return out


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 3D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class Conv2dBlock(nn.Module):
    def __init__(self,in_channels,out_channels,norm_cfg,activation_cfg,kernel_size,stride=1,padding=0,dilation=1,bias=False):
        super(Conv2dBlock,self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.norm = Norm_layer(norm_cfg, out_channels)
        self.nonlin = Activation_layer(activation_cfg, inplace=True)
    def forward(self,x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.nonlin(x)
        return x

class Conv3dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg, activation_cfg, kernel_size, stride=(1, 1, 1),
                 padding=(0, 0, 0), dilation=(1, 1, 1), bias=False):
        super(Conv3dBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, bias=bias)
        self.norm = Norm_layer(norm_cfg, out_channels)
        self.nonlin = Activation_layer(activation_cfg, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.nonlin(x)
        return x


class CNNBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, norm_cfg, activation_cfg, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0),
                 dilation=(1, 1, 1), bias=False):
        super(CNNBasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding,
                               dilation=dilation, bias=bias)
        self.norm1 = Norm_layer(norm_cfg, planes)
        self.nonlin = Activation_layer(activation_cfg, inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=kernel_size, stride=1, padding=1, dilation=dilation,
                               bias=bias)
        self.norm2 = Norm_layer(norm_cfg, planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.nonlin(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out += residual
        out = self.nonlin(out)

        return out


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, modal_type=None):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            #     self.sr3D = nn.Conv3d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            
            if modal_type == '2D':
                self.sr2D = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, groups=dim)
            elif modal_type == '3D':
                self.sr3D = nn.Conv3d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, groups=dim)
            elif modal_type == 'MM':
                self.sr2D = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, groups=dim)
                self.sr3D = nn.Conv3d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, groups=dim)

            self.norm = nn.LayerNorm(dim)

    def forward(self, x, dims, modal_type):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        
        is_training = True
        if modal_type == '2D_val':
            modal_type = '2D'
            is_training = False
        elif modal_type == '3D_val':
            modal_type = '3D'
            is_training = False

        if self.sr_ratio > 1:
            if (modal_type == '2D' and len(x.view(-1)) == (B * C * dims[0] * dims[1])) \
               or (modal_type == '3D' and len(x.view(-1)) == (B * C * dims[0] * dims[1] * dims[2])):
                if modal_type == '2D':
                    x_ = x.permute(0, 2, 1).contiguous().reshape(B, C, dims[0], dims[1])
                    x_ = self.sr2D(x_).reshape(B, C, -1).permute(0, 2, 1).contiguous()
                elif modal_type == '3D':
                    x_ = x.permute(0, 2, 1).contiguous().reshape(B, C, dims[0], dims[1], dims[2])
                    x_ = self.sr3D(x_).reshape(B, C, -1).permute(0, 2, 1).contiguous()
            else:
                if modal_type == '2D':
                    x_ = x[:, 1::].permute(0, 2, 1).contiguous().reshape(B, C, dims[0], dims[1])
                    x_ = self.sr2D(x_).reshape(B, C, -1).permute(0, 2, 1).contiguous()
                    x_ = torch.cat((x[:, 0:1], x_), dim=1)
                elif modal_type == '3D':
                    x_ = x[:, 1::].permute(0, 2, 1).contiguous().reshape(B, C, dims[0], dims[1], dims[2])
                    x_ = self.sr3D(x_).reshape(B, C, -1).permute(0, 2, 1).contiguous()
                    x_ = torch.cat((x[:, 0:1], x_), dim=1)

            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, modal_type=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, modal_type=modal_type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, dim, modal_type):
        x = x + self.drop_path(self.attn(self.norm1(x), dim, modal_type))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
        
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)         

    def forward(self, q, k, v):
        B, N, C = q.shape
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, k.shape[1], self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, v.shape[1], self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
       
class CrossBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1_q = norm_layer(dim)
        self.norm1_k = norm_layer(dim)
        self.norm1_v = norm_layer(dim)
        self.attn = CrossAttention(
            dim,
            num_heads=num_heads, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, q, k, v):
        q = q + self.drop_path(self.attn(self.norm1_q(q), self.norm1_k(k), self.norm1_v(v)))
        q = q + self.drop_path(self.mlp(self.norm2(q)))

        return q

       
class Attention_proto(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, modal_type=None,
                 mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            #     self.sr3D = nn.Conv3d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            
            if modal_type == '2D':
                self.sr2D = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, groups=dim)
            elif modal_type == '3D':
                self.sr3D = nn.Conv3d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, groups=dim)
            elif modal_type == 'MM':
                self.sr2D = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, groups=dim)
                self.sr3D = nn.Conv3d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, groups=dim)


            self.norm = nn.LayerNorm(dim)
            
        self.cross_blk = CrossBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qk_scale=qk_scale, 
                                        drop=drop, attn_drop=attn_drop, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer)

    def forward(self, x, dims, modal_type, epoch, p, proto):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        
        is_training = True
        if modal_type == '2D_val':
            modal_type = '2D'
            is_training = False
        elif modal_type == '3D_val':
            modal_type = '3D'
            is_training = False

        if self.sr_ratio > 1:
            if (modal_type == '2D' and len(x.view(-1)) == (B * C * dims[0] * dims[1])) \
               or (modal_type == '3D' and len(x.view(-1)) == (B * C * dims[0] * dims[1] * dims[2])):
                if modal_type == '2D':
                    x_ = x.permute(0, 2, 1).contiguous().reshape(B, C, dims[0], dims[1])
                    x_ = self.sr2D(x_).reshape(B, C, -1).permute(0, 2, 1).contiguous()
                elif modal_type == '3D':
                    x_ = x.permute(0, 2, 1).contiguous().reshape(B, C, dims[0], dims[1], dims[2])
                    x_ = self.sr3D(x_).reshape(B, C, -1).permute(0, 2, 1).contiguous()
            else:
                if modal_type == '2D':
                    x_ = x[:, 1::].permute(0, 2, 1).contiguous().reshape(B, C, dims[0], dims[1])
                    x_ = self.sr2D(x_).reshape(B, C, -1).permute(0, 2, 1).contiguous()
                    x_ = torch.cat((x[:, 0:1], x_), dim=1)
                elif modal_type == '3D':
                    x_ = x[:, 1::].permute(0, 2, 1).contiguous().reshape(B, C, dims[0], dims[1], dims[2])
                    x_ = self.sr3D(x_).reshape(B, C, -1).permute(0, 2, 1).contiguous()
                    x_ = torch.cat((x[:, 0:1], x_), dim=1)

            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k, v = kv[0], kv[1]
        
        # ---------------------update proto of this modality---------------------#
        if is_training and (epoch >= 0) and (p is not None):
            with torch.no_grad():
                xx_ = x.reshape(x.shape[0] * x.shape[1], x.shape[2]).clone()
                if init_proto(p):
                    p = init_proto(p, xx_)
                else:
                    p = update_proto(p, xx_)
        
        # ---------------------use proto of another modality--------------------- #
        start_epoch = 0 
        if ((is_training and (epoch >= start_epoch)) or (not is_training)) and (proto is not None):
            
            proto = proto.view(1, proto.shape[0], proto.shape[1]).repeat(B, 1, 1)
            k_ = k.permute(0, 2, 1, 3).reshape(B, N, C)
            v_ = v.permute(0, 2, 1, 3).reshape(B, N, C) 
            
            proto_ = self.cross_blk(proto, k_, v_).reshape(B, proto.shape[1], self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            
            k = torch.cat((k, proto_), dim=2)
            v = torch.cat((v, proto_), dim=2)
        

        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, p


class Block_proto(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, modal_type=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_proto(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, modal_type=modal_type,
            mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, dim, modal_type, epoch=None, p=None, proto=None):
        attention, p = self.attn(self.norm1(x), dim, modal_type, epoch, p, proto)
        x = x + self.drop_path(attention)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, p


class PatchEmbed2D_en(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, norm_cfg='BN', activation_cfg='ReLU', img_size=[224, 224], patch_size=[16, 16], in_chans=3,
                 embed_dim=768, is_proj1=False):
        super().__init__()
        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.H = (img_size[0] // patch_size[0])
        self.W = (img_size[1] // patch_size[1])
        self.is_proj1 = is_proj1

        self.proj = Conv2dBlock(in_chans, embed_dim, norm_cfg=norm_cfg, activation_cfg=activation_cfg, kernel_size=3, stride=patch_size, padding=1)
        if self.is_proj1:
            self.proj1 = Conv2dBlock(embed_dim, embed_dim, norm_cfg=norm_cfg, activation_cfg=activation_cfg, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        B, C, H, W = x.shape
        if self.is_proj1:
            x = self.proj1(self.proj(x)).flatten(2).transpose(1, 2).contiguous()
        else:
            x = self.proj(x).flatten(2).transpose(1, 2).contiguous()

        return x, (H // self.patch_size[0], W // self.patch_size[1])

class PatchEmbed3D_en(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, norm_cfg='BN', activation_cfg='ReLU', img_size=[16, 96, 96],
                 patch_size=[16, 16, 16], in_chans=1, embed_dim=768, is_proj1=False):
        super().__init__()
        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.D = (img_size[0] // patch_size[0])
        self.H = (img_size[1] // patch_size[1])
        self.W = (img_size[2] // patch_size[2])
        self.is_proj1 = is_proj1

        self.proj = Conv3dBlock(in_chans, embed_dim, norm_cfg=norm_cfg, activation_cfg=activation_cfg, kernel_size=3, stride=patch_size, padding=1)
        if self.is_proj1:
            # self.proj1 = nn.Sequential(Conv3dBlock(embed_dim, embed_dim, norm_cfg=norm_cfg, activation_cfg=activation_cfg, kernel_size=3, stride=1, padding=1),
            #                            Conv3dBlock(embed_dim, embed_dim, norm_cfg=norm_cfg, activation_cfg=activation_cfg, kernel_size=3, stride=1, padding=1))
            self.proj1 = Conv3dBlock(embed_dim, embed_dim, norm_cfg=norm_cfg, activation_cfg=activation_cfg, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        B, C, D, H, W = x.shape
        if self.is_proj1:
            x = self.proj1(self.proj(x)).flatten(2).transpose(1, 2)
        else:
            x = self.proj(x).flatten(2).transpose(1, 2)

        return x, (D // self.patch_size[0], H // self.patch_size[1], W // self.patch_size[1])


class PatchEmbed_de(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, norm_cfg='BN', activation_cfg='ReLU', img_size=[16, 96, 96],
                 patch_size=[16, 16, 16], in_chans=1, embed_dim=768, is_proj1=False):
        super().__init__()
        num_patches = (img_size[0] * patch_size[0]) * (img_size[1] * patch_size[1]) * (img_size[2] * patch_size[2])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.is_proj1 = is_proj1

        self.proj = nn.ConvTranspose3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if self.is_proj1:
            # self.proj1 = nn.Sequential(Conv3dBlock(embed_dim, embed_dim, norm_cfg=norm_cfg, activation_cfg=activation_cfg, kernel_size=3, stride=1, padding=1),
            #                            Conv3dBlock(embed_dim, embed_dim, norm_cfg=norm_cfg, activation_cfg=activation_cfg, kernel_size=3, stride=1, padding=1))
            self.proj1 = Conv3dBlock(embed_dim, embed_dim, norm_cfg=norm_cfg, activation_cfg=activation_cfg, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        B, C, D, H, W = x.shape
        if self.is_proj1:
            x = self.proj1(self.proj(x)).flatten(2).transpose(1, 2)
        else:
            x = self.proj(x).flatten(2).transpose(1, 2)

        return x, (D * self.patch_size[0], H * self.patch_size[1], W * self.patch_size[1])

#----------------------------MiTNet encoder---------------------------------#

class MiT_encoder(nn.Module):
    """ Vision Transformer """

    def __init__(self, norm_cfg2D='BN2', norm_cfg3D='BN3', activation_cfg='ReLU', img_size2D=[224, 224], img_size3D=[16, 96, 96], in_channels=1,  
                 embed_dims=[32, 64, 128, 256, 320, 320], num_heads=[1, 2, 4, 8],
                 mlp_ratios=[8, 8, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[1, 2, 4, 2], sr_ratios=[6, 4, 2, 1], is_proj1=False, num_classes2D=None, modal_type="MM"):

        super().__init__()

        self.embed_dims = embed_dims

        if modal_type == '2D' or modal_type == 'MM':
            self.ConvBlock2D0 = nn.Sequential(Conv2dBlock(3, embed_dims[0], norm_cfg=norm_cfg2D, activation_cfg=activation_cfg, kernel_size=3, stride=1, padding=1))

            self.ConvBlock2D1 = nn.Sequential(Conv2dBlock(embed_dims[0], embed_dims[1], norm_cfg=norm_cfg2D, activation_cfg=activation_cfg, kernel_size=3, stride=2, padding=1),
                                        Conv2dBlock(embed_dims[1], embed_dims[1], norm_cfg=norm_cfg2D, activation_cfg=activation_cfg, kernel_size=3, stride=1, padding=1))

            self.patch_embed2D1 = PatchEmbed2D_en(norm_cfg=norm_cfg2D, activation_cfg=activation_cfg, img_size=[img_size2D[0] // 2, img_size2D[1] // 2], patch_size=[2, 2], 
                                            in_chans=embed_dims[1], embed_dim=embed_dims[2], is_proj1=is_proj1)
            self.patch_embed2D2 = PatchEmbed2D_en(norm_cfg=norm_cfg2D, activation_cfg=activation_cfg, img_size=[img_size2D[0] // 4, img_size2D[1] // 4], patch_size=[2, 2],
                                            in_chans=embed_dims[2], embed_dim=embed_dims[3], is_proj1=is_proj1)
            self.patch_embed2D3 = PatchEmbed2D_en(norm_cfg=norm_cfg2D, activation_cfg=activation_cfg, img_size=[img_size2D[0] // 8, img_size2D[1] // 8], patch_size=[2, 2],
                                            in_chans=embed_dims[3], embed_dim=embed_dims[4], is_proj1=is_proj1)
            self.patch_embed2D4 = PatchEmbed2D_en(norm_cfg=norm_cfg2D, activation_cfg=activation_cfg, img_size=[img_size2D[0] // 16, img_size2D[1] // 16], patch_size=[2, 2],
                                            in_chans=embed_dims[4], embed_dim=embed_dims[5], is_proj1=is_proj1)

            self.pos_embed2D1 = nn.Parameter(torch.zeros(1, self.patch_embed2D1.num_patches + 1, embed_dims[2]))
            self.pos_drop2D1 = nn.Dropout(p=drop_rate)
            self.pos_embed2D2 = nn.Parameter(torch.zeros(1, self.patch_embed2D2.num_patches + 1, embed_dims[3]))
            self.pos_drop2D2 = nn.Dropout(p=drop_rate)
            self.pos_embed2D3 = nn.Parameter(torch.zeros(1, self.patch_embed2D3.num_patches + 1, embed_dims[4]))
            self.pos_drop2D3 = nn.Dropout(p=drop_rate)
            self.pos_embed2D4 = nn.Parameter(torch.zeros(1, self.patch_embed2D4.num_patches + 1, embed_dims[5]))
            self.pos_drop2D4 = nn.Dropout(p=drop_rate)

            trunc_normal_(self.pos_embed2D1, std=.02)
            trunc_normal_(self.pos_embed2D2, std=.02)
            trunc_normal_(self.pos_embed2D3, std=.02)
            trunc_normal_(self.pos_embed2D4, std=.02)
            
        if modal_type == '3D' or modal_type == 'MM':
            # self.patch_embed3D0 = Conv3dBlock(1, 32, norm_cfg=norm_cfg3D, activation_cfg=activation_cfg, kernel_size=7, stride=(1, 2, 2), padding=3)
            # self.ConvBlock0 = nn.Sequential(Conv3dBlock(in_channels, embed_dims[0], norm_cfg=norm_cfg3D, activation_cfg=activation_cfg, kernel_size=3, stride=(1, 1, 1), padding=1),
            #                                 Conv3dBlock(embed_dims[0], embed_dims[0], norm_cfg=norm_cfg3D, activation_cfg=activation_cfg, kernel_size=3, stride=(1, 1, 1), padding=1))
            self.ConvBlock3D0 = nn.Sequential(Conv3dBlock(in_channels, embed_dims[0], norm_cfg=norm_cfg3D, activation_cfg=activation_cfg, kernel_size=3, stride=(1, 1, 1), padding=1))

            self.ConvBlock3D1 = nn.Sequential(Conv3dBlock(embed_dims[0], embed_dims[1], norm_cfg=norm_cfg3D, activation_cfg=activation_cfg, kernel_size=3, stride=(1, 2, 2), padding=1),
                                        Conv3dBlock(embed_dims[1], embed_dims[1], norm_cfg=norm_cfg3D, activation_cfg=activation_cfg, kernel_size=3, stride=(1, 1, 1), padding=1))

            self.patch_embed3D1 = PatchEmbed3D_en(norm_cfg=norm_cfg3D, activation_cfg=activation_cfg,
                                            img_size=[img_size3D[0] // 1, img_size3D[1] // 2, img_size3D[2] // 2],
                                            patch_size=[2, 2, 2], in_chans=embed_dims[1], embed_dim=embed_dims[2],
                                            is_proj1=is_proj1)
            self.patch_embed3D2 = PatchEmbed3D_en(norm_cfg=norm_cfg3D, activation_cfg=activation_cfg,
                                            img_size=[img_size3D[0] // 2, img_size3D[1] // 4, img_size3D[2] // 4],
                                            patch_size=[2, 2, 2], in_chans=embed_dims[2], embed_dim=embed_dims[3],
                                            is_proj1=is_proj1)
            self.patch_embed3D3 = PatchEmbed3D_en(norm_cfg=norm_cfg3D, activation_cfg=activation_cfg,
                                            img_size=[img_size3D[0] // 4, img_size3D[1] // 8, img_size3D[2] // 8],
                                            patch_size=[1, 2, 2], in_chans=embed_dims[3], embed_dim=embed_dims[4],
                                            is_proj1=is_proj1)
            self.patch_embed3D4 = PatchEmbed3D_en(norm_cfg=norm_cfg3D, activation_cfg=activation_cfg,
                                            img_size=[img_size3D[0] // 4, img_size3D[1] // 16, img_size3D[2] // 16],
                                            patch_size=[1, 2, 2], in_chans=embed_dims[4], embed_dim=embed_dims[5],
                                            is_proj1=is_proj1)

            self.pos_embed3D1 = nn.Parameter(torch.zeros(1, self.patch_embed3D1.num_patches, embed_dims[2]))
            self.pos_drop3D1 = nn.Dropout(p=drop_rate)
            self.pos_embed3D2 = nn.Parameter(torch.zeros(1, self.patch_embed3D2.num_patches, embed_dims[3]))
            self.pos_drop3D2 = nn.Dropout(p=drop_rate)
            self.pos_embed3D3 = nn.Parameter(torch.zeros(1, self.patch_embed3D3.num_patches, embed_dims[4]))
            self.pos_drop3D3 = nn.Dropout(p=drop_rate)
            self.pos_embed3D4 = nn.Parameter(torch.zeros(1, self.patch_embed3D4.num_patches, embed_dims[5]))
            self.pos_drop3D4 = nn.Dropout(p=drop_rate)

            trunc_normal_(self.pos_embed3D1, std=.02)
            trunc_normal_(self.pos_embed3D2, std=.02)
            trunc_normal_(self.pos_embed3D3, std=.02)
            trunc_normal_(self.pos_embed3D4, std=.02)

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0], modal_type=modal_type)
            for i in range(depths[0])])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1], modal_type=modal_type)
            for i in range(depths[1])])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[4], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2], modal_type=modal_type)
            for i in range(depths[2])])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block_proto(
            dim=embed_dims[5], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3], modal_type=modal_type)
            for i in range(depths[3])])
            
        if modal_type == '2D' or modal_type == 'MM':
            self.cls_tokens1 = nn.Parameter(torch.zeros(1, 1, embed_dims[2]))
            self.cls_tokens2 = nn.Linear(embed_dims[2], embed_dims[3])
            self.cls_tokens3 = nn.Linear(embed_dims[3], embed_dims[4])
            self.cls_tokens4 = nn.Linear(embed_dims[4], embed_dims[5])

            trunc_normal_(self.cls_tokens1, std=.02)

            # Classifier head
            self.head_fc = nn.Linear(embed_dims[-1], num_classes2D)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv3d, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d, nn.BatchNorm2d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm3d, nn.InstanceNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding2D(self, pos_embed, x, w, h):
        npatch = x.shape[1] - 1
        N = pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return pos_embed
        class_pos_embed = self.position_embeddings[:, 0]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = x.shape[-1]
        h0 = h
        w0 = w
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        h0, w0 = h0 + 0.1, w0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(h0 / math.sqrt(N), w0 / math.sqrt(N)),
            mode="bicubic",
            align_corners=False,
        )
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).contiguous().view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
    
    def interpolate_pos_encoding3D(self, pos_embed, x, d, w, h, patch_embed):
        npatch = x.shape[1] - 1
        N = pos_embed.shape[1] - 1
        if npatch == N:
            return pos_embed
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        d_ds = d * 3

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, d_ds, int(math.sqrt(N / d_ds)), int(math.sqrt(N / d_ds)), dim).permute(0, 4, 1,
                                                                                                              2, 3),
            scale_factor=(d / d_ds, w / math.sqrt(N / d_ds), h / math.sqrt(N / d_ds)),
            mode='trilinear',
        )
        assert int(d) == patch_pos_embed.shape[-3] and int(w) == patch_pos_embed.shape[-2] and int(h) == \
               patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 4, 1).contiguous().view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
        
    def forward2D(self, x, proto, proto_ano, modal_type, epoch):
        B = x.shape[0]

        x = self.ConvBlock2D0(x)
        #print('2D', x.shape)
        x = self.ConvBlock2D1(x)
        #print('2D', x.shape)

        # stage 1
        x, (H, W) = self.patch_embed2D1(x)
        cls_tokens = self.cls_tokens1.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding2D(self.pos_embed2D1, x, W, H)
        x = self.pos_drop2D1(x)
        for blk in self.block1:
            x = blk(x, (H, W), modal_type)
        cls_tokens = x[:, 0]
        x = x[:, 1::].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        #print('2D', "s1", x.shape)

        # stage 2
        x, (H, W) = self.patch_embed2D2(x)
        cls_tokens = self.cls_tokens2(cls_tokens)
        x = torch.cat((cls_tokens.unsqueeze(1), x), dim=1)
        x = x + self.interpolate_pos_encoding2D(self.pos_embed2D2, x, W, H)
        x = self.pos_drop2D2(x)
        for blk in self.block2:
            x = blk(x, (H, W), modal_type)
        cls_tokens = x[:, 0]
        x = x[:, 1::].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        #print('2D', "s2", x.shape)

        # stage 3
        x, (H, W) = self.patch_embed2D3(x)
        cls_tokens = self.cls_tokens3(cls_tokens)
        x = torch.cat((cls_tokens.unsqueeze(1), x), dim=1)
        x = x + self.interpolate_pos_encoding2D(self.pos_embed2D3, x, W, H)
        x = self.pos_drop2D3(x)
        for blk in self.block3:
            x = blk(x, (H, W), modal_type)
        cls_tokens = x[:, 0]
        x = x[:, 1::].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        #print('2D', "s3", x.shape)

        # stage 4
        x, (H, W) = self.patch_embed2D4(x)
        cls_tokens = self.cls_tokens4(cls_tokens)
        x = torch.cat((cls_tokens.unsqueeze(1), x), dim=1)
        x = x + self.interpolate_pos_encoding2D(self.pos_embed2D4, x, W, H)
        x = self.pos_drop2D4(x)
        for blk in self.block4:
            if proto_ano.mean() == 0.:
                print('Init', modal_type, epoch)
                x, proto = blk(x, (H, W), modal_type, epoch, proto)
            else:
                x, proto = blk(x, (H, W), modal_type, epoch, proto, proto_ano)
        #print('2D', "s4", x.shape)

        out_cls = self.head_fc(x[:, 0])
        return out_cls, proto


    def forward3D(self, x, proto, proto_ano, modal_type, epoch):
        out = []

        B = x.shape[0]

        x = self.ConvBlock3D0(x)
        #print('3D', x.shape)
        out.append(x)

        x = self.ConvBlock3D1(x)
        #print('3D', x.shape)
        out.append(x)

        # stage 1
        x, (D, H, W) = self.patch_embed3D1(x)
        #print("s1-1", x.shape)
        x = x + self.interpolate_pos_encoding3D(self.pos_embed3D1, x, D, W, H, self.patch_embed3D1)
        x = self.pos_drop3D1(x)
        for blk in self.block1:
            x = blk(x, (D, H, W), modal_type)
        x = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        out.append(x)
        #print('3D', "s1", x.shape)

        # stage 2
        x, (D, H, W) = self.patch_embed3D2(x)
        #print("s2-1", x.shape)
        x = x + self.interpolate_pos_encoding3D(self.pos_embed3D2, x, D, W, H, self.patch_embed3D2)
        x = self.pos_drop3D2(x)
        for blk in self.block2:
            x = blk(x, (D, H, W), modal_type)
        x = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        out.append(x)
        #print('3D', "s2", x.shape)

        # stage 3
        x, (D, H, W) = self.patch_embed3D3(x)
        #print("s3-1", x.shape)
        x = x + self.interpolate_pos_encoding3D(self.pos_embed3D3, x, D, W, H, self.patch_embed3D3)
        x = self.pos_drop3D3(x)
        for blk in self.block3:
            x = blk(x, (D, H, W), modal_type)
        x = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        out.append(x)
        #print('3D', "s3", x.shape)

        # stage 4
        x, (D, H, W) = self.patch_embed3D4(x)
        #print("s4-1", x.shape)
        x = x + self.interpolate_pos_encoding3D(self.pos_embed3D4, x, D, W, H, self.patch_embed3D4)
        x = self.pos_drop3D4(x)
        for blk in self.block4:
            if proto_ano.mean() == 0.:
                print('Init', modal_type, epoch)
                x, proto = blk(x, (D, H, W), modal_type, epoch, proto)
            else:
                x, proto = blk(x, (D, H, W), modal_type, epoch, proto, proto_ano)
        #print('3D', "s4", x.shape)

        out.append(x)

        return out, (D, H, W), proto
        
    
    def forward(self, x, proto, proto_ano, modal_type, epoch):

        if modal_type == '2D' or modal_type == '2D_val':
            return self.forward2D(x, proto, proto_ano, modal_type, epoch)
        elif modal_type == '3D' or modal_type == '3D_val':
            return self.forward3D(x, proto, proto_ano, modal_type, epoch)
        assert 0

#----------------------------MiTNet---------------------------------#

class MiTnet(nn.Module):
    """ Vision Transformer """

    def __init__(self, norm_cfg2D='BN2', norm_cfg3D='BN3', activation_cfg='ReLU', img_size2D=[224, 224], img_size3D=[48, 192, 192], num_classes2D=None, num_classes3D=None,
                 embed_dims=[320, 256, 128, 64, 32], num_heads=[4, 2, 1], mlp_ratios=[4, 8, 8], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[1, 1, 1], sr_ratios=[2, 4, 6], encoder='mediumv7',
                 is_proj1=False, modal_type="MM"):

        super().__init__()
        self.MODEL_NUM_CLASSES = num_classes3D
        self.embed_dims = embed_dims
        self.embed_dim = embed_dims[2]

        # Transformer encoder
        if encoder == 'mediumv7':
            self.transformer = MiT_encoder_mediumv7(norm_cfg2D=norm_cfg2D, norm_cfg3D=norm_cfg3D, activation_cfg=activation_cfg, img_size2D=img_size2D, img_size3D=img_size3D, is_proj1=True, num_classes2D=num_classes2D, modal_type=modal_type)

        total = sum([param.nelement() for param in self.transformer.parameters()])
        print('  + Number of Transformer Encoder Layers: %.f and Params: %.2f(e6)' % (len(self.transformer.state_dict()), total / 1e6))

        if modal_type == '3D' or modal_type == 'MM':
            # upsampling
            # self.upsamplex122 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear')

            self.DecEmbed3D1 = PatchEmbed_de(norm_cfg=norm_cfg3D, activation_cfg=activation_cfg, img_size=[img_size3D[0] // 4, img_size3D[1] // 16, img_size3D[2] // 16],
                                         patch_size=[1, 2, 2], in_chans=self.transformer.embed_dims[-1],
                                         embed_dim=embed_dims[0], is_proj1=is_proj1)
            self.DecEmbed3D2 = PatchEmbed_de(norm_cfg=norm_cfg3D, activation_cfg=activation_cfg, img_size=[img_size3D[0] // 4, img_size3D[1] // 8, img_size3D[2] // 8],
                                         patch_size=[1, 2, 2], in_chans=embed_dims[0], embed_dim=embed_dims[1],
                                         is_proj1=is_proj1)
            self.DecEmbed3D3 = PatchEmbed_de(norm_cfg=norm_cfg3D, activation_cfg=activation_cfg, img_size=[img_size3D[0] // 2, img_size3D[1] // 4, img_size3D[2] // 4],
                                         patch_size=[2, 2, 2], in_chans=embed_dims[1], embed_dim=embed_dims[2],
                                         is_proj1=is_proj1)

            # pos_embed
            self.DecPosEmbed3D1 = nn.Parameter(torch.zeros(1, self.transformer.patch_embed3D3.num_patches, embed_dims[0]))
            self.DecPosDrop3D1 = nn.Dropout(p=drop_rate)
            self.DecPosEmbed3D2 = nn.Parameter(torch.zeros(1, self.transformer.patch_embed3D2.num_patches, embed_dims[1]))
            self.DecPosDrop3D2 = nn.Dropout(p=drop_rate)
            self.DecPosEmbed3D3 = nn.Parameter(torch.zeros(1, self.transformer.patch_embed3D1.num_patches, embed_dims[2]))
            self.DecPosDrop3D3 = nn.Dropout(p=drop_rate)

            trunc_normal_(self.DecPosEmbed3D1, std=.02)
            trunc_normal_(self.DecPosEmbed3D2, std=.02)
            trunc_normal_(self.DecPosEmbed3D3, std=.02)

            # transformer encoder
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
            cur = 0
            self.Decblock1 = nn.ModuleList([Block(
                dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
                sr_ratio=sr_ratios[0], modal_type='3D')
                for i in range(depths[0])])

            cur += depths[0]
            self.Decblock2 = nn.ModuleList([Block(
                dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
                sr_ratio=sr_ratios[1], modal_type='3D')
                for i in range(depths[1])])

            cur += depths[1]
            self.Decblock3 = nn.ModuleList([Block(
                dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
                sr_ratio=sr_ratios[2], modal_type='3D')
                for i in range(depths[2])])

            self.norm = norm_layer(embed_dims[2])

            # Seg head
            self.TransposeConv3D1 = nn.ConvTranspose3d(embed_dims[2], embed_dims[3], kernel_size=2, stride=2)
            self.DeConvBlock3D1 = Conv3dBlock(embed_dims[3], embed_dims[3], norm_cfg=norm_cfg3D, activation_cfg=activation_cfg, kernel_size=3, stride=1, padding=1)

            self.TransposeConv3D2 = nn.ConvTranspose3d(embed_dims[3], embed_dims[4], kernel_size=(1, 2, 2), stride=(1, 2, 2))
            # self.DeConvBlock3D2 = Conv3dBlock(embed_dims[4], embed_dims[4], norm_cfg=norm_cfg3D, activation_cfg=activation_cfg, kernel_size=3, stride=1, padding=1)

            self.ds1_cls_conv = nn.Conv3d(embed_dims[0], self.MODEL_NUM_CLASSES, kernel_size=1)
            self.ds2_cls_conv = nn.Conv3d(embed_dims[1], self.MODEL_NUM_CLASSES, kernel_size=1)
            self.ds3_cls_conv = nn.Conv3d(embed_dims[2], self.MODEL_NUM_CLASSES, kernel_size=1)
            self.ds4_cls_conv = nn.Conv3d(embed_dims[3], self.MODEL_NUM_CLASSES, kernel_size=1)
            self.ds5_cls_conv = nn.Conv3d(embed_dims[4], self.MODEL_NUM_CLASSES, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv3d, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d, nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm3d, nn.InstanceNorm2d, nn.SyncBatchNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding3D(self, pos_embed, x, d, w, h, patch_embed):
        npatch = x.shape[1] - 1
        N = pos_embed.shape[1] - 1
        if npatch == N:
            return pos_embed
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        d_ds = d * 3

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, d_ds, int(math.sqrt(N / d_ds)), int(math.sqrt(N / d_ds)), dim).permute(0, 4, 1, 2, 3),
            scale_factor=(d / d_ds, w / math.sqrt(N / d_ds), h / math.sqrt(N / d_ds)),
            mode='trilinear',
        )
        assert int(d) == patch_pos_embed.shape[-3] and int(w) == patch_pos_embed.shape[-2] and int(h) == \
               patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 4, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward2D(self, inputs, proto, proto_ano, modal_type, epoch):

        logits, proto = self.transformer(inputs, proto, proto_ano, modal_type, epoch)
        
        return logits, proto

    def forward3D(self, inputs, proto, proto_ano, modal_type, epoch):

        B = inputs.shape[0]

        x_encoder_3D, (D, H, W), proto = self.transformer(inputs, proto, proto_ano, modal_type, epoch)
        x_trans = x_encoder_3D[-1].reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        #print('3D', "de", x_trans.shape)

        ####### decoder
        # stage 0
        x, (D, H, W) = self.DecEmbed3D1(x_trans)
        x = x + x_encoder_3D[-2].flatten(2).transpose(1, 2)
        x = x + self.interpolate_pos_encoding3D(self.DecPosEmbed3D1, x, D, W, H, self.DecEmbed3D1)
        x = self.DecPosDrop3D1(x)
        for blk in self.Decblock1:
            x = blk(x, (D, H, W), modal_type)
        x = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        #print('3D', "de0", x.shape)
        ds1= self.ds1_cls_conv(x)

        # stage 1
        x, (D, H, W) = self.DecEmbed3D2(x)
        x = x + x_encoder_3D[-3].flatten(2).transpose(1, 2)
        x = x + self.interpolate_pos_encoding3D(self.DecPosEmbed3D2, x, D, W, H, self.DecEmbed3D2)
        x = self.DecPosDrop3D2(x)
        for blk in self.Decblock2:
            x = blk(x, (D, H, W), modal_type)
        x = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        #print('3D', "de1", x.shape)
        ds2 = self.ds2_cls_conv(x)

        # stage 2
        x, (D, H, W) = self.DecEmbed3D3(x)
        x = x + x_encoder_3D[-4].flatten(2).transpose(1, 2)
        x = x + self.interpolate_pos_encoding3D(self.DecPosEmbed3D3, x, D, W, H, self.DecEmbed3D3)
        x = self.DecPosDrop3D3(x)
        for blk in self.Decblock3:
            x = blk(x, (D, H, W), modal_type)
        x = self.norm(x)
        x = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        #print('3D', "de2", x.shape)
        ds3 = self.ds3_cls_conv(x)

        # stage 3
        x = self.TransposeConv3D1(x)
        skip3 = x_encoder_3D[-5]
        x = x + skip3
        x = self.DeConvBlock3D1(x)
        #print('3D', "de3", x.shape)
        ds4 = self.ds4_cls_conv(x)


        # stage 4
        x = self.TransposeConv3D2(x)
        skip4 = x_encoder_3D[-6]
        x = x + skip4
        # x = self.DeConvBlock2(x)
        ds5 = self.ds5_cls_conv(x)
        #print('3D', "de4", ds5.shape)

        # print(ds5.shape)
        # print(ds4.shape)
        # print(ds3.shape)
        # print(ds2.shape)
        # print(ds1.shape)

        return [ds5, ds4, ds3, ds2, ds1], proto
        
    def forward(self, x, proto, proto_ano, modal_type, epoch):

        if modal_type == '2D' or modal_type == '2D_val':
            return self.forward2D(x, proto, proto_ano, modal_type, epoch)
        elif modal_type == '3D' or modal_type == '3D_val':
            return self.forward3D(x, proto, proto_ano, modal_type, epoch)



class MiTNET(nn.Module):
    """
    MiTNET
    """

    def __init__(self, arch, norm2D='BN2', norm3D='BN3', act='ReLU', img_size2D=[224, 224], img_size3D=[48, 192, 192], num_classes2D=None, num_classes3D=None, pretrain=False, pretrain_path=None, modal_type="MM"):
        super().__init__()
        if arch == 'mediumv7':
            self.model = MiTnet(norm_cfg2D=norm2D, norm_cfg3D=norm3D, activation_cfg=act, img_size2D=img_size2D, img_size3D=img_size3D,
                                num_classes2D=num_classes2D, num_classes3D=num_classes3D, embed_dims=[320, 256, 128, 64, 32],
                                num_heads=[4, 2, 1], depths=[1, 1, 1], mlp_ratios=[4, 4, 4], sr_ratios=[2, 4, 8],
                                encoder='mediumv7', is_proj1=True, modal_type=modal_type)


        total = sum([param.nelement() for param in self.model.parameters()])
        print('  + Number of Total Params: %.2f(e6)' % (total / 1e6))
     
        if pretrain:
            pre_type = 'student'  #teacher student
            print('*********loading from checkpoint: {}'.format(pretrain_path))
            pre_dict_ori = torch.load(pretrain_path, map_location="cpu")[pre_type]

            if 'En' in pretrain_path:
                pre_dict_ori = {k.replace("module.backbone.", "transformer."): v for k, v in pre_dict_ori.items()}
                pre_dict_ori = {k.replace("module.head.", "head."): v for k, v in pre_dict_ori.items()}

                pre_dict_encoder = pre_dict_ori.copy()   
                for k, v in pre_dict_ori.items():
                    if ('transformer' not in k) or ('2D' in k):
                        del pre_dict_encoder[k]

                print('Student: length of pre-trained layers: %.f' % (len(pre_dict_encoder)))

                model_dict = self.model.state_dict()
                print('length of new layers: %.f' % (len(model_dict)))
                print('before loading weights: %.12f' % (self.model.state_dict()['transformer.block1.0.mlp.fc1.weight'].mean()))

                # Patch_embeddings
                print('Patch_embeddings layer1 weights: %.12f' % (self.model.state_dict()['transformer.patch_embed3D1.proj.conv.weight'].mean()))
                print('Patch_embeddings layer2 weights: %.12f' % (self.model.state_dict()['transformer.patch_embed3D2.proj.conv.weight'].mean()))

                # Position_embeddings
                print('Position_embeddings weights: %.12f' % (self.model.transformer.pos_embed3D1.data.mean()))

                for k, v in pre_dict_encoder.items():
                    if 'transformer.pos_embed3D' in k: 
                        posemb = pre_dict_encoder[k]
                        posemb_new = model_dict[k]                        

                        if posemb.size() == posemb_new.size():
                            print(k+'layer is matched')
                            pre_dict_encoder[k] = posemb
                        else:
                            ntok_new = posemb_new.size(1)
                            posemb_zoom = ndimage.zoom(posemb[0], (ntok_new / posemb.size(1), 1), order=1)
                            posemb_zoom = np.expand_dims(posemb_zoom, 0)
                            pre_dict_encoder[k] = torch.from_numpy(posemb_zoom)

                pre_dict = {k: v for k, v in pre_dict_encoder.items() if k in model_dict}
                print('length of matched layers: %.f' % (len(pre_dict)))

                # Update weigts
                model_dict.update(pre_dict)
                self.model.load_state_dict(model_dict)
                print('after loading weights: %.12f' % (self.model.state_dict()['transformer.block1.0.mlp.fc1.weight'].mean()))
                print('Patch_embeddings layer1 pretrained weights: %.12f' % (self.model.state_dict()['transformer.patch_embed3D1.proj.conv.weight'].mean()))
                print('Patch_embeddings layer2 pretrained weights: %.12f' % (self.model.state_dict()['transformer.patch_embed3D2.proj.conv.weight'].mean()))
                print('Position_embeddings pretrained weights: %.12f' % (self.model.transformer.pos_embed3D1.data.mean()))

            elif 'ED' in pretrain_path:
                pre_dict_ori = {k.replace("module.backbone.", ""): v for k, v in pre_dict_ori.items()}
                pre_dict_ori = {k.replace("module.head.", "head."): v for k, v in pre_dict_ori.items()}
                
                print('Student: length of pre-trained layers: %.f' % (len(pre_dict_ori)))

                model_dict = self.model.state_dict()
                print('length of new layers: %.f' % (len(model_dict)))
                print('before loading weights: %.12f' % (self.model.state_dict()['transformer.block1.0.mlp.fc1.weight'].mean()))

                # Patch_embeddings
                print('Patch_embeddings layer1 weights: %.12f' % (self.model.state_dict()['transformer.patch_embed3D1.proj.conv.weight'].mean()))
                print('Patch_embeddings layer2 weights: %.12f' % (self.model.state_dict()['transformer.patch_embed3D2.proj.conv.weight'].mean()))

                # Position_embeddings
                print('Position_embeddings weights: %.12f' % (self.model.transformer.pos_embed3D1.data.mean()))

                for k, v in pre_dict_ori.items():
                    if ('transformer.pos_embed3D' in k) or ('DecPosEmbed3D' in k):
                        posemb = pre_dict_ori[k]
                        posemb_new = model_dict[k]

                        if posemb.size() == posemb_new.size():
                            print(k+'layer is matched')
                            pre_dict_ori[k] = posemb
                        else:
                            print(k+' layer is not matched', posemb.size(), "->", posemb_new.size())
                            ntok_new = posemb_new.size(1)
                            posemb_zoom = ndimage.zoom(posemb[0], (ntok_new / posemb.size(1), 1), order=1)
                            posemb_zoom = np.expand_dims(posemb_zoom, 0)
                            pre_dict_ori[k] = torch.from_numpy(posemb_zoom)
                            
                    if 'transformer.pos_embed2D' in k:
                        posemb = pre_dict_ori[k]
                        posemb_new = model_dict[k]

                        if posemb.size() == posemb_new.size():
                            print(k+'layer is matched')
                            pre_dict_ori[k] = posemb
                        else:
                            print(k+' layer is not matched', posemb.size(), "->", posemb_new.size())
                            ntok_new = posemb_new.size(1)
                            posemb_zoom = ndimage.zoom(posemb[0], (ntok_new / posemb.size(1), 1), order=1)
                            posemb_zoom = np.expand_dims(posemb_zoom, 0)
                            pre_dict_ori[k] = torch.from_numpy(posemb_zoom)

                pre_dict = {k: v for k, v in pre_dict_ori.items() if k in model_dict}
                print('length of matched layers: %.f' % (len(pre_dict)))

                # Update weigts
                model_dict.update(pre_dict)
                self.model.load_state_dict(model_dict)
                print('after loading weights: %.12f' % (self.model.state_dict()['transformer.block1.0.mlp.fc1.weight'].mean()))
                print('Patch_embeddings layer1 pretrained weights: %.12f' % (self.model.state_dict()['transformer.patch_embed3D1.proj.conv.weight'].mean()))
                print('Patch_embeddings layer2 pretrained weights: %.12f' % (self.model.state_dict()['transformer.patch_embed3D2.proj.conv.weight'].mean()))
                print('Position_embeddings pretrained weights: %.12f' % (self.model.transformer.pos_embed3D1.data.mean()))


    def forward(self, x, proto=None, proto_ano=None, modal_type='3D', epoch=0):
        output, proto = self.model(x, proto, proto_ano, modal_type, epoch)
        return output, proto
        


def MiT_encoder_mediumv7(norm_cfg2D='BN2', norm_cfg3D='BN3', activation_cfg='ReLU', img_size2D=[224, 224], img_size3D=[16, 96, 96], is_proj1=False, **kwargs):
    model = MiT_encoder(norm_cfg2D=norm_cfg2D, norm_cfg3D=norm_cfg3D, activation_cfg=activation_cfg, img_size2D=img_size2D, img_size3D=img_size3D, in_channels=1, 
                        embed_dims=[32, 64, 128, 256, 320, 320], depths=[1, 2, 4, 2],
                        num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4],
                        sr_ratios=[8, 4, 2, 1], qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        is_proj1=is_proj1, **kwargs)
    return model