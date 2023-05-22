# All rights reserved.
from math import ceil, sqrt
from collections import OrderedDict
import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import _cfg
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from .build import MODEL_REGISTRY
import os

import slowfast.utils.logging as logging

logger = logging.get_logger(__name__)

global_attn = None
token_indices = None

model_path = 'path_to_models'
model_path = {
    'uniformer_xxs_128_in1k': os.path.join(model_path, 'uniformer_xxs_128_in1k.pth'),
    'uniformer_xxs_160_in1k': os.path.join(model_path, 'uniformer_xxs_160_in1k.pth'),
    'uniformer_xxs_192_in1k': os.path.join(model_path, 'uniformer_xxs_192_in1k.pth'),
    'uniformer_xxs_224_in1k': os.path.join(model_path, 'uniformer_xxs_224_in1k.pth'),
    'uniformer_xs_192_in1k': os.path.join(model_path, 'uniformer_xs_192_in1k.pth'),
    'uniformer_xs_224_in1k': os.path.join(model_path, 'uniformer_xs_224_in1k.pth'),
}


def conv_3xnxn(inp, oup, kernel_size=3, stride=3, groups=1):
    return nn.Conv3d(inp, oup, (3, kernel_size, kernel_size), (2, stride, stride), (1, 0, 0), groups=groups)
    
def conv_1xnxn(inp, oup, kernel_size=3, stride=3, groups=1):
    return nn.Conv3d(inp, oup, (1, kernel_size, kernel_size), (1, stride, stride), (0, 0, 0), groups=groups)

def conv_3xnxn_std(inp, oup, kernel_size=3, stride=3, groups=1):
    return nn.Conv3d(inp, oup, (3, kernel_size, kernel_size), (1, stride, stride), (1, 0, 0), groups=groups)

def conv_1x1x1(inp, oup, groups=1):
    return nn.Conv3d(inp, oup, (1, 1, 1), (1, 1, 1), (0, 0, 0), groups=groups)

def conv_3x3x3(inp, oup, groups=1):
    return nn.Conv3d(inp, oup, (3, 3, 3), (1, 1, 1), (1, 1, 1), groups=groups)

def conv_5x5x5(inp, oup, groups=1):
    return nn.Conv3d(inp, oup, (5, 5, 5), (1, 1, 1), (2, 2, 2), groups=groups)

def bn_3d(dim):
    return nn.BatchNorm3d(dim)


# code is from https://github.com/YifanXu74/Evo-ViT
def easy_gather(x, indices):
    # x => B x N x C
    # indices => B x N
    B, N, C = x.shape
    N_new = indices.shape[1]
    offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
    indices = indices + offset
    # only select the informative tokens
    out = x.reshape(B * N, C)[indices.view(-1)].reshape(B, N_new, C)
    return out


# code is from https://github.com/YifanXu74/Evo-ViT
def merge_tokens(x_drop, score):
    # x_drop => B x N_drop
    # score => B x N_drop
    weight = score / torch.sum(score, dim=1, keepdim=True)
    x_drop = weight.unsqueeze(-1) * x_drop
    return torch.sum(x_drop, dim=1, keepdim=True)


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
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., trade_off=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # updating weight for global score
        self.trade_off = trade_off

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # update global score
        global global_attn
        tradeoff = self.trade_off
        if isinstance(global_attn, int):
            global_attn = torch.mean(attn[:, :, 0, 1:], dim=1)
        elif global_attn.shape[1] == N - 1:
            # no additional token and no pruning, update all global scores
            cls_attn = torch.mean(attn[:, :, 0, 1:], dim=1)
            global_attn = (1 - tradeoff) * global_attn + tradeoff * cls_attn
        else:
            # only update the informative tokens
            # the first one is class token
            # the last one is rrepresentative token
            cls_attn = torch.mean(attn[:, :, 0, 1:-1], dim=1)
            if self.training:
                temp_attn = (1 - tradeoff) * global_attn[:, :(N - 2)] + tradeoff * cls_attn
                global_attn = torch.cat((temp_attn, global_attn[:, (N - 2):]), dim=1)
            else:
                # no use torch.cat() for fast inference
                global_attn[:, :(N - 2)] = (1 - tradeoff) * global_attn[:, :(N - 2)] + tradeoff * cls_attn

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = conv_1x1x1(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = conv_1x1x1(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
    
class CBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = conv_3x3x3(dim, dim, groups=dim)
        self.norm1 = bn_3d(dim)
        self.conv1 = conv_1x1x1(dim, dim, 1)
        self.conv2 = conv_1x1x1(dim, dim, 1)
        self.attn = conv_5x5x5(dim, dim, groups=dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = bn_3d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x   


class EvoSABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, prune_ratio=1,
                 trade_off=0, downsample=False):
        super().__init__()
        self.pos_embed = conv_3x3x3(dim, dim, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, trade_off=trade_off)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.prune_ratio = prune_ratio
        self.downsample = downsample
        if downsample:
            self.avgpool = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

    def forward(self, cls_token, x):
        x = x + self.pos_embed(x)
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  

        if self.prune_ratio == 1:
            x = torch.cat([cls_token, x], dim=1)
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            cls_token, x = x[:, :1], x[:, 1:]
            x = x.transpose(1, 2).reshape(B, C, T, H, W)
            return cls_token, x  
        else:
            global global_attn, token_indices
            # calculate the number of informative tokens
            N = x.shape[1]
            N_ = int(N * self.prune_ratio)
            # sort global attention
            indices = torch.argsort(global_attn, dim=1, descending=True)

            # concatenate x, global attention and token indices => x_ga_ti
            # rearrange the tensor according to new indices 
            x_ga_ti = torch.cat((x, global_attn.unsqueeze(-1), token_indices.unsqueeze(-1)), dim=-1)
            x_ga_ti = easy_gather(x_ga_ti, indices)
            x_sorted, global_attn, token_indices = x_ga_ti[:, :, :-2], x_ga_ti[:, :, -2], x_ga_ti[:, :, -1]

            # informative tokens
            x_info = x_sorted[:, :N_]
            # merge dropped tokens
            x_drop = x_sorted[:, N_:]
            score = global_attn[:, N_:]
            #  B x N_drop x C => B x 1 x C
            rep_token = merge_tokens(x_drop, score)
            # concatenate new tokens
            x = torch.cat((cls_token, x_info, rep_token), dim=1)

            # slow update
            fast_update = 0
            tmp_x = self.attn(self.norm1(x))
            fast_update = fast_update + tmp_x[:, -1:]
            x = x + self.drop_path(tmp_x)
            tmp_x = self.mlp(self.norm2(x))
            fast_update = fast_update + tmp_x[:, -1:]
            x = x + self.drop_path(tmp_x)
            # fast update
            x_drop = x_drop + fast_update.expand(-1, N - N_, -1)

            cls_token, x = x[:, :1, :], x[:, 1:-1, :]
            if self.training:
                x_sorted = torch.cat((x, x_drop), dim=1)
            else:
                x_sorted[:, N_:] = x_drop
                x_sorted[:, :N_] = x

            # recover token
            # scale for normalization
            old_global_scale = torch.sum(global_attn, dim=1, keepdim=True)
            # recover order
            indices = torch.argsort(token_indices, dim=1)
            x_ga_ti = torch.cat((x_sorted, global_attn.unsqueeze(-1), token_indices.unsqueeze(-1)), dim=-1)
            x_ga_ti = easy_gather(x_ga_ti, indices)
            x_patch, global_attn, token_indices = x_ga_ti[:, :, :-2], x_ga_ti[:, :, -2], x_ga_ti[:, :, -1]
            x_patch = x_patch.transpose(1, 2).reshape(B, C, T, H, W)

            if self.downsample:
                # downsample global attention
                global_attn = global_attn.reshape(B, 1, T, H, W)
                global_attn = self.avgpool(global_attn).view(B, -1)
                # normalize global attention
                new_global_scale = torch.sum(global_attn, dim=1, keepdim=True)
                scale = old_global_scale / new_global_scale
                global_attn = global_attn * scale
            
            return cls_token, x_patch


class SABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = conv_3x3x3(dim, dim, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, C, T, H, W)
        return x    


class SplitSABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = conv_3x3x3(dim, dim, groups=dim)
        self.t_norm = norm_layer(dim)
        self.t_attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, C, T, H, W = x.shape
        attn = x.view(B, C, T, H * W).permute(0, 3, 2, 1).contiguous()
        attn = attn.view(B * H * W, T, C)
        attn = attn + self.drop_path(self.t_attn(self.t_norm(attn)))
        attn = attn.view(B, H * W, T, C).permute(0, 2, 1, 3).contiguous()
        attn = attn.view(B * T, H * W, C)
        residual = x.view(B, C, T, H * W).permute(0, 2, 3, 1).contiguous()
        residual = residual.view(B * T, H * W, C)
        attn = residual + self.drop_path(self.attn(self.norm1(attn)))
        attn = attn.view(B, T * H * W, C)
        out = attn + self.drop_path(self.mlp(self.norm2(attn)))
        out = out.transpose(1, 2).reshape(B, C, T, H, W)
        return out


class SpeicalPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        
        self.proj = nn.Sequential(
            nn.Conv3d(in_chans, embed_dim // 2, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(embed_dim // 2),
            nn.GELU(),
            nn.Conv3d(embed_dim // 2, embed_dim, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(embed_dim),
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x.reshape(B, T, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        return x
    

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = conv_1xnxn(in_chans, embed_dim, kernel_size=patch_size[0], stride=patch_size[0])

    def forward(self, x):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, T, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        return x


@MODEL_REGISTRY.register()
class Uniformer_light(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """
    def __init__(self, cfg):
        super().__init__()

        depth = cfg.UNIFORMER.DEPTH
        num_classes = cfg.MODEL.NUM_CLASSES 
        in_chans = cfg.DATA.INPUT_CHANNEL_NUM[0]
        embed_dim = cfg.UNIFORMER.EMBED_DIM
        head_dim = cfg.UNIFORMER.HEAD_DIM
        mlp_ratio = cfg.UNIFORMER.MLP_RATIO
        qkv_bias = cfg.UNIFORMER.QKV_BIAS
        qk_scale = cfg.UNIFORMER.QKV_SCALE
        representation_size = cfg.UNIFORMER.REPRESENTATION_SIZE
        drop_rate = cfg.UNIFORMER.DROPOUT_RATE
        attn_drop_rate = cfg.UNIFORMER.ATTENTION_DROPOUT_RATE
        drop_path_rate = cfg.UNIFORMER.DROP_DEPTH_RATE
        prune_ratio = cfg.UNIFORMER.PRUNE_RATIO
        trade_off = cfg.UNIFORMER.TRADE_OFF

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = partial(nn.LayerNorm, eps=1e-6) 
        
        self.patch_embed1 = SpeicalPatchEmbed(
            patch_size=4, in_chans=in_chans, embed_dim=embed_dim[0])
        self.patch_embed2 = PatchEmbed(
            patch_size=2, in_chans=embed_dim[0], embed_dim=embed_dim[1])
        self.patch_embed3 = PatchEmbed(
            patch_size=2, in_chans=embed_dim[1], embed_dim=embed_dim[2])
        self.patch_embed4 = PatchEmbed(
            patch_size=2, in_chans=embed_dim[2], embed_dim=embed_dim[3])

        # class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim[2]))
        self.cls_upsample = nn.Linear(embed_dim[2], embed_dim[3])

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        num_heads = [dim // head_dim for dim in embed_dim]
        self.blocks1 = nn.ModuleList([
            CBlock(
                dim=embed_dim[0], num_heads=num_heads[0], mlp_ratio=mlp_ratio[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])
        self.blocks2 = nn.ModuleList([
            CBlock(
                dim=embed_dim[1], num_heads=num_heads[1], mlp_ratio=mlp_ratio[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]], norm_layer=norm_layer)
            for i in range(depth[1])])
        self.blocks3 = nn.ModuleList([
            EvoSABlock(
                dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer,
                prune_ratio=prune_ratio[2][i], trade_off=trade_off[2][i],
                downsample=True if i == depth[2] - 1 else False)
            for i in range(depth[2])])
        self.blocks4 = nn.ModuleList([
            EvoSABlock(
                dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer,
                prune_ratio=prune_ratio[3][i], trade_off=trade_off[3][i])
        for i in range(depth[3])])
        self.norm = bn_3d(embed_dim[-1])
        self.norm_cls = nn.LayerNorm(embed_dim[-1])
        
        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()
        
        # Classifier head
        self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.head_cls = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
        
        self.apply(self._init_weights)

        for name, p in self.named_parameters():
            # fill proj weight with 1 here to improve training dynamics. Otherwise temporal attention inputs
            # are multiplied by 0*0, which is hard for the model to move out of.
            if 't_attn.qkv.weight' in name:
                nn.init.constant_(p, 0)
            if 't_attn.qkv.bias' in name:
                nn.init.constant_(p, 0)
            if 't_attn.proj.weight' in name:
                nn.init.constant_(p, 1)
            if 't_attn.proj.bias' in name:
                nn.init.constant_(p, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def inflate_weight(self, weight_2d, time_dim, center=False):
        if center:
            weight_3d = torch.zeros(*weight_2d.shape)
            weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
            middle_idx = time_dim // 2
            weight_3d[:, :, middle_idx, :, :] = weight_2d
        else:
            weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
            weight_3d = weight_3d / time_dim
        return weight_3d

    def get_pretrained_model(self, cfg):
        if cfg.UNIFORMER.PRETRAIN_NAME:
            checkpoint = torch.load(model_path[cfg.UNIFORMER.PRETRAIN_NAME], map_location='cpu')

            state_dict_3d = self.state_dict()
            for k in checkpoint.keys():
                if checkpoint[k].shape != state_dict_3d[k].shape:
                    if len(state_dict_3d[k].shape) <= 2:
                        logger.info(f'Ignore: {k}')
                        continue
                    logger.info(f'Inflate: {k}, {checkpoint[k].shape} => {state_dict_3d[k].shape}')
                    time_dim = state_dict_3d[k].shape[2]
                    checkpoint[k] = self.inflate_weight(checkpoint[k], time_dim)

            if self.num_classes != checkpoint['head.weight'].shape[0]:
                del checkpoint['head.weight'] 
                del checkpoint['head.bias'] 
                del checkpoint['head_cls.weight'] 
                del checkpoint['head_cls.bias'] 
            return checkpoint
        else:
            return None

    def forward_features(self, x):
        x = self.patch_embed1(x)
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)
        x = self.patch_embed3(x)
        # add cls_token in stage3
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        global global_attn, token_indices
        global_attn = 0
        token_indices = torch.arange(x.shape[2] * x.shape[3] * x.shape[4], dtype=torch.long, device=x.device).unsqueeze(0)
        token_indices = token_indices.expand(x.shape[0], -1)
        for blk in self.blocks3:
            cls_token, x = blk(cls_token, x)
        # upsample cls_token before stage4
        cls_token = self.cls_upsample(cls_token)
        x = self.patch_embed4(x)
        # whether reset global attention? Now simple avgpool
        token_indices = torch.arange(x.shape[2] * x.shape[3] * x.shape[4], dtype=torch.long, device=x.device).unsqueeze(0)
        token_indices = token_indices.expand(x.shape[0], -1)
        for blk in self.blocks4:
            cls_token, x = blk(cls_token, x)
        if self.training:
            # layer normalization for cls_token
            cls_token = self.norm_cls(cls_token)
        x = self.norm(x)
        x = self.pre_logits(x)
        return cls_token, x

    def forward(self, x):
        x = x[0]
        cls_token, x = self.forward_features(x)
        x = x.flatten(2).mean(-1)
        if self.training:
            x = self.head(x), self.head_cls(cls_token.squeeze(1))
        else:
            x = self.head(x)
        return x
