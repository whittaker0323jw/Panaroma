import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import Tensor
from torch.nn import init
from torch.nn.modules.utils import _pair
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mmcv.runner import load_checkpoint


class Trans4PASS(nn.Module):
    def __init__(self, num_classes=19, emb_chans=128, encoder='trans4pass_v1'):
        super().__init__()
        if encoder == 'trans4pass_v1':
            self.encoder = trans4pass_v1()
        elif encoder == 'trans4pass_v2':
            self.encoder = trans4pass_v2()
        else:
            raise NotImplementedError
        self.dede_head = DMLP(num_classes, emb_chans)
        self.__setattr__('decoder', ['dede_head'])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        size = x.size()[2:]
        c1, c2, c3, c4 = self.encoder(x)
        x_feats = [c1, c2, c3, c4]
        x_feats_de, x  = self.dede_head(c1, c2, c3, c4)
        x_out = F.interpolate(x, size, mode='bilinear', align_corners=True)
        # return x_feats, x_out
        return x_feats_de, x_out

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.encoder)
        b.append(self.dede_head)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        pass

    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.learning_rate},
                # {'params': self.get_10x_lr_params(), 'lr': 10 * args.learning_rate}
                ]


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
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
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x

class StackDilatedPatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 7.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=64, dilate=[1, 2]):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        padding = (patch_size[0] // 2, patch_size[1] // 2)
        padding = (padding[0] + (padding[0]+1) // 2, padding[1] + (padding[1]+1) // 2)
        self.projs = nn.ModuleList([nn.Conv2d(in_chans, embed_dim//2, kernel_size=patch_size, stride=stride,
                                              padding=(patch_size[0] // 2, patch_size[1] // 2)),
                                    nn.Conv2d(in_chans, embed_dim // 2, kernel_size=patch_size, stride=stride,
                                              padding=padding, dilation=dilate[1])])
        # ModuleList(
        #   (0): Conv2d(3, C//2, kernel_size=7, stride=(4, 4), padding=(2, 2), d=1)
        #   (1): Conv2d(3, C//2, kernel_size=7, stride=(4, 4), padding=(3, 3), d=2)
        # )
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        _, _, H, W = x.shape
        xs = []
        for i in range(len(self.projs)):
            tx = self.projs[i](x)
            _, _, H, W = tx.shape
            tx = tx.flatten(2).transpose(1, 2)
            xs.append(tx)  # B Ph*Pw C
        x = torch.cat(xs, dim=2)
        x = self.norm(x)
        return x, H, W


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768, use_dcn=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.kernel_size = patch_size[0]
        self.padding = patch_size[0] // 2#确保输入图像的边界能被卷积核完全覆盖
        self.norm = nn.LayerNorm(embed_dim)
        self.stride = to_2tuple(stride)
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.use_dcn = use_dcn
        # ==== define as same name, in order to load self.proj.
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                          padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.apply(self._init_weights)
        if use_dcn:
            self.offset_conv = nn.Conv2d(in_chans,
                                         2 * self.kernel_size * self.kernel_size,
                                         kernel_size=self.kernel_size,
                                         stride=stride,
                                         padding=self.padding)
            nn.init.constant_(self.offset_conv.weight, 0.)
            nn.init.constant_(self.offset_conv.bias, 0.)
            self.modulator_conv = nn.Conv2d(in_chans,
                                            1 * patch_size[0] * patch_size[0],
                                            kernel_size=self.kernel_size,
                                            stride=stride,
                                            padding=self.padding)
            nn.init.constant_(self.modulator_conv.weight, 0.)
            nn.init.constant_(self.modulator_conv.bias, 0.)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        if self.use_dcn:
            x = self.deform_proj(x)
        else:
            x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W

    def deform_proj(self, x):
        # h, w = x.shape[2:]
        max_offset = min(x.shape[-2], x.shape[-1]) // 4
        offset = self.offset_conv(x).clamp(-max_offset, max_offset)
        # print("Offset shape:", offset.shape)#Offset shape: torch.Size([1, 98, 100, 512])
        # print("Offset_x_test:",offset[0,:,0,0])

        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        # print("MODULATOR_test:",modulator[0,:,0,0])
        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.proj.weight,
                                          bias=self.proj.bias,
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          )
        return x


class Trans4PASS_Backbone(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        img_size = img_size[-1] if isinstance(img_size, tuple) else img_size
        USE_DCN = [True, False, False, False]
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0], use_dcn=USE_DCN[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1], use_dcn=USE_DCN[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2], use_dcn=USE_DCN[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3], use_dcn=USE_DCN[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained, map_location='cpu', strict=False)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


def trans4pass_v1(*args):
    return Trans4PASS_Backbone(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


def trans4pass_v2(*args):
    return Trans4PASS_Backbone(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class DWConv2d(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size=3, padding=1, bias=False):
        super(DWConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_chans, in_chans, kernel_size=kernel_size,
                                   padding=padding, groups=in_chans, bias=bias)
        self.pointwise = nn.Conv2d(in_chans, out_chans, kernel_size=1, bias=bias)

        nn.init.kaiming_uniform_(self.depthwise.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.pointwise.weight, a=math.sqrt(5))

    def forward(self, x):
        return self.pointwise(self.depthwise(x))



class DeformableProjEmbed(nn.Module):
    """ feature map to Projected Embedding
    """
    def __init__(self, in_chans=512, emb_chans=128):
        super().__init__()
        self.kernel_size = kernel_size = 3
        self.stride = stride = 1
        self.padding = padding = 1
        self.proj = nn.Conv2d(in_chans, emb_chans, kernel_size=kernel_size, stride=stride,
                              padding=padding)
        # --- deformable offset and modulator
        self.offset_conv = nn.Conv2d(in_chans, 2 * kernel_size * kernel_size, kernel_size=kernel_size,
                                     stride=stride, padding=padding)#固定的是卷积内部weight而不是offset
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        self.modulator_conv = nn.Conv2d(in_chans, 1 * kernel_size * kernel_size, kernel_size=kernel_size,
                                     stride=stride, padding=padding)
        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)
        self.norm = nn.BatchNorm2d(emb_chans)
        self.act = nn.GELU()

    def deform_proj(self, x):
        # h, w = x.shape[2:]
        max_offset = min(x.shape[-2], x.shape[-1]) // 4
        offset = self.offset_conv(x).clamp(-max_offset, max_offset)
        # print("Offset_decoder shape:", offset.shape)#Offset shape: torch.Size([1, 98, 100, 512])
        # print("Offset_decoder_test:",offset[0,:,0,0])
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.proj.weight,
                                          bias=self.proj.bias,
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          )
        return x

    def forward(self, x):
        x = self.deform_proj(x)
        x = self.act(self.norm(x))
        return x


class DeformableMLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super(DeformableMLP, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        if stride != 1:
            raise ValueError('stride must be 1')
        if padding != 0:
            raise ValueError('padding must be 0')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, 1, 1))  # kernel size == 1

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.offset_modulator_conv = DWConv2d(in_channels, 3 * in_channels)

        self.norm = nn.BatchNorm2d(in_channels)
        self.act = nn.GELU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        B, C, H, W = input.size()
        offset_modulator = self.offset_modulator_conv(input)
        offset_y, offset_x, modulator = torch.chunk(offset_modulator, 3, dim=1)
        modulator = 2. * torch.sigmoid(modulator)
        offset = torch.cat((offset_y, offset_x), dim=1)
        max_offset = max(H, W) // 4
        offset = offset.clamp(-max_offset, max_offset)
        x = torchvision.ops.deform_conv2d(input=input,
                                          offset=offset,
                                          weight=self.weight,
                                          bias=self.bias,
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          dilation=self.dilation
                                          )

        x = self.act(self.norm(x))
        return x

    def extra_repr(self) -> str:
        # s = self.__class__.__name__ + '('
        s = ''
        s += '{in_channels}'
        s += ', {out_channels}'
        s += ', kernel_size={kernel_size}'
        s += ', stride={stride}'
        s += ', padding={padding}' if self.padding != (0, 0) else ''
        s += ', dilation={dilation}' if self.dilation != (1, 1) else ''
        s += ', groups={groups}' if self.groups != 1 else ''
        s += ', bias=False' if self.bias is None else ''
        # s += ')'
        return s.format(**self.__dict__)


class DeformableMLPBlock(nn.Module):
    def __init__(self, in_chans=512, emb_chans=64):
        super().__init__()
        # spatial deformable proj
        self.sdp = DeformableProjEmbed(in_chans=in_chans, emb_chans=emb_chans)
        self.h_mlp = DeformableMLP(emb_chans, emb_chans, (1, 3), 1, 0)
        self.w_mlp = DeformableMLP(emb_chans, emb_chans, (3, 1), 1, 0)
        self.c_mlp = nn.Linear(emb_chans, emb_chans)
        self.proj = nn.Linear(emb_chans, emb_chans)

    def forward(self, x):
        x = self.sdp(x)
        # B, C, H, W = x.shape
        h = self.h_mlp(x).permute(0, 2, 3, 1)
        w = self.w_mlp(x).permute(0, 2, 3, 1)
        x = x.permute(0, 2, 3, 1)
        x = x + h + w
        c = self.c_mlp(x)
        x = x + c
        x = self.proj(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class DMLP(nn.Module):
    def __init__(self, num_classes=19, emb_chans=128):
        super().__init__()
        self.head1 = DeformableMLPBlock(in_chans=64, emb_chans=emb_chans)
        self.head2 = DeformableMLPBlock(in_chans=128, emb_chans=emb_chans)
        self.head3 = DeformableMLPBlock(in_chans=320, emb_chans=emb_chans)
        self.head4 = DeformableMLPBlock(in_chans=512, emb_chans=emb_chans)
        self.pred = nn.Conv2d(emb_chans, num_classes, 1)

    def forward(self, c1, c2, c3, c4):
        size = c1.size()[2:]
        c4 = self.head4(c4) # shape: B, 128, H/4, W/4
        c4 = F.interpolate(c4, size, mode='bilinear', align_corners=True)

        c3 = self.head3(c3)
        c3 = F.interpolate(c3, size, mode='bilinear', align_corners=True)

        c2 = self.head2(c2)
        c2 = F.interpolate(c2, size, mode='bilinear', align_corners=True)

        c1 = self.head1(c1)
        x_feats_de = [c1, c2, c3, c4]
        out = c1 + c2 + c3 + c4
        out = self.pred(out)
        return x_feats_de, out


def Trans4PASS_v1(num_classes=19, emb_chans=128):
    model = Trans4PASS(num_classes, emb_chans)
    return model

def Trans4PASS_v2(num_classes=19, emb_chans=128):
    model = Trans4PASS(num_classes, emb_chans, encoder='trans4pass_v2')
    return model