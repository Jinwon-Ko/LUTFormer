import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ========================== Attention blocks ==========================
class EncoderBlock(nn.Module):
    def __init__(self, dim, n_block, n_heads, pre_norm=True):
        super(EncoderBlock, self).__init__()
        self.self_attention = nn.ModuleList()
        self.feed_forward = nn.ModuleList()
        for _ in range(n_block):
            self.self_attention.append(Spatial_Self_Attention(dim, n_heads, pre_norm=pre_norm))
            self.feed_forward.append(Feed_Forward_2D(dim, pre_norm=pre_norm))

    def forward(self, x):
        for self_attention, feed_forward in zip(self.self_attention, self.feed_forward):
            x = x + self_attention(x)
            x = x + feed_forward(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, dim, n_block, n_heads, pre_norm=True):
        super(DecoderBlock, self).__init__()
        diff_src_kv = 1
        same_src_kv = n_block - 1

        self.cross_attention1 = nn.ModuleList()
        self.feed_forward1 = nn.ModuleList()
        for _ in range(diff_src_kv):
            self.cross_attention1.append(LUT_Cross_Attention(dim, n_heads, pre_norm=pre_norm))
            self.feed_forward1.append(Feed_Forward_3D(dim, pre_norm=pre_norm))

        self.cross_attention2 = nn.ModuleList()
        self.feed_forward2 = nn.ModuleList()
        for _ in range(same_src_kv):
            self.cross_attention2.append(LUT_Cross_Attention(dim, n_heads, pre_norm=pre_norm))
            self.feed_forward2.append(Feed_Forward_3D(dim, pre_norm=pre_norm))

    def forward(self, src_q, src_k, src_v):
        for cross_attention, feed_forward in zip(self.cross_attention1, self.feed_forward1):
            src_q = src_q + cross_attention(src_q, src_k, src_v)
            src_q = src_q + feed_forward(src_q)

        for cross_attention, feed_forward in zip(self.cross_attention2, self.feed_forward2):
            src_q = src_q + cross_attention(src_q, src_v, src_v)
            src_q = src_q + feed_forward(src_q)
        return src_q


# ========================== Encoder blocks ==========================
class Spatial_Self_Attention(nn.Module):
    def __init__(self, dim, n_heads=1, pre_norm=True):
        super(Spatial_Self_Attention, self).__init__()
        assert dim % n_heads == 0

        self.pre_norm = pre_norm
        if pre_norm:
            self.norm_feat = nn.LayerNorm(dim)

        self.n_heads = n_heads
        self.depth = dim // n_heads

        self.tau = nn.Parameter(torch.zeros(n_heads, 1, 1))
        self.qkv_embed = nn.Linear(dim, dim * 3, bias=False)
        self.linear = nn.Linear(dim, dim)

    def forward(self, feat):
        ori_shape = feat.shape
        feat = feat.flatten(2).permute(0, 2, 1)
        if self.pre_norm:
            feat = self.norm_feat(feat)

        # 1. QKV Embed
        qkv = self.qkv_embed(feat)
        q, k, v = qkv.chunk(chunks=3, dim=-1)  # [b, h*w, c]

        # 2. Split heads
        q_feat = self.split_heads(q)  # [b, n, h*w, d]
        k_feat = self.split_heads(k)  # [b, n, h*w, d]
        v_feat = self.split_heads(v)  # [b, n, h*w, d]

        # 3. Self Attention
        self_attn_weight = F.softmax((q_feat @ k_feat.transpose(-2, -1)) * torch.exp(self.tau), dim=-1)  # [b, n, h*w, h*w]
        res = (self_attn_weight @ v_feat)  # [b, n, h*w, d]
        res = self.merge_heads(res)  # [b, h*w, c]
        res = self.linear(res)  # [b, h*w, c]
        res = res.permute(0, 2, 1).reshape(ori_shape)  # [b, c, h, w]
        return res

    def split_heads(self, x):
        # [b, n_token, c] -> [b, n_heads, n_token, d]
        return x.view(len(x), -1, self.n_heads, self.depth).permute(0, 2, 1, 3)

    def merge_heads(self, x):
        # [b, n_heads, n_token, d] -> [b, n_token, c]
        return x.permute(0, 2, 1, 3).reshape(len(x), -1, self.n_heads * self.depth)


class Feed_Forward_2D(nn.Module):
    def __init__(self, dim, pre_norm=True):
        super(Feed_Forward_2D, self).__init__()

        self.pre_norm = pre_norm
        if pre_norm:
            self.norm = nn.GroupNorm(1, dim)  # equivalent with LayerNorm

        feedforward_dim = dim * 4
        self.body = nn.Sequential(
            nn.Conv2d(dim, feedforward_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(feedforward_dim, dim, kernel_size=1))

    def forward(self, feat):
        if self.pre_norm:
            feat = self.norm(feat)
        return self.body(feat)


# ========================== Decoder blocks ==========================
class LUT_Cross_Attention(nn.Module):
    def __init__(self, dim, n_heads=1, pre_norm=True):
        super(LUT_Cross_Attention, self).__init__()
        assert dim % n_heads == 0

        self.pre_norm = pre_norm
        if pre_norm:
            self.norm_q = nn.LayerNorm(dim)
            self.norm_k = nn.LayerNorm(dim)
            self.norm_v = nn.LayerNorm(dim)

        self.n_heads = n_heads
        self.depth = dim // n_heads

        self.tau = nn.Parameter(torch.zeros(n_heads, 1, 1))
        self.q_embed = nn.Linear(dim, dim, bias=False)
        self.k_embed = nn.Linear(dim, dim, bias=False)
        self.v_embed = nn.Linear(dim, dim, bias=False)
        self.linear = nn.Linear(dim, dim)

    def forward(self, LUT_token, color_k, color_v):
        ori_shape = LUT_token.shape
        LUT_token = LUT_token.flatten(2).permute(0, 2, 1)   # [b, M, c]
        color_k = color_k.flatten(2).permute(0, 2, 1)       # [b, h*w, c]
        color_v = color_v.flatten(2).permute(0, 2, 1)       # [b, h*w, c]

        if self.pre_norm:
            LUT_token = self.norm_q(LUT_token)
            color_k = self.norm_k(color_k)
            color_v = self.norm_v(color_v)

        # 1. QKV Embed
        q_LUT = self.q_embed(LUT_token)     # [b, M, c]
        k_feat = self.k_embed(color_k)      # [b, h*w, c]
        v_feat = self.v_embed(color_v)      # [b, h*w, c]
        q_LUT = self.split_heads(q_LUT)     # [b, nh, M, d]
        k_feat = self.split_heads(k_feat)   # [b, nh, h*w, d]
        v_feat = self.split_heads(v_feat)   # [b, nh, h*w, d]

        # 2. Cross Attention
        cross_attn_affinity = q_LUT @ k_feat.transpose(-2, -1)
        cross_attn_weight = F.softmax(cross_attn_affinity * torch.exp(self.tau), dim=-1)  # [b, nh, M, h*w]
        res = (cross_attn_weight @ v_feat)  # [b, nh, M, d]
        res = self.merge_heads(res)         # [b, M, c]
        res = self.linear(res)
        res = res.permute(0, 2, 1).reshape(ori_shape)

        # (3. optional for debug or viz)
        self.cross_attention_map = F.softmax(cross_attn_affinity, dim=-1).squeeze(1)
        return res

    def split_heads(self, x):
        # [b, n_token, c] -> [b, n_heads, n_token, d]
        return x.view(len(x), -1, self.n_heads, self.depth).permute(0, 2, 1, 3)

    def merge_heads(self, x):
        # [b, n_heads, n_token, d] -> [b, n_token, c]
        return x.permute(0, 2, 1, 3).reshape(len(x), -1, self.n_heads * self.depth)


class Feed_Forward_3D(nn.Module):
    def __init__(self, dim, pre_norm=True):
        super(Feed_Forward_3D, self).__init__()

        self.pre_norm = pre_norm
        if pre_norm:
            self.norm = nn.GroupNorm(1, dim)  # equivalent with LayerNorm

        hidden_dim = dim * 4
        self.body = nn.Sequential(
            nn.Conv3d(dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden_dim, dim, kernel_size=1))

    def forward(self, LUT_token):
        if self.pre_norm:
            LUT_token = self.norm(LUT_token)
        return self.body(LUT_token)


# ========================== Downsample methods ==========================
class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=1, norm=False):
        super(ConvBlock, self).__init__()
        layer = [nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding),
                 nn.GELU()]
        if norm:
            layer += [nn.GroupNorm(out_dim, out_dim)]  # equivalent with InstanceNorm
        self.layer = nn.Sequential(*layer)

    def forward(self, x):
        return self.layer(x)


class Backbone(nn.Module):
    def __init__(self, dim, n_down):
        super(Backbone, self).__init__()
        self.inc = nn.Sequential(nn.Conv2d(3, dim, kernel_size=1),
                                 nn.GELU(),
                                 nn.Conv2d(dim, dim, kernel_size=1))

        layer = [nn.Sequential(ConvBlock(dim, dim, kernel_size=3, stride=2, padding=1, norm=True),
                               ConvBlock(dim, dim, kernel_size=1, stride=1, padding=0, norm=True))
                 for _ in range(n_down - 1)]
        layer += [nn.Sequential(ConvBlock(dim, dim, kernel_size=3, stride=2, padding=1, norm=True),
                                nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0))]

        self.layer = nn.Sequential(*layer)

    def forward(self, x):
        feat = self.layer(self.inc(x))
        return feat


class ResNet18_based_backbone(nn.Module):
    def __init__(self, dim, pretrained=True):
        super(ResNet18_based_backbone, self).__init__()
        resnet18 = models.resnet18(pretrained=pretrained)
        self.layer = nn.Sequential(*list(resnet18.children())[:6])      # [H//8, W//8]
        self.channel_fitting = nn.Conv2d(128, dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.channel_fitting(self.layer(x))
