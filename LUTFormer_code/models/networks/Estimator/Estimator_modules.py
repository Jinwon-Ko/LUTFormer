from models.networks.base import BaseModel, Three_Dimensional_LUT
from models.networks.Estimator.Estimator_blocks import *


# ========================== Networks ==========================
class LookUpTable_Estimator(Three_Dimensional_LUT):
    def __init__(self, cfg):
        super(LookUpTable_Estimator, self).__init__(cfg)

        self.downsample = DownSample(cfg)
        self.feature_extractor = Feature_extractor(cfg)
        self.generate_lut = LUT_token_update(cfg)

    def freeze_params_(self):
        for name, p in self.named_parameters():
            p.requires_grad = False

    def forward(self, img):
        resized_img = self.downsample(img)                              # [b, 3, h, w]
        image_tokens = self.feature_extractor(resized_img)              # [b, c, h//8, w//8]

        identity = self.identity[None].repeat(len(img), 1, 1, 1, 1)     # [b, 3, N, N, N]
        LUT = self.generate_lut(identity, image_tokens)                 # [b, 3, N, N, N]

        enhanced = self.TrilinearInterpolation(img, LUT)
        return {'LUT': LUT, 'enhanced': enhanced}


# ========================== Encoder ==========================
class DownSample(nn.Module):
    def __init__(self, cfg):
        super(DownSample, self).__init__()
        if cfg.dataset_name == 'PPR10K':
            self.downsample = nn.Upsample(size=(224, 224), mode='bilinear')
        else:
            self.downsample = nn.Upsample(size=(256, 256), mode='bilinear')

    def forward(self, x):
        return self.downsample(x)


class Feature_extractor(nn.Module):
    def __init__(self, cfg):
        super(Feature_extractor, self).__init__()

        dim = cfg.network['hidden_dim']
        if cfg.dataset_name == 'PPR10K':
            self.down_block = ResNet18_based_backbone(dim, pretrained=True)
        else:
            self.down_block = Backbone(dim, n_down=cfg.network['n_down_layer'])

        self.enc_block = nn.ModuleList()
        for level in range(cfg.network['n_attn_module']):
            n_enc_block = cfg.network['n_enc_block'][level]
            n_enc_heads = cfg.network['n_enc_heads'][level]
            self.enc_block.append(EncoderBlock(dim, n_enc_block, n_enc_heads, pre_norm=True))

    def forward(self, x):
        feat = self.down_block(x)
        pixel_tokens = [feat]

        for enc_block in self.enc_block:
            feat = enc_block(feat)
            pixel_tokens.append(feat)

        return pixel_tokens


# ========================== Decoder ==========================
class LUT_token_update(nn.Module):
    def __init__(self, cfg):
        super(LUT_token_update, self).__init__()

        dim = cfg.network['hidden_dim']

        self.inc = nn.Sequential(nn.Conv3d(3, dim, kernel_size=1),
                                 nn.GELU(),
                                 nn.Conv3d(dim, dim, kernel_size=1))

        self.start_block = DecoderBlock(dim, n_block=cfg.network['n_dec_block'][0], n_heads=cfg.network['n_dec_heads'][0], pre_norm=True)
        self.dec_block = nn.ModuleList()
        for level in range(cfg.network['n_attn_module']):
            n_dec_block = cfg.network['n_dec_block'][level]
            n_dec_heads = cfg.network['n_dec_heads'][level]
            self.dec_block.append(DecoderBlock(dim, n_dec_block, n_dec_heads, pre_norm=True))

        self.outc = nn.Sequential(nn.Conv3d(dim, dim, kernel_size=1),
                                  nn.GELU(),
                                  nn.Conv3d(dim, 3, kernel_size=1))

    def forward(self, LUT_identity, image_tokens):
        LUT_token = self.inc(LUT_identity)
        LUT_token = self.start_block(LUT_token, image_tokens[0], image_tokens[0])
        for dec_block, token_k, token_v in zip(self.dec_block, image_tokens[:-1], image_tokens[1:]):
            LUT_token = dec_block(LUT_token, token_k, token_v)
        LUT = self.outc(LUT_token) + LUT_identity
        return LUT
