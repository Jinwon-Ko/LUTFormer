from models.networks.base import Three_Dimensional_LUT
from models.networks.Estimator.Estimator_modules import LookUpTable_Estimator


class LUTFormer(Three_Dimensional_LUT):
    def __init__(self, cfg):
        super(LUTFormer, self).__init__(cfg)
        self.LUT_estimator = LookUpTable_Estimator(cfg)

    def forward(self, img):
        out = self.LUT_estimator(img)
        return out


def get_model(cfg):
    model = LUTFormer(cfg)
    return model
