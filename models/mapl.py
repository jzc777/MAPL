import torch.nn as nn
from .decoder import Decoder
from .msff import MSFF
from .Encoder import Encoder
# from .PseudoLabeler import PseudoLabeler
from .PL import PseudoLabeler

class MAPL(nn.Module):
    def __init__(self, memory_bank, pseudo_labeler, feature_extractor):
        super(MAPL, self).__init__()

        self.memory_bank = memory_bank
        self.pseudo_labeler = pseudo_labeler
        self.feature_extractor = feature_extractor
        self.msff = MSFF()
        self.decoder = Decoder()


    def forward(self, inputs):
        # 使用自定义Encoder提取特征
        features = self.feature_extractor(inputs)
        # 特征图列表：[f1, f2, f3, f4, f5]

        f_in = features[0]
        f_out = features[-1]
        f_ii = features[1:-1]

        concat_features = self.memory_bank.select(features = f_ii)

        # 使用MSFF模块处理特征
        msff_outputs = self.msff(features = concat_features)

        # 使用Decoder生成预测掩码
        predicted_mask = self.decoder(
            encoder_output=f_out,
            concat_features=[f_in] + msff_outputs
        )

        return predicted_mask
