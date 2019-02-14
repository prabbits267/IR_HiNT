from torch import nn
from torch.nn import MultiLabelMarginLoss
from torch.optim import Adam

from MatchingModel.sources.data_loader import MyDataInterator
from MatchingModel.sources.global_decision import GlobalDecision
from MatchingModel.sources.local_matching import LocalMatching


class IrHiNT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, para_size, glo_dec, epoch, learn_rate=0.001):
        """
        :param in_size: input size
        :param hid_size: hidden size
        :param out_size: output size
        :param para_size: paragraph size
        :param glo_dec: global decision model
        :param learn_rate learning rate
        :param epoch number of epoch
        """
        super(IrHiNT, self).__init__()

        self.in_size = in_size
        self.hid_size = hid_size
        self.out_size = out_size
        self.para_size = para_size
        self.glo_dec = glo_dec
        self.learn_rate = learn_rate
        self.epoch = epoch

        data_iter = MyDataInterator()
        vocab_size = data_iter.src_vocab_len

        self.local_matching = LocalMatching(in_size, vocab_size, out_size, para_size)
        self.global_decision = GlobalDecision(in_size, hid_size, glo_dec)

        self.criterion = MultiLabelMarginLoss()

        self.ir_optim = Adam(self.parameters(), lr=learn_rate)

    def forward(self, quer, doc):
        local_sigs = self.local_matching(quer, doc)
        rel_score = self.global_decision(local_sigs)
        return rel_score