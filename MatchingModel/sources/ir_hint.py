from torch import nn
from torch.nn import MultiLabelMarginLoss
from torch.optim import Adam

from MatchingModel.sources.data_loader import MyDataInterator
from MatchingModel.sources.global_decision import GlobalDecision
from MatchingModel.sources.local_matching import LocalMatching


class IrHiNT(nn.Module):
    def __init__(self, in_size, hid_size, glo_dec, vocab_size, para_size=100):
        """
        :param in_size: input size
        :param hid_size: hidden size
        :param para_size: paragraph size
        :param glo_dec: global decision model
        :param learn_rate learning rate
        """
        super(IrHiNT, self).__init__()

        self.in_size = in_size
        self.hid_size = hid_size
        self.para_size = para_size
        self.glo_dec = glo_dec

        self.local_matching = LocalMatching(in_size, vocab_size)
        self.global_decision = GlobalDecision(in_size, hid_size)

    def forward(self, quer, doc):
        local_sigs = self.local_matching(quer, doc)
        rel_score = self.global_decision(local_sigs)
        return rel_score


def main():
    # hid_size, out_size, para_size, glo_dec, epoch
    ir = IrHiNT(in_size=100, hid_size=256, glo_dec='hybrid', epoch=10)

    for i in ir.data_iter.train_iter:
        score = ir(i.que, i.doc)

    print(score)


if __name__ == '__main__':
    main()
