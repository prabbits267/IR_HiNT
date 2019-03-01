import torch
from torch import nn


class GlobalDecision(nn.Module):
    def __init__(self, in_size, hid_size, out_size=100, glo_dec='hybrid', n_layers=1):
        """
        :param in_size: input size
        :param glo_dec: type of global decision model, default 'hybrid'
        :param hid_size: hidden size of LSTM
        :param n_layers number of layers
        """
        super(GlobalDecision, self).__init__()
        self.glo_dec = glo_dec
        self.in_size = in_size
        self.hid_size = hid_size
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_size=in_size,
                            hidden_size=hid_size,
                            num_layers=n_layers,
                            batch_first=True)

        self.transform = nn.Linear(self.hid_size, out_size)

        if glo_dec not in ['independent', 'accumulate', 'hybrid']:
            raise Exception('The global decision layer is wrong')
        self.local_trans = nn.Linear(in_size, in_size)
        self.out = nn.Linear(in_size, 1)

    def forward(self, local_sigs):
        if self.glo_dec == 'independent':
            return self.cal_indscore(local_sigs)
        if self.glo_dec == 'accumulate':
            return self.cal_accscore(local_sigs)
        return self.cal_hybscore(local_sigs)

    def cal_indscore(self, local_sigs):
        top_sigs = local_sigs.topk(1, dim=0)[0].squeeze(0)
        return self.out(top_sigs)

    def cal_accscore(self, local_sigs):
        lstm_out, hidden = self.lstm(local_sigs.unsqueeze(0))
        top_sigs = lstm_out.squeeze(0).topk(1, dim=0)[0]
        return self.out(top_sigs)

    def cal_hybscore(self, local_sigs):
        local_sigs = torch.tanh(self.local_trans(local_sigs))
        lstm_out, hidden = self.lstm(local_sigs.unsqueeze(0))
        lstm_sig = self.transform(lstm_out.squeeze(0))
        concat_sigs = torch.cat((lstm_sig, local_sigs), dim=0)
        top_sigs = concat_sigs.topk(1, dim=0)[0]
        return self.out(top_sigs)


def main():
    # in_size, hid_size, glo_dec, n_layers
    glo = GlobalDecision(100, 100, 'hybrid')
    sigs = torch.randn(4, 100)
    score = glo(sigs)
    print(score)


if __name__ == '__main__':
    main()
