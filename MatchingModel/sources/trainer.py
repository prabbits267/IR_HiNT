from torch.nn import MultiLabelMarginLoss
from torch.optim import Adam

from MatchingModel.sources.data_loader import MyDataInterator
from MatchingModel.sources.ir_hint import IrHiNT


class Trainer:
    def __init__(self, in_size, hid_size, glo_dec, epoch, batch_size, para_size=100, learn_rate=0.001):
        # in_size, hid_size, glo_dec, epoch, para_size=100, learn_rate=0.001
        self.epoch = epoch
        self.batch_size = batch_size
        self.data_loader = MyDataInterator()
        vocab_size = self.data_loader.src_vocab_len
        self.train_iter = self.data_loader.train_iter

        self.model = IrHiNT(in_size, hid_size, glo_dec, epoch,
                            vocab_size=vocab_size, para_size=para_size)
        self.criterion = MultiLabelMarginLoss()
        self.optimizer = Adam(self.model.parameters(), lr=learn_rate)

    def train(self):
        for i in range(self.epoch):
            self.optimizer.zero_grad()
            for j, pair in enumerate(self.train_iter):
                score = self.model(pair.que, pair.doc)
                query, doc = next(iter(self.train_iter))
                score_random = self.model(query, doc)
                loss = self.criterion(score, score_random)
                loss.backward()

                if (j + 1) % self.batch_size == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

def main():
    # in_size, hid_size, glo_dec, epoch, batch_size, para_size=100, learn_rate=0.001
    trainer = Trainer(100, 256, )