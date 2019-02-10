from torchtext.data import Field, BucketIterator, TabularDataset
from MatchingModel.sources.config import  *


class MyDataInterator:
    def __init__(self, batch_size):
        '''
        :param batch_size: batch_size
        '''

        self.batch_size = batch_size


        self.SRC_TEXT = Field(sequential=True,
                              tokenize=self.tokenizer,
                              use_vocab=True)
        self.TRG_TEXT = Field(sequential=True,
                              tokenize=self.tokenizer,
                              use_vocab=True)
        self.CLUSTER_TEXT = Field(sequential=False,
                                  use_vocab=False)

        self.data_fields = [('source', self.SRC_TEXT),
                            ('target', self.TRG_TEXT),
                            ('cluster', self.CLUSTER_TEXT)]

        self.train, self.val = TabularDataset.splits(path=PATH,
                                                     train=IR_TRAIN,
                                                     test=IR_TEST,
                                                     format='csv',
                                                     fields=self.data_fields)

        self.SRC_TEXT.build_vocab(self.train, self.val)
        self.TRG_TEXT.build_vocab()
        self.CLUSTER_TEXT.build_vocab()

        self.train_iter = BucketIterator(dataset=self.train,
                                         batch_size=self.batch_size,
                                         sort_key=lambda x: len(x.source),
                                         sort_within_batch=True,
                                         shuffle=True)

        self.test_iter = BucketIterator(dataset=self.val,
                                        batch_size=batch_size,
                                        sort_key=lambda x: len(x.sources),
                                        sort_within_batch=True,
                                        shuffle=False)

        self.src_vocab_len = self.SRC_TEXT.vocab.__len__()

    def tokenizer(self, sent):
        return sent.split()

    def word2token(self, word):
        try:
            return self.SRC_TEXT.vocab.stoi[word]
        except KeyError:
            return self.SRC_TEXT.void.stoi['<unk>']

    def token2word(self, token):
        try:
            return self.SRC_TEXT.vocab.itos[token]
        except IndexError:
            return '<unk>'



def main():
    iterator = MyDataInterator(1)
    for i in iterator.train_iter:
        # print(i.source)
        print(i.target)

if __name__ == '__main__':
    main()





