import re
from os.path import join

from pyvi.ViTokenizer import tokenize
from torchtext.data import Field, BucketIterator, TabularDataset, Pipeline
from MatchingModel.sources.config import  *


class MyDataInterator:
    def __init__(self, batch_size):
        '''
        :param batch_size: batch_size
        '''
        self.batch_size = batch_size

        self.stop_words = self.get_stopwords()

        self.SRC_TEXT = Field(sequential=True,
                              tokenize=self.tokenizer,
                              lower=True,
                              preprocessing=Pipeline(self.post_process),
                              stop_words=self.stop_words)
        self.TRG_TEXT = Field(sequential=True,
                              tokenize=self.tokenizer,
                              lower=True,
                              preprocessing=Pipeline(self.post_process),
                              stop_words=self.stop_words)

        self.data_fields = [("source", self.SRC_TEXT),
                            ("summ", self.TRG_TEXT)]

        self.train, self.val = TabularDataset.splits(path=PATH,
                                                     train=IR_TRAIN,
                                                     test=IR_TEST,
                                                     format='csv',
                                                     fields=self.data_fields)

        self.SRC_TEXT.build_vocab(self.train)
        self.TRG_TEXT.build_vocab(self.train)

        self.train_iter, self.test_iter = BucketIterator.splits(datasets=(self.train, self.val),
                                         batch_size=self.batch_size,
                                         sort_key=lambda x: len(x.source),
                                         sort_within_batch=False,
                                         shuffle=True,
                                         repeat=False)

        self.src_vocab_len = self.SRC_TEXT.vocab.__len__()

    def tokenizer(self, sent):
        # sent = tokenize(sent)
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

    def get_stopwords(self):
        with open(join(PATH, STOP_WORDS), 'rt',
                  encoding='utf-8') as file_reader:
            stopwords = file_reader.read().splitlines()
        return stopwords

    def post_process(self, sent):
        regex = re.compile('[^\w\d]')
        return regex.sub(' ', sent)




def main():
    iterator = MyDataInterator(1)
    for i in iterator.train_iter:
        print(i.source.size())

if __name__ == '__main__':
    main()





