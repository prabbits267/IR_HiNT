import math
import sys
import traceback
import time as t

import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from torch import nn

from MatchingModel.sources.data_loader import MyDataInterator


class LocalMatching(nn.Module):
    def __init__(self, in_size, vocab_len, out_size=100, para_size=100):
        """
        :param in_size: size of embedding vector
        :param para_size: size of paragraph
        :param out_size: size of output vector from CNN
        :param vocab_len: size of vocabulay
        """
        super(LocalMatching, self).__init__()
        self.out_size = out_size
        self.embed_size = in_size
        self.para_size = para_size
        self.vocab_len = vocab_len

        self.embedded_layer = nn.Embedding(num_embeddings=vocab_len,
                                           embedding_dim=in_size,
                                           padding_idx=1)

        self.transform = nn.Linear(in_size, 1)

        cnn_out = int(out_size / 2)
        self.conv1 = nn.Conv2d(3, cnn_out, kernel_size=3)
        self.conv2 = nn.Conv2d(50, 40, kernel_size=3)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(5520, cnn_out)

    def convolute(self, input):
        x = F.relu(self.mp(self.conv1(input.unsqueeze(0))))
        x = F.relu(self.mp(self.conv2(x)))
        x = x.view(-1)
        x = self.fc(x)
        return F.log_softmax(x, dim=0)

    def con_sig(self, S_xor, S_cos):
        return torch.cat((self.convolute(S_xor),
                          self.convolute(S_cos)), 0)

    def create_matrix(self, quer, doc):
        """
        :param quer: tensor with size (max_len_sent)
        :param para: tensor with size (max_len_para)
        :return: list of cosine simmilarity, xor from quer and doc
        """
        len_query = quer.size(0)
        doc_size = doc.size(0)
        num_para = math.ceil(doc_size/self.para_size)

        # split paragrah
        xor_matrix = torch.zeros(num_para, len_query, self.para_size)
        cos_matrix = torch.zeros(num_para, len_query, self.para_size)

        Sxor_para = torch.Tensor(num_para, 3, len_query, self.para_size)
        Scos_para = torch.Tensor(num_para, 3, len_query, self.para_size)

        query_embed = self.embedded_layer(quer)
        doc_embed = self.embedded_layer(doc)

        for i in range(num_para):
            star_ind = i * self.para_size
            end_ind = star_ind + self.para_size
            if end_ind > doc_size:
                end_ind = doc_size
            xor_matrix[i], cos_matrix[i], Sxor_para[i], Scos_para[i]\
                = self.xor_cos_matrix(quer, doc[star_ind:end_ind], query_embed,
                                      doc_embed[star_ind:end_ind])

        return xor_matrix, cos_matrix, Sxor_para, Scos_para

    def xor_cos_matrix(self, query, para, que_embed, para_embed):
        """
        :param query: tensor of query
        :param para: tensor of paragraph
        :param que_embed: tensor of embedded query
        :param para_embed: tensor of embedded paragraph
        :return: M_xor, M_cos, S_xor, S_cos
        """
        query_len = query.size(0)
        para_len = para.size(0)

        if para_embed.size(0) < self.para_size:
            para_emb = torch.zeros(self.para_size, self.batch_size, self.embed_size)
            para_emb[:para_len] = para_embed
            para_embed = para_emb

        S_xor = torch.Tensor(3, query_len, self.para_size)
        S_cos = torch.Tensor(3, query_len, self.para_size)

        M_xor = torch.zeros(query_len, self.para_size)
        M_cos = torch.from_numpy(cdist(que_embed.squeeze(1).detach().numpy(),
                                       para_embed.squeeze(1).detach().numpy(),
                                       metric='cosine'))

        trans_x = self.transform(que_embed)[:, 0, 0]
        trans_y = self.transform(para_embed)[:, 0, 0]

        trans_x = torch.transpose(trans_x.repeat(self.para_size, 1), 0, 1)
        trans_y = trans_y.repeat(query_len, 1)

        for i, item in enumerate(query):
            ind = (para == item).nonzero()
            if ind.size(0) != 0:
                for j in ind:
                    M_xor[i, j] = 1

        S_xor[0] = trans_x
        S_xor[1] = trans_y
        S_xor[2] = M_xor

        S_cos[0] = trans_x
        S_cos[1] = trans_y
        S_cos[2] = M_cos

        return M_xor, M_cos, S_xor, S_cos

    def forward(self, quer, doc):
        """
        :param quer: embededing of query
        :param doc:
        :return: a list of paragraph relevant signals according to query
        """
        _, _, Sxor_para, Scos_para = self.create_matrix(quer, doc)

        num_para = Sxor_para.size(0)
        doc_sig = torch.Tensor(num_para, self.out_size)
        for i in range(num_para):
            doc_sig[i] = self.con_sig(Sxor_para[i], Scos_para[i])
        return doc_sig

def main():
    pass
    # hi = LocalMatching(100)
    # for i in hi.my_iterator.train_iter:
    #     z = hi.cal_score(i.que, i.doc)


if __name__ == '__main__':
    main()


