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
    def __init__(self, embed_size, batch_size=1, para_size=100):
        """
        :param embed_size: size of embedding vector
        :param batch_size: size of batch
        :param para_size: size of paragraph
        """

        super(LocalMatching, self).__init__()
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.para_size = para_size

        self.my_iterator = MyDataInterator(batch_size=batch_size)

        self.vocab_len = self.my_iterator.src_vocab_len

        self.embedded_layer = nn.Embedding(num_embeddings=self.vocab_len,
                                           embedding_dim=embed_size,
                                           padding_idx=1)

        self.transform = nn.Linear(embed_size, 1)

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

def main():
    hi = LocalMatching(100)
    for inz, i in enumerate(hi.my_iterator.train_iter):
        M_xor, M_cos, _, _ = hi.create_matrix(i.summ, i.source)
        print(inz)

if __name__ == '__main__':
    main()


