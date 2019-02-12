import math
import sys
import traceback

import torch
import torch.nn.functional as F
from torch import nn

from MatchingModel.sources.data_loader import MyDataInterator


class LocalMatching(nn.Module):
    def __init__(self, embed_size, batch_size, para_size=100):
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
        xor_matrix = torch.zeros(num_para, len_query, self.para_size).long()
        cos_matrix = torch.zeros(num_para, len_query, self.para_size)

        S_xor = torch.Tensor(num_para, 3, len_query, self.para_size)
        S_cos = torch.Tensor(num_para, 3, len_query, self.para_size)

        query_embed = self.embedded_layer(quer)
        doc_embed = self.embedded_layer(doc)
        for i in range(num_para - 1):
            # process para here
            index = i * self.para_size
            xor_matrix[i], cos_matrix[i], S_xor[i], S_cos[i] = self.xor_cos_matrix(quer, doc[index:index+self.para_size],
                                                     query_embed, doc_embed[index:index+self.para_size])

        index = (num_para - 1) * self.para_size
        xor_matrix[-1, :, :(doc_size - index)], cos_matrix[-1, :, :(doc_size - index)], \
                        S_xor[-1, :, :, :doc_size - index], S_cos[-1, :, :, :doc_size - index] = \
                        self.xor_cos_matrix(quer, doc[index:doc_size], query_embed, doc_embed[index:doc_size])
        return xor_matrix, cos_matrix

    # process on single query, para pair
    def xor_cos_matrix(self, query, para, que_embed, para_embed):
        query_len = query.size(0)
        para_len = para.size(0)

        S_xor = torch.Tensor(3, query_len, para_len)
        S_cos = torch.Tensor(3, query_len, para_len)

        M_xor = torch.LongTensor(query_len, para_len).long()
        M_cos = torch.Tensor(query_len, para_len)

        trans_x = self.transform(que_embed)[:, 0, 0]
        trans_y = self.transform(para_embed)[:, 0, 0]

        for i in range(query_len):
            for j in range(para_len):
                M_xor[i, j] = 1 \
                    if query[i] == para[j] else 0
                M_cos[i, j] = F.cosine_similarity\
                    (que_embed[i], para_embed[j], dim=1)

                S_xor[0, i, j] = trans_x[i]
                S_xor[1, i, j] = trans_y[j]
                S_xor[2, i, j] = M_xor[i, j]

                S_cos[0, i, j] = trans_x[i]
                S_cos[1, i, j] = trans_y[j]
                S_cos[2, i, j] = M_cos[i, j]

        return M_xor, M_cos, S_xor, S_cos

def main():
    hi = LocalMatching(100, 1)
    for i in hi.my_iterator.train_iter:
        M_xor, M_cos = hi.create_matrix(i.summ, i.source)
        print(M_xor)
        print(M_cos)




if __name__ == '__main__':
    main()


