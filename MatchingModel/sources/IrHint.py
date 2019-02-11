import math

import torch
import torch.nn.functional as F
from torch import nn

from MatchingModel.sources.data_loader import MyDataInterator


class HiNT(nn.Module):
    def __init__(self, embeded_size, batch_size, para_size=100):
        """
        :param embeded_size: size of embedding vector
        :param batch_size: size of batch
        :param para_size: size of paragraph
        """

        super(HiNT, self).__init__()
        self.embedded_size = embeded_size
        self.batch_size = batch_size
        self.para_size = para_size

        self.my_iterator = MyDataInterator(batch_size=batch_size)

        self.vocab_len = self.my_iterator.src_vocab_len

        self.embedded_layer = nn.Embedding(num_embeddings=self.vocab_len,
                                           embedding_dim=embeded_size,
                                           padding_idx=1)


    def create_matrix(self, quer, doc):
        """
        :param quer: tensor with size (max_len_sent)
        :param para: tensor with size (max_len_para)
        :return: list of cosine simmilarity, xor from quer and doc
        """
        len_query = quer.size(0)
        doc = doc.size(0)
        num_para = math.floor(doc/self.para_size)

        # split paragrah
        para_list = list()
        for i in range(num_para) - 1 :
            # process para here
            pass



        # xor_matrix = torch.LongTensor(len_query, len_para)
        # cos_matrix = torch.Tensor(len_query, len_para)

    # process on single query, para pair
    def xor_cos_matrix(self, query, para, que_embed, para_embed):
        query_len = query.size(-1)
        para_len = para.size(-1)

        xor_matrix = torch.LongTensor(query_len, para_len)
        cos_matrix = torch.Tensor(query_len, para_len)

        for i in range(query_len):
            for j in range(para_len):
                xor_matrix[i, j] = 1 \
                    if query[i] == para[j] else 0
                cos_matrix[i, j] = F.cosine_similarity\
                    (que_embed[i], para_embed[j], dim=0)


        return xor_matrix, cos_matrix



def main():
    hi = HiNT(100, 50)
    a = torch.LongTensor([1,2,3,4,5,6,7,8])
    b = torch.LongTensor([1,2,3,4,5,6,7,8, 9, 10])

    a_e = hi.embedded_layer(a)
    b_e = hi.embedded_layer(b)

    xor, cos = hi.xor_cos_matrix(a, b, a_e, b_e)

    print(xor)
    print(cos)



if __name__ == '__main__':
    main()


