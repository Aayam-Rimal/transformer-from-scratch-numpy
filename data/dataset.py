from vocab import process_pair
import numpy as np


class Dataset:

    def __init__(self, data_pair,src_vocab,trgt_vocab):

        self.data_pair= data_pair
        self.src_vocab= src_vocab
        self.trgt_vocab= trgt_vocab

    
    def __len__(self):
        return len(self.data_pair)
    

    def __getitem__(self, idx):
        
        src,trgt= self.data_pair[idx]
        return process_pair(src, trgt, self.src_vocab, self.trgt_vocab)
    