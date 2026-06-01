import torch.nn as nn
from vocab import process_pair
import numpy as np

class DataLoader:
    
    def __init__(self,batch_size,dataset,shuffle):

        self.batch_size= batch_size
        self.shuffle= shuffle
        self.dataset= dataset
        self.indices= np.arange(len(dataset))


    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        for i in range(0, len(self.indices), self.batch_size):

            batch_idx= self.indices[i:i+self.batch_size]

            batch= [self.dataset[j] for j in batch_idx]

            yield self.collate(batch)
            

    def pad_sequence(self, seq, max_len, pad_value=0):
        return seq + [pad_value] * (max_len- len(seq))

    
    def collate(self, batch):

        src,tgt_inp, tgt_out= zip(*batch)

        src_max= max(len(x) for x in src)
        tgt_max= max(len(x) for x in tgt_inp)

        src= [self.pad_sequence(x, src_max) for x in src]
        tgt_inp= [self.pad_sequence(x, tgt_max) for x in tgt_inp]
        tgt_out= [self.pad_sequence(x, tgt_max) for x in tgt_out]


        return {
            "src": np.array(src),
            "tgt_in": np.array(tgt_inp),
            "tgt_out": np.array(tgt_out)
        }




