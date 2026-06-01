import torch
import torch.nn as nn


data=[
  ("i like", "yo amo"),
  ("i love", "yo amo"),
  ("you like", "tu amas"),
]

source_vocab= {
    "<pad>": 0,
    "<start>": 1,
    "<end>": 2,
    "i": 3,
    "like": 4,
    "you": 5
}

target_vocab= {
    "<pad>": 0,
    "<start>": 1,
    "<end>": 2,
    "yo": 3,
    "amo": 4,
    "tu": 5
}

S= len(source_vocab)
embed_dim=50

embedding= nn.Embedding(S,embed_dim)


def tokenizer(text):

    return text.lower().split()

def encode(tokens,vocab):

    return [vocab[token] for token in tokens]


def process_pair(src, tgt, src_vocab, tgt_vocab):
    
    
    src_tokens = tokenizer(src)
    tgt_tokens = tokenizer(tgt)


    src_ids = encode(src_tokens, src_vocab)


    tgt_ids = encode(tgt_tokens, tgt_vocab)

    tgt_input_ids = [tgt_vocab["<start>"]] + tgt_ids
    tgt_output_ids = tgt_ids + [tgt_vocab["<end>"]]

    return src_ids, tgt_input_ids, tgt_output_ids



