import numpy as np
from FFN import FFN
from LayerNorm import LN
from Attention import MHA

class decoder_block:

    def __init__(self, d_model, d_ff, num_heads, eps=1e-5):

        self.ffn= FFN(d_model,d_ff)
        self.ln1= LN(d_model, eps=1e-5)
        self.ln2= LN(d_model, eps=1e-5)
        self.ln3= LN(d_model, eps=1e-5)
        self.self_attn= MHA(d_model,num_heads)
        self.cross_attn= MHA(d_model,num_heads)

        self.eps= eps

    def forward(self,tgt,enc_out):

        B,S,D= tgt.shape

        mask= np.ones((S,S))
        mask= np.triu(mask, k=1)

        mask= mask * -1e9
        mask = mask[None, None, :, :]

        z= self.self_attn.forward(tgt,tgt,tgt,mask=mask)
        residual1= z + tgt
        ln1= self.ln1.norm(residual1)

        z2= self.cross_attn.forward(q=ln1, k=enc_out, v=enc_out, mask=None)

        x= ln1 + z2
        x = self.ln2.norm(x)

        z3 = self.ffn.forward(x)
        x = x + z3
        x = self.ln3.norm(x)

        return z,x
    

if __name__=="__main__":
    tgt= np.random.randn(1,4,6)
    encoder_out= np.random.randn(1,4,6)
    block1= decoder_block(6,12,1,eps=1e-5)

    z1,x1= block1.forward(tgt,encoder_out)

    print(z1)






    






