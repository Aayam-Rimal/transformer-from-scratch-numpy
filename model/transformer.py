import numpy as np
from encoder import encoder_block
from decoder import decoder_block

class EncoderStack:

    def __init__(self,layers):
        self.layers= layers
        
    def forward(self,src):

        x=src

        for layer in self.layers:
            x= layer.forward(src)

        return x
    

class DecoderStack:

    def __init__(self,layers):
        self.layers= layers

    def forward(self,trgt,enc_out):

        x=trgt

        for layer in self.layers:
            x= layer.forward(trgt,enc_out)

        return x 


class Transformer:

    def __init__(self,encoder,decoder):

        self.encoder= encoder
        self.decoder= decoder

    def forward(self,src,trgt):

        enc_out= self.encoder.forward(src)
        dec_out= self.decoder.forward(trgt, enc_out)

        return dec_out
    


if __name__=="__main__":

    src= np.random.randn(1,5,5)
    trgt= np.random.randn(1,5,5)

    encoder_list= [encoder_block(5,1,10,eps=1e-5) for _ in range(5)]
    decoder_list= [decoder_block(5,1,10,eps=1e-5) for _ in range(5)]

    encoder= EncoderStack(encoder_list)
    decoder= DecoderStack(decoder_list)

    model= Transformer(encoder,decoder)

    output= model.forward(src,trgt)





        


    






    

