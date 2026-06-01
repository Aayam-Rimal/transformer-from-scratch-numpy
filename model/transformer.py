import numpy as np
from encoder import encoder_block
from decoder import decoder_block

class EncoderStack:

    def __init__(self,layers):

        self.layers= layers
        
    def forward(self,x):

        for layer in self.layers:
            x= layer.forward(x)
        return x
    

class DecoderStack:

    def __init__(self,layers):
        
        self.layers= layers

    def forward(self,x):

        for layer in self.layers:
            x= layer.forward(x)

        return x 



    






    

