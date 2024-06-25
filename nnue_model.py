import torch as tc
import numpy as np
from torch import nn

#HalfKAv2 architecture

class Clip_Relu(nn.Module):

    def __init__(self, in_feats, out_feats):
        """
        Linear layer with Cliped Relu [0,1]
        """
        super().__init__()        
        self.linear = nn.Linear(in_feats, out_feats)
        
    def __call__(self, x):
        return tc.clamp(x, 0.0, 1.0)

class HalfKAv2(nn.Module): 

    def __init__(self, in_feats):
        """
        Implementation of HalfKAv2 algorithm based on Stockfish architecture
        """
        super().__init__()
        self.l1 = nn.Linear(in_feats, 257)
        self.cl_2 = Clip_Relu(512, 32)
        self.cl_3 = Clip_Relu(32, 32)
        self.l4 = nn.Linear(32, 1)

    def __call__( self, white, black):
        w1, b1 = self.l1(white), self.l1(black)
        out = tc.concat([w1[1:], b1[1:]], dim = 1)
        out = self.cl_2(out)
        out = self.cl_3(out)
        out = self.l4(out)

        res = (w1[0] - b1[0])/2

        return res + out
    

class Sparse_Linear(nn.Module):

    def __init__(self, in_feats):
        super().__init__()

    def __call__(self, x):
        pass

