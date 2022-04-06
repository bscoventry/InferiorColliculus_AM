# -*- coding: utf-8 -*-
"""
Created on Wed May 27 15:20:06 2015

@author: Alex
"""

def VCNtoVSgen2(freq):
    import numpy as np;
    
    freq = np.array(freq);
    vs=1/(2**np.absolute(np.log2((float(125)/freq))));
    vs = np.array(vs);

    return vs
    

